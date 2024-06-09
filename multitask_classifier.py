import argparse
import os
from pprint import pformat
import random
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert import BertModel
from datasets import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from evaluation import model_eval_multitask, test_model_multitask
from optimizer import AdamW
from datasets import preprocess_string
import random
import csv

TQDM_DISABLE = True


# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    """
    This module should use BERT for these tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    (- Paraphrase type detection (predict_paraphrase_types))
    """

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()

        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert parameters.
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True
        ### TODO
        # raise NotImplementedError
        self.sentiment_classifier = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.paraphrase_classifier = nn.Linear(BERT_HIDDEN_SIZE, 1)
        self.similarity_classifier = nn.Linear(BERT_HIDDEN_SIZE, 1)
        self.paraphrase_type_classifier = nn.Linear(BERT_HIDDEN_SIZE, 7)

        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask):
        """Takes a batch of sentences and produces embeddings for them."""

        # The final BERT embedding is the hidden state of [CLS] token (the first token).
        # See BertModel.forward() for more details.
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        # raise NotImplementedError
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

    def predict_sentiment(self, input_ids, attention_mask):
        """
        Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        Dataset: SST
        """
        ### TODO
        # raise NotImplementedError
        embeddings = self.forward(input_ids, attention_mask)
        sentiment_logits = self.sentiment_classifier(embeddings)
        return sentiment_logits

    def predict_paraphrase(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """
        Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        Dataset: Quora
        """
        ### TODO
        # raise NotImplementedError
        embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        embeddings_2 = self.forward(input_ids_2, attention_mask_2)
        combined_embeddings = torch.cat(
            [embeddings_1, embeddings_2, torch.abs(embeddings_1 - embeddings_2)], dim=-1
        )
        paraphrase_logits = self.paraphrase_classifier(combined_embeddings).squeeze(-1)
        return paraphrase_logits

    def predict_similarity(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """
        Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Since the similarity label is a number in the interval [0,5], your output should be normalized to the interval [0,5];
        it will be handled as a logit by the appropriate loss function.
        Dataset: STS
        """
        ### TODO
        # raise NotImplementedError
        embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        embeddings_2 = self.forward(input_ids_2, attention_mask_2)
        combined_embeddings = torch.cat(
            [embeddings_1, embeddings_2, torch.abs(embeddings_1 - embeddings_2)], dim=-1
        )
        similarity_logits = self.similarity_classifier(combined_embeddings).squeeze(-1)
        similarity_logits = torch.sigmoid(similarity_logits) * 5.0
        return similarity_logits

    def predict_paraphrase_types(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """
        Given a batch of pairs of sentences, outputs logits for detecting the paraphrase types.
        There are 7 different types of paraphrases.
        Thus, your output should contain 7 unnormalized logits for each sentence. It will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        Dataset: ETPC
        """
        ### TODO
        # raise NotImplementedError
        embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        embeddings_2 = self.forward(input_ids_2, attention_mask_2)
        combined_embeddings = torch.cat(
            [embeddings_1, embeddings_2, torch.abs(embeddings_1 - embeddings_2)], dim=-1
        )
        paraphrase_type_logits = self.paraphrase_type_classifier(combined_embeddings)
        return paraphrase_type_logits


def save_model(model, optimizer, args, config, filepath):
    # save the model
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "model_config": config,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"Saving the model to {filepath}.")


# TODO Currently only trains on SST dataset!
def train_multitask(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # Load data
    # Create the data and its corresponding datasets and dataloader:
    sst_train_data, _, quora_train_data, sts_train_data, etpc_train_data = (
        load_multitask_data(
            args.sst_train,
            args.quora_train,
            args.sts_train,
            args.etpc_train,
            split="train",
        )
    )
    sst_dev_data, _, quora_dev_data, sts_dev_data, etpc_dev_data = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="train"
    )

    sst_train_dataloader = None
    sst_dev_dataloader = None
    quora_train_dataloader = None
    quora_dev_dataloader = None
    sts_train_dataloader = None
    sts_dev_dataloader = None
    etpc_train_dataloader = None
    etpc_dev_dataloader = None

    # SST dataset
    if args.task == "sst" or args.task == "multitask":
        sst_train_data = SentenceClassificationDataset(sst_train_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_train_dataloader = DataLoader(
            sst_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sst_train_data.collate_fn,
        )
        sst_dev_dataloader = DataLoader(
            sst_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=sst_dev_data.collate_fn,
        )

    ### TODO
    #   Load data for the other datasets
    # If you are doing the paraphrase type detection with the minBERT model as well, make sure
    # to transform the the data labels into binaries (as required in the bart_detection.py script)

    # Init model
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "hidden_size": BERT_HIDDEN_SIZE,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
    }

    config = SimpleNamespace(**config)

    separator = "-" * 30
    print(separator)
    print("    BERT Model Configuration")
    print(separator)
    print(pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)}))
    print(separator)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = float("-inf")

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        if args.task == "sst" or args.task == "multitask":
            # Train the model on the sst dataset.

            for batch in tqdm(
                sst_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):
                b_ids, b_mask, b_labels = (
                    batch["token_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "sts" or args.task == "multitask":
            # Trains the model on the sts dataset
            ### TODO
            # raise NotImplementedError
            for batch in tqdm(
                sts_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
                )

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                loss = F.mse_loss(logits, b_labels.view(-1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "qqp" or args.task == "multitask":
            # Trains the model on the qqp dataset
            ### TODO
            # raise NotImplementedError
            for batch in tqdm(
                quora_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
                )

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                loss = F.binary_cross_entropy_with_logits(logits, b_labels.view(-1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "etpc" or args.task == "multitask":
            # Trains the model on the etpc dataset
            ### TODO
            # raise NotImplementedError
            for batch in tqdm(
                etpc_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
                )

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_paraphrase_types(
                    b_ids_1, b_mask_1, b_ids_2, b_mask_2
                )
                loss = F.binary_cross_entropy_with_logits(logits, b_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        train_loss = train_loss / num_batches

        (
            quora_train_acc,
            _,
            _,
            sst_train_acc,
            _,
            _,
            sts_train_corr,
            _,
            _,
            etpc_train_acc,
            _,
            _,
        ) = model_eval_multitask(
            sst_train_dataloader,
            quora_train_dataloader,
            sts_train_dataloader,
            etpc_train_dataloader,
            model=model,
            device=device,
            task=args.task,
        )

        (
            quora_dev_acc,
            _,
            _,
            sst_dev_acc,
            _,
            _,
            sts_dev_corr,
            _,
            _,
            etpc_dev_acc,
            _,
            _,
        ) = model_eval_multitask(
            sst_dev_dataloader,
            quora_dev_dataloader,
            sts_dev_dataloader,
            etpc_dev_dataloader,
            model=model,
            device=device,
            task=args.task,
        )

        train_acc, dev_acc = {
            "sst": (sst_train_acc, sst_dev_acc),
            "sts": (sts_train_corr, sts_dev_corr),
            "qqp": (quora_train_acc, quora_dev_acc),
            "etpc": (etpc_train_acc, etpc_dev_acc),
            "multitask": (0, 0),  # TODO
        }[args.task]

        print(
            f"Epoch {epoch+1:02} ({args.task}): train loss :: {train_loss:.3f}, train :: {train_acc:.3f}, dev :: {dev_acc:.3f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)


def test_model(args):
    with torch.no_grad():
        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
        saved = torch.load(args.filepath)
        config = saved["model_config"]

        model = MultitaskBERT(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        return test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()

    # Training task
    parser.add_argument(
        "--task",
        type=str,
        help='choose between "sst","sts","qqp","etpc","multitask" to train for different tasks ',
        choices=("sst", "sts", "qqp", "etpc", "multitask"),
        default="sst",
    )

    # Model configuration
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--option",
        type=str,
        help="pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated",
        choices=("pretrain", "finetune"),
        default="pretrain",
    )
    parser.add_argument("--use_gpu", action="store_true")

    args, _ = parser.parse_known_args()

    # Dataset paths
    parser.add_argument("--sst_train", type=str, default="data/sst-sentiment-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/sst-sentiment-dev.csv")
    parser.add_argument(
        "--sst_test", type=str, default="data/sst-sentiment-test-student.csv"
    )

    parser.add_argument(
        "--quora_train", type=str, default="data/quora-paraphrase-train.csv"
    )
    parser.add_argument(
        "--quora_dev", type=str, default="data/quora-paraphrase-dev.csv"
    )
    parser.add_argument(
        "--quora_test", type=str, default="data/quora-paraphrase-test-student.csv"
    )

    parser.add_argument(
        "--sts_train", type=str, default="data/sts-similarity-train.csv"
    )
    parser.add_argument("--sts_dev", type=str, default="data/sts-similarity-dev.csv")
    parser.add_argument(
        "--sts_test", type=str, default="data/sts-similarity-test-student.csv"
    )

    # TODO
    # You should split the train data into a train and dev set first and change the
    # default path of the --etpc_dev argument to your dev set.

    ## Done down in the main

    parser.add_argument(
        "--etpc_train", type=str, default="data/etpc-paraphrase-train.csv"
    )
    parser.add_argument("--etpc_dev", type=str, default="data/etpc-paraphrase-dev.csv")

    parser.add_argument(
        "--etpc_test",
        type=str,
        default="data/etpc-paraphrase-detection-test-student.csv",
    )

    # Output paths
    parser.add_argument(
        "--sst_dev_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sst_test_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-test-output.csv"
        ),
    )

    parser.add_argument(
        "--quora_dev_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--quora_test_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-test-output.csv"
        ),
    )

    parser.add_argument(
        "--sts_dev_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sts_test_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-test-output.csv"
        ),
    )

    parser.add_argument(
        "--etpc_dev_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--etpc_test_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-test-output.csv"
        ),
    )

    # Hyperparameters
    parser.add_argument(
        "--batch_size", help="sst: 64 can fit a 12GB GPU", type=int, default=64
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
        default=1e-3 if args.option == "pretrain" else 1e-5,
    )
    parser.add_argument("--local_files_only", action="store_true")

    args = parser.parse_args()
    return args


def etpc_dev(args):
    sst_data, num_labels, quora_data, sts_data, etpc_data = load_multitask_data(
        args.sst_train, args.quora_train, args.sts_train, args.etpc_train, split="train"
    )
    random.shuffle(etpc_data)
    train_size = int(len(etpc_data) * 0.75)
    etpc_data_train = etpc_data[:train_size]
    etpc_data_dev = etpc_data[train_size:]

    with open(args.etpc_train, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(
            [
                "sentence1",
                "sentence2",
                "paraphrase_types",
                "sentence1_segment_location",
                "sentence2_segment_location",
                "sentence1_tokenized",
                "sentence2_tokenized",
                "id",
            ]
        )
        for item in etpc_data_train:
            writer.writerow(item)

    print(
        f"Shuffled etpcs_data_train is saved to {args.etpc_train} with size {len(etpc_data_train)} and ratio {len(etpc_data_train)/len(etpc_data)}"
    )

    with open(args.etpc_dev, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(
            [
                "sentence1",
                "sentence2",
                "paraphrase_types",
                "sentence1_segment_location",
                "sentence2_segment_location",
                "sentence1_tokenized",
                "sentence2_tokenized",
                "id",
            ]
        )
        for item in etpc_data_dev:
            writer.writerow(item)

    print(
        f"Shuffled etpcs_data_train is saved to {args.etpc_dev} with size {len(etpc_data_dev)} and ratio {len(etpc_data_dev)/len(etpc_data)}"
    )

    # print(f"etpc_dev: {etpc_dev[0]}\n")
    # print(f"etpc_dev: {len(etpc_dev)}\n")

    # print(f"sst: {sst_data[0]}\n")
    # print(f"quora: {quora_data[0]}\n")
    # print(f"sts: {sts_data[0]}\n")
    # print(f"etpc: {etpc_data[0]}\n")


if __name__ == "__main__":
    args = get_args()
    # args.filepath = (
    #     f"models/{args.option}-{str(args.epochs)}-{str(args.lr)}-{args.task}.pt"  # save path
    # )
    seed_everything(args.seed)  # fix the seed for reproducibility
    etpc_dev(args)
    # train_multitask(args)
    # test_model(args)
