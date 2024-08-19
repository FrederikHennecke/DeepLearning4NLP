import argparse
import csv
import os
from pathlib import Path
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
from optimizer import AdamW, SophiaG
from contextlib import nullcontext
from bert import BertLayer
from transformers import get_linear_schedule_with_warmup


TQDM_DISABLE = False


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
N_HIDDEN_LAYERS = 12
HIDDEN_DIM = 128


class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(AttentionLayer, self).__init__()
        self.linear_transform = nn.Linear(input_size, input_size)
        self.linear_transform1 = nn.Linear(input_size, 1, bias=False)

    def forward(self, embeddings):
        # embeddings: [batch_size, seq_len, hidden_size]
        transformed_embeddings = torch.tanh(self.linear_transform(embeddings))
        attention_weights = torch.softmax(
            self.linear_transform1(transformed_embeddings), dim=1
        )
        attended_embeddings = torch.sum(attention_weights * embeddings, dim=1)
        return attended_embeddings


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
        self.config = config
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )

        # BILSTM layer
        self.bilstm = torch.nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        bilstm_output_dim = config.hidden_dim * 2

        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True

        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # attention layer
        self.attention_layer = AttentionLayer(config.hidden_size)

        if config.add_layers:
            # more layers
            self.sentiment_linear = torch.nn.Linear(
                bilstm_output_dim, config.hidden_size
            )
            self.sentiment_linear1 = torch.nn.Linear(
                config.hidden_size, config.hidden_size
            )
            self.sentiment_linear2 = torch.nn.Linear(
                config.hidden_size, bilstm_output_dim
            )
            self.sentiment_classifier = nn.Linear(
                bilstm_output_dim, N_SENTIMENT_CLASSES
            )
        else:
            self.sentiment_classifier = nn.Linear(
                bilstm_output_dim, N_SENTIMENT_CLASSES
            )

        self.paraphrase_classifier = nn.Linear(3 * bilstm_output_dim, 2)

    def forward(self, input_ids, attention_mask):
        """Takes a batch of sentences and produces embeddings for them."""

        batch_size, num_segments, segment_length = input_ids.size()
        input_ids = input_ids.view(batch_size * num_segments, segment_length)
        attention_mask = attention_mask.view(batch_size * num_segments, segment_length)

        _, pooler_output, all_sequences, _ = self.bert(input_ids, attention_mask)
        hidden_states = all_sequences["last_hidden_state"]
        bilstm_output, _ = self.bilstm(hidden_states)
        bilstm_output = self.dropout(bilstm_output)

        if self.config.pooling == None:
            pooled_output = bilstm_output

        elif self.config.pooling == "max":
            pooled_output, _ = torch.max(bilstm_output, dim=1)
            pooled_output = pooled_output.view(batch_size, num_segments, -1)
            pooled_output = torch.max(pooled_output, dim=1).values

            # pooled_output = nn.MaxPool1d(kernel_size=self.config.max_position_embeddings)(bilstm_output).squeeze(-1)

        elif self.config.pooling == "mean":
            pooled_output = torch.mean(bilstm_output, dim=1)
            # pooled_output = nn.AvgPool1d(kernel_size= self.config.max_position_embeddings)(bilstm_output).squeeze(-1)

        return pooled_output

    def predict_sentiment(self, input_ids, attention_mask):
        """
        Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        Dataset: SST
        """
        sentiment_logits = self.forward(input_ids, attention_mask)
        if self.config.add_layers:
            sentiment_logits = F.relu(self.sentiment_linear(sentiment_logits))
            sentiment_logits = F.relu(self.sentiment_linear1(sentiment_logits))
            sentiment_logits = F.relu(self.sentiment_linear2(sentiment_logits))
            sentiment_logits = self.sentiment_classifier(sentiment_logits)

        else:
            sentiment_logits = self.sentiment_classifier(sentiment_logits)

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
        embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        embeddings_2 = self.forward(input_ids_2, attention_mask_2)
        similarity_logits = F.cosine_similarity(embeddings_1, embeddings_2)

        return similarity_logits * 5


def save_model(model, optimizer, args, config, filepath):
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

    if args.task == "sts" or args.task == "multitask":
        sts_train_data = SentencePairDataset(sts_train_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args)

        sts_train_dataloader = DataLoader(
            sts_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sts_train_data.collate_fn,
        )
        sts_dev_dataloader = DataLoader(
            sts_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=sts_dev_data.collate_fn,
        )

    if args.task == "qqp" or args.task == "multitask":
        quora_train_data = SentencePairDataset(quora_train_data, args)
        quora_dev_data = SentencePairDataset(quora_dev_data, args)

        quora_train_dataloader = DataLoader(
            quora_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=quora_train_data.collate_fn,
        )
        quora_dev_dataloader = DataLoader(
            quora_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=quora_dev_data.collate_fn,
        )

    if args.task == "etpc" or args.task == "multitask":
        etpc_train_data = SentencePairDataset(etpc_train_data, args)
        etpc_dev_data = SentencePairDataset(etpc_dev_data, args)

        etpc_train_dataloader = DataLoader(
            etpc_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=etpc_train_data.collate_fn,
        )
        etpc_dev_dataloader = DataLoader(
            etpc_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=etpc_dev_data.collate_fn,
        )

    # Init model
    # global config_dict

    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "hidden_size": BERT_HIDDEN_SIZE,
        "num_hidden_layers": N_HIDDEN_LAYERS,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
        "pooling": args.pooling,
        "additional_inputs": args.additional_inputs,
        "add_layers": args.add_layers,
        "max_position_embeddings": args.max_position_embeddings,
        "hidden_dim": HIDDEN_DIM,
        "max_length": args.max_length,
        "segment_length": args.segment_length,
    }

    config = SimpleNamespace(**config)
    print(f"config: {config}")

    ctx = (
        nullcontext()
        if not args.use_gpu
        else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    )

    separator = "-" * 30
    print(separator)
    print("    BERT Model Configuration")
    print(separator)
    print(pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)}))
    print(separator)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    if args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif args.optimizer == "sophiag":
        optimizer = SophiaG(
            model.parameters(),
            lr=lr,
            eps=1e-12,
            rho=0.03,
            betas=(0.985, 0.99),
            weight_decay=1e-2,
        )
    hess_interval = 10
    iter_num = 0

    if args.scheduler == "linear_warmup":
        if args.task == "sst":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=len(sst_train_dataloader) * args.epochs,
            )
        elif args.task == "sts":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=len(sts_train_dataloader) * args.epochs,
            )
        elif args.task == "qqp":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=len(quora_train_dataloader) * args.epochs,
            )
    else:
        scheduler = None

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

                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1))
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

                if (
                    hasattr(optimizer, "update_hessian")
                    and iter_num % hess_interval == hess_interval - 1
                ):
                    with ctx:
                        logits = model.predict_sentiment(b_ids, b_mask)
                        samp_dist = torch.distributions.Categorical(logits=logits)
                        y_sample = samp_dist.sample()
                        loss_sampled = F.cross_entropy(logits, y_sample.view(-1))
                    loss_sampled.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.update_hessian(bs=args.batch_size)
                    optimizer.zero_grad(set_to_none=True)

                train_loss += loss.item()
                num_batches += 1

        if args.task == "sts" or args.task == "multitask":
            # Trains the model on the sts dataset
            for batch in tqdm(
                sts_train_dataloader, desc=f"train-{epoch + 1:02}", disable=TQDM_DISABLE
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
                # print(f"STS b_labels: {b_labels}")
                b_labels = b_labels.to(device).float()

                optimizer.zero_grad()
                logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                loss = F.mse_loss(logits.float(), b_labels.view(-1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "qqp" or args.task == "multitask":
            # Trains the model on the qqp dataset
            for batch in tqdm(
                quora_train_dataloader,
                desc=f"train-{epoch + 1:02}",
                disable=TQDM_DISABLE,
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
                loss = F.cross_entropy(logits, b_labels.view(-1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "etpc" or args.task == "multitask":
            # Trains the model on the etpc dataset
            for batch in tqdm(
                etpc_train_dataloader,
                desc=f"train-{epoch + 1:02}",
                disable=TQDM_DISABLE,
            ):
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
                )

                binary_labels = [
                    [1 if i in label else 0 for i in range(7)] for label in b_labels
                ]
                binary_labels = torch.tensor(binary_labels)

                # print("labels", torch.tensor(binary_labels))
                # break

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                binary_labels = binary_labels.to(device).float()

                optimizer.zero_grad()
                logits = model.predict_paraphrase_types(
                    b_ids_1, b_mask_1, b_ids_2, b_mask_2
                )
                loss = F.binary_cross_entropy_with_logits(
                    logits.float().flatten(),
                    binary_labels.flatten(),
                )
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

        test_model_multitask(args, model, device)


def split_csv(split=0.8):

    file_path = "data/etpc-paraphrase-dev.csv"
    file = Path(file_path)
    if file.exists():
        print("The dev file already exists")
        return
    with open("data/etpc-paraphrase-orig.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)

    header, rows = data[0], data[1:]

    split_idx = int(split * len(rows))  # 80/20 split like in description

    train = [header] + rows[:split_idx]
    dev = [header] + rows[split_idx:]

    with open("data/etpc-paraphrase-train.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(train)

    with open("data/etpc-paraphrase-dev.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(dev)


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
    parser.add_argument(
        "--option",
        type=str,
        help="pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated",
        choices=("pretrain", "finetune"),
        default="finetune",
    )
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--smoketest", action="store_true", help="Run a smoke test")

    args, _ = parser.parse_known_args()

    parser.add_argument("--epochs", type=int, default=10 if not args.smoketest else 1)
    parser.add_argument(
        "--samples_per_epoch", type=int, default=10_000 if not args.smoketest else 10
    )

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

    # You should split the train data into a train and dev set first and change the
    # default path of the --etpc_dev argument to your dev set.

    # split_csv()

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
        "--batch_size",
        help="sst: 64 can fit a 12GB GPU",
        type=int,
        default=64 if not args.smoketest else 64,
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
        default=1e-3 if args.option == "pretrain" else 2e-5,
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        choices=["adamw", "sophiag"],
        help="choose the optimizer",
    )
    parser.add_argument(
        "--scheduler",
        default="linear_warmup",
        choices=["linear_warmup", None],
        help="choose the scheduler",
    )
    parser.add_argument(
        "--pooling",
        default="max",
        choices=[None, "max", "mean"],
        help="choose the pooling method",
    )
    parser.add_argument(
        "--additional_inputs",
        default=False,
        help="use additional inputs for the model",
    )
    parser.add_argument(
        "--add_layers",
        default=False,
        help="add more layers to the model",
    )

    parser.add_argument(
        "--improve_dir",
        type=str,
        default="./improve_dir",
        help="path to save the params",
    )
    parser.add_argument(
        "--sst_improve_dir",
        type=str,
        default="./improve_dir/sst",
        help="path to save the params",
    )
    parser.add_argument(
        "--sts_improve_dir",
        type=str,
        default="./improve_dir/sts",
        help="path to save the params",
    )
    parser.add_argument(
        "--qqp_improve_dir",
        type=str,
        default="./improve_dir/qqp",
        help="path to save the params",
    )
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        default=512,
        help="max position embeddings for the model",
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=34,
        help="segment length for the model",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=68,
        help="max length for the model",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # create the folder before starting this script
    args.filepath = (
        f"models1/{args.option}-{args.epochs}-{args.lr}-{args.task}.pt"  # save path
    )
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
