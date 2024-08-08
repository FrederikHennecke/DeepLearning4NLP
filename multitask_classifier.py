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
from optimizer import AdamW, SophiaG, SophiaH
from transformers import get_linear_schedule_with_warmup
import csv
import pandas as pd
from pathlib import Path

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
N_HIDDEN_LAYERS = 12
N_SENTIMENT_CLASSES = 5
N_PARAPHRASE_TYPES = 7


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

    def __init__(
        self,
        config,
        train_mode=None,
        layers=None,
        pooling=None,
        attention=False,
        add_dropout=False,
        more_layers=False,
    ):
        super(MultitaskBERT, self).__init__()

        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert parameters.
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )
        self.additional_inputs = config.additional_inputs
        # Dropout for regularization
        self.add_dropout = add_dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooling = pooling
        self.train_mode = train_mode
        self.more_layers = more_layers

        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True

        ### TODO
        # raise NotImplementedError
        if layers is None or layers[0] == -2 or self.train_mode == "last_layer":
            self.layers = []
        elif layers[0] == -1 or self.train_mode == "all_layers":
            self.layers = [i for i in range(config.num_hidden_layers)]
        else:
            self.layers = layers

        self.sentiment_classifier = nn.Linear(
            config.hidden_size
            * max(1, len(self.layers) if self.pooling is None else 1),
            N_SENTIMENT_CLASSES,
        )
        # self.sentiment_classifier = nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)

        self.paraphrase_classifier = nn.Linear(
            3 * config.hidden_size, 1
        )  # WARN Not needed anymore
        self.similarity_classifier = nn.Linear(3 * config.hidden_size, 1)
        self.paraphrase_type_classifier = nn.Linear(
            3 * config.hidden_size, N_PARAPHRASE_TYPES
        )

        # Dropout for regularization
        self.pooling = pooling
        self.train_mode = train_mode

        # add more layers before the classifier
        self.attention = attention
        self.attention_layer = AttentionLayer(config.hidden_size)

        self.sentiment_linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.sentiment_linear1 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.sentiment_linear2 = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_ids, attention_mask):
        """Takes a batch of sentences and produces embeddings for them."""

        # The final BERT embedding is the hidden state of [CLS] token (the first token).
        # See BertModel.forward() for more details.
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        # raise NotImplementedError
        output = self.bert(input_ids, attention_mask, self.additional_inputs)

        if self.train_mode == "all_layers":
            pooled_output = self.dropout(output["pooler_output"]).unsqueeze(0)
            pooled_output2 = self.dropout(output["pooled_output2"]).unsqueeze(0)
            pooled_output3 = self.dropout(output["pooled_output3"]).unsqueeze(0)
            pooled_output4 = self.dropout(output["pooled_output4"]).unsqueeze(0)
            pooled_output5 = self.dropout(output["pooled_output5"]).unsqueeze(0)
            pooled_output6 = self.dropout(output["pooled_output6"]).unsqueeze(0)
            pooled_output7 = self.dropout(output["pooled_output7"]).unsqueeze(0)
            pooled_output8 = self.dropout(output["pooled_output8"]).unsqueeze(0)
            pooled_output9 = self.dropout(output["pooled_output9"]).unsqueeze(0)
            pooled_output10 = self.dropout(output["pooled_output10"]).unsqueeze(0)
            pooled_output11 = self.dropout(output["pooled_output11"]).unsqueeze(0)
            pooled_output12 = self.dropout(output["pooled_output12"]).unsqueeze(
                0
            )  # 12, batchsize, hidden_dim
            hidden_state = torch.cat(
                (
                    pooled_output,
                    pooled_output2,
                    pooled_output3,
                    pooled_output4,
                    pooled_output5,
                    pooled_output6,
                    pooled_output7,
                    pooled_output8,
                    pooled_output9,
                    pooled_output10,
                    pooled_output11,
                    pooled_output12,
                ),
                0,
            )

            # seq_len, batch_size, hidden_dim = hidden_state.size()
            # pooled_output = self.dropout(hidden_state)

            # add_layer = self.linear_first(
            #     pooled_output
            # )  # seq_len. batchsize. hidden_dim
            # add_layer = F.tanh(add_layer)
            # add_layer = self.linear_second(add_layer)
            # add_layer = F.softmax(add_layer, dim=0)

            # b = []
            # y = []
            # for i in range(self.attn_heads):
            #     b.append(add_layer[:, :, i])
            #     b[i] = b[i].unsqueeze(2).expand(seq_len, batch_size, hidden_dim)
            #     y.append((b[i] * pooled_output).sum(dim=0))  #  batchsize, hidden_dim
            # hidden_state = torch.cat(y, 1)  # batchsize, hidden_dim*heads

        elif self.train_mode == "single_layer":
            all_encoded_layers = output["all_encoded_layers"]
            if len(self.layers) > 0 and ([-1, -2] not in self.layers):
                hidden_state = []
                for l in self.layers:
                    hidden_state.append(all_encoded_layers[l][:, 0].unsqueeze(1))
                hidden_state = torch.cat(hidden_state, dim=1)

        elif self.train_mode == "last_layer":
            hidden_state = output["last_hidden_state"]

        else:
            raise ValueError("Invalid train mode")

        if self.attention:
            hidden_state = self.attention_layer(hidden_state)

        if self.pooling == None:
            # pooled_output = hidden_state.view(hidden_state.size(0), -1)
            pooled_output = hidden_state
        elif self.pooling == "max":
            # pooled_output, _ = torch.max(hidden_state, dim=1)
            pooled_output = nn.MaxPool1d(1)(hidden_state).squeeze(-1)
        elif self.pooling == "mean":
            # pooled_output = torch.mean(hidden_state, dim=1)
            pooled_output = nn.AvgPool1d(1)(hidden_state).squeeze(-1)

        return pooled_output

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
        sentiment_logits = self.forward(input_ids, attention_mask)
        if self.more_layers:
            sentiment_logits = F.relu(self.sentiment_linear(sentiment_logits))
            sentiment_logits = F.relu(self.sentiment_linear1(sentiment_logits))
            sentiment_logits = F.relu(self.sentiment_linear2(sentiment_logits))
        if self.add_dropout:
            sentiment_logits = self.dropout(sentiment_logits)
        sentiment_logits = self.sentiment_classifier(sentiment_logits)
        return sentiment_logits

    def predict_paraphrase(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):  # WARN This function is not needed anymore since is mentioned in paraphrase_types
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
    # print(f'save info: {save_info}')

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
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "hidden_size": BERT_HIDDEN_SIZE,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
        "num_hidden_layers": N_HIDDEN_LAYERS,
        "additional_inputs": args.additional_inputs,
        "train_mode": args.train_mode,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "attention": args.attention,
        "add_dropout": args.add_dropout,
        "weight_decay": args.weight_decay,
        "more_layers": args.more_layers,
    }

    config = SimpleNamespace(**config)

    separator = "-" * 30
    print(separator)
    print("    BERT Model Configuration")
    print(separator)
    print(pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)}))
    print(separator)

    model = MultitaskBERT(
        config,
        args.train_mode,
        args.layers,
        args.pooling_type,
        args.attention,
        args.add_dropout,
        args.more_layers,
    )
    model = model.to(device)

    lr = args.lr
    weight_decay = args.weight_decay
    if args.optimizer == "AdamW":
        optimizer = AdamW(
            model.parameters(), lr=lr, eps=1e-8, weight_decay=weight_decay
        )
    elif args.optimizer == "SophiaG":
        optimizer = SophiaG(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-12,
            betas=(0.985, 0.99),
            rho=0.03,
        )
    elif args.optimizer == "SophiaH":
        optimizer = SophiaH(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-12,
            betas=(0.985, 0.99),
            rho=0.05,
            update_period=10,
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    if args.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(sst_train_dataloader) * args.epochs,
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

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1))
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

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
                b_labels = b_labels.to(device).float()

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                loss = F.binary_cross_entropy_with_logits(
                    logits.float(), b_labels.view(-1)
                )
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

        model = MultitaskBERT(
            config,
            args.train_mode,
            args.layers,
            args.pooling_type,
            args.attention,
            args.add_dropout,
            args.more_layers,
        )
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        return test_model_multitask(args, model, device)


def etpc_split(args):
    file_path = args.etpc_dev
    file = Path(file_path)
    if file.exists():
        print("etpc data already split into train and dev sets.")
        return None

    print("Splitting etpc data into train and dev sets...")
    with open("data/etpc-paraphrase-original.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)

    header, rows = data[0], data[1:]

    split_idx = int(0.80 * len(rows))  # 80/20 split like in description

    train = [header] + rows[:split_idx]
    dev = [header] + rows[split_idx:]

    with open(args.etpc_train, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(train)

    with open(args.etpc_dev, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(dev)

    df_train = pd.read_csv(args.etpc_train, sep="\t")
    df_dev = pd.read_csv(args.etpc_dev, sep="\t")
    print("train size", df_train.shape)
    print("dev size", df_dev.shape)


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
        default="finetune",
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
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
        default=1e-3 if args.option == "pretrain" else 2e-5,
    )
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[11],
        help="choose the layers that used for downstream tasks, "
        "-2 means use pooled output, -1 means all layer,"
        "else means the detail layers. default is -2",
    )

    parser.add_argument("--pooling_type", default=None, choices=[None, "mean", "max"])

    parser.add_argument(
        "--train_mode",
        default="single_layer",
        type=str,
        choices=["last_layer", "all_layers", "single_layer"],
        help="choose the training mode, last_layer: only train the last layer,"
        "all_layers: train all layers, single_layer: train the specified layers",
    )
    parser.add_argument(
        "--optimizer",
        default="SophiaG",
        type=str,
        choices=["AdamW", "SophiaG", "SophiaH"],
        help="choose the optimizer",
    )
    parser.add_argument(
        "--scheduler",
        default=None,
        type=str,
        choices=["linear_warmup", None],
        help="choose the scheduler",
    )
    parser.add_argument(
        "--additional_inputs",
        type=bool,
        default=True,
        help="feed more features to the model",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="weight decay for the optimizer",
    )
    parser.add_argument(
        "--attention",
        type=bool,
        default=True,
        help="use attention layer for the model",
    )
    parser.add_argument(
        "--add_dropout",
        type=bool,
        default=False,
        help="use dropout layer for the model",
    )
    parser.add_argument(
        "--more_layers",
        type=bool,
        default=False,
        help="add more layers before the classifier",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"models1/{args.option}-{str(args.epochs)}-{str(args.lr)}-{args.task}.pt"  # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    # etpc_split(args)
    train_multitask(args)
    test_model(args)
