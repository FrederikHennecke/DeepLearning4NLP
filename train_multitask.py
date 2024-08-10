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
from optimizer import AdamW, SophiaG, SophiaH
from contextlib import nullcontext
from bert import BertLayer
from transformers import get_linear_schedule_with_warmup
import math


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
N_HIDDEN_LAYERS = 12


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

        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True

        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # attention layer
        self.attention_layer = AttentionLayer(config.hidden_size)

        # sentiment
        self.sentiment_linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.sentiment_linear1 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.sentiment_linear2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.sentiment_classifier = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)

        # paraphrase
        self.paraphrase_linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.paraphrase_linear1 = torch.nn.Linear(
            config.hidden_size * 2, config.hidden_size
        )
        self.paraphrase_linear2 = torch.nn.Linear(
            config.hidden_size, config.hidden_size
        )
        self.paraphrase_classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        all_encoded_layers, pooled_output, all_sequences, all_pooled = self.bert(
            input_ids, attention_mask
        )
        attention_results = self.attention_layer(
            all_encoded_layers["last_hidden_state"]
        )
        return attention_results

    def predict_sentiment(self, input_ids, attention_mask):
        bert_embeddings = F.relu(self.forward(input_ids, attention_mask))
        sentiment_logits = F.relu(self.sentiment_linear(bert_embeddings))
        sentiment_logits = F.relu(self.sentiment_linear1(sentiment_logits))
        sentiment_logits = F.relu(self.sentiment_linear2(sentiment_logits))
        sentiment_logits = self.sentiment_classifier(sentiment_logits)
        return sentiment_logits

    def predict_paraphrase_train(
        self, input_ids1, attention_mask1, input_ids2, attention_mask2
    ):

        bert_embeddings1 = self.forward(input_ids1, attention_mask1)
        bert_embeddings2 = self.forward(input_ids2, attention_mask2)

        # apply relu later
        combined_bert_embeddings1 = self.paraphrase_linear(bert_embeddings1)
        combined_bert_embeddings2 = self.paraphrase_linear(bert_embeddings2)

        abs_diff = torch.abs(combined_bert_embeddings1 - combined_bert_embeddings2)
        abs_sum = torch.abs(combined_bert_embeddings1 + combined_bert_embeddings2)

        concatenated_features = torch.cat(abs_diff, abs_sum, dim=1)

        paraphrase_logits = F.relu(self.paraphrase_linear1(concatenated_features))
        paraphrase_logits = F.relu(self.paraphrase_linear2(paraphrase_logits))
        paraphrase_logits = self.paraphrase_classifier(paraphrase_logits)
        return paraphrase_logits

    def predict_paraphrase(
        self, input_ids1, attention_mask1, input_ids2, attention_mask2
    ):
        paraphrase_logits = self.predict_paraphrase_train(
            input_ids1, attention_mask1, input_ids2, attention_mask2
        )
        return paraphrase_logits.argmax(dim=-1)

    def predict_similarity(
        self, input_ids1, attention_mask1, input_ids2, attention_mask2
    ):
        bert_embeddings1 = self.forward(input_ids1, attention_mask1)
        bert_embeddings2 = self.forward(input_ids2, attention_mask2)

        similarity = F.cosine_similarity(bert_embeddings1, bert_embeddings2)
        return similarity * 5

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

def load_model(filepath, model, optimizer, use_gpu):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    save_info = torch.load(filepath, map_location= torch.device('cuda') if use_gpu else 'cpu'))
    model.load_state_dict(save_info["model"])
    optimizer.load_state_dict(save_info["optim"])
    args= save_info["args"]
    args.use_gpu = use_gpu
    config = save_info["model_config"]

    random.setstate(save_info["system_rng"])
    np.random.set_state(save_info["numpy_rng"])
    torch.random.set_rng_state(save_info["torch_rng"])
    print(f"Loading the model from {filepath}.")
    return model, optimizer, args, config

def train_multitask(args):
    if isinstance(args, dict):
        args= SimpleNamespace(**args)

    train_all_datasets= True
    n_datasets= args.sst + args.sts + args.para
    if args.sst or args.sts or args.para:
        train_all_datasets= False
    if n_datasets == 0:
        n_datasets= 3

    # Load data
    # Create the data and its corresponding datasets and dataloader:
    sst_train_data, num_labels, para_train_data, sts_train_data, etpc_train_data = (
        load_multitask_data(
            args.sst_train,
            args.quora_train,
            args.sts_train,
            args.etpc_train,
            split="train",
        )
    )
    sst_dev_data, num_labels, para_dev_data, sts_dev_data, etpc_dev_data = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="train"
    )

    sst_train_dataloader = None
    sst_dev_dataloader = None
    para_train_dataloader = None
    para_dev_dataloader = None
    sts_train_dataloader = None
    sts_dev_dataloader = None
    etpc_train_dataloader = None
    etpc_dev_dataloader = None
    total_num_batches = 0

    sst_train_data = SentenceClassificationDataset(sst_train_data, args, override_length=args.sample_per_epoch)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args, override_length=10 if args.smoketest else None)

    sst_train_dataloader = DataLoader(
        sst_train_data,
        shuffle= False,     # check the shuffling later
        batch_size=args.batch_size,
        collate_fn=sst_train_data.collate_fn,
        num_workers= 2,
    )
    sst_dev_dataloader = DataLoader(
        sst_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sst_dev_data.collate_fn,
        num_workers=2,
    )

    para_train_data = SentencePairDataset(para_train_data, args, override_length=args.sample_per_epoch)
    para_dev_data = SentencePairDataset(para_dev_data, args, override_length=10 if args.smoketest else None)

    para_train_dataloader = DataLoader(
        para_train_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=para_train_data.collate_fn,
        num_workers=2,
    )
    para_dev_dataloader = DataLoader(
        para_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn,
        num_workers=2,
    )


    sts_train_data = SentencePairDataset(sts_train_data, args, override_length=args.sample_per_epoch)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, override_length=10 if args.smoketest else None)

    sts_train_dataloader = DataLoader(
        sts_train_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sts_train_data.collate_fn,
        num_workers=2,
    )
    sts_dev_dataloader = DataLoader(
        sts_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sts_dev_data.collate_fn,
        num_workers=2,
    )

    tottal_num_batches += math.ceil(args.samples_per_epoch / args.batch_size)
    # Init model
    global config_dict

    config_dict = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        'num_labels': num_labels,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
        "additional_inputs": args.additional_inputs,
    }

    config = SimpleNamespace(**config_dict)
    separator = "-" * 30
    print(separator)
    print("    Multitask BERT Model Configuration")
    print(separator)
    print(pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)}))
    print(separator)

    model = MultitaskBERT(config)
    device= torch.device('*cpu')
    if torch.cuda.is_available() and args.use_gpu:
        device= torch.device('cuda')
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs", file=sys.stderr)
            model= nn.DataParallel(model)
    model = model.to(device)

    lr = args.lr
    hess_interval= args.hess_interval
    ctx= (nullcontext()
    if not args.use_gpu else torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16))

    if args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay= args.weight_decay)
    elif args.optimizer == "sophiah":
        optimizer = SophiaH(
            model.parameters(),
            lr=lr,
            eps=1e-12,
            rho= args.rho,
            betas=(0.985, 0.99),
            weight_decay=args.weight_decay,
            update_interval= hess_interval
        )
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")

    if args.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(sst_train_dataloader) * args.epochs,
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=2, verbose=True
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1, eta_min=1)
    else:
        scheduler = None

    best_dev_acc_para = 0
    best_dev_acc_sst = 0
    best_dev_acc_sts = 0

    if args.checkpoint:
        model, optimizer, _, config = load_model(args.checkpoint, model, optimizer, args.use_gpu)
