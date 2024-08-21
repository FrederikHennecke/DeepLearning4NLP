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
import subprocess
from datetime import datetime
from itertools import cycle
from contextlib import nullcontext
import math
import warnings
import json
import uuid
from filelock import FileLock
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from transformers import get_linear_schedule_with_warmup
from transformers import RobertaModel, RobertaTokenizer

from tokenizer import BertTokenizer
from bert import BertModel, BertLayer, BertModelWithPAL
from datasets import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from config import BertConfig
from combined_models import CombinedModel
from layers import AttentionLayer
from evaluation_separate import (
    model_eval_multitask,
    test_model_multitask,
    model_eval_paraphrase,
    model_eval_sts,
    model_eval_sentiment,
)
from optimizer import AdamW, SophiaG, SophiaH
from pcgrad import PCGrad
from gradvac_amp import GradVacAMP
from pcgrad_amp import PCGradAMP
from smart_regularization import smart_regularization

warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd")


# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


TQDM_DISABLE = False
BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5
N_HIDDEN_LAYERS = 12
HIDDEN_DIM = 512
output_dim = 64
os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"


def get_term_width():
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class MultitaskBERT(nn.Module):
    """
    This should use BERT for these tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    (- Paraphrase type detection (predict_paraphrase_types))
    """

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()

        if args.model_name == "roberta-base":
            self.bert = RobertaModel.from_pretrained(
                "roberta-base", local_files_only=config.local_files_only
            )
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        elif args.model_name == "bert-large":
            self.bert = BertModel.from_pretrained(
                "bert-large-uncased", local_files_only=config.local_files_only
            )
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
            config.hidden_size = 1024
            BERT_HIDDEN_SIZE = 1024

        else:
            self.bert = BertModel.from_pretrained(
                "bert-base-uncased", local_files_only=config.local_files_only
            )
            self.tokenizer = BertTokenizer.from_pretrained("bert-base")

        # initialize params
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True

        self.config = config
        self.attention_layer = AttentionLayer(BERT_HIDDEN_SIZE)

        # Dropout for regularization
        self.dropout_sentiment = nn.ModuleList(
            [
                nn.Dropout(config.hidden_dropout_prob)
                for _ in range(config.n_hidden_layers + 1)
            ]
        )

        # sentiment
        self.sentiment_linear = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.hidden_size)
                for _ in range(config.n_hidden_layers)
            ]
            + [nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)]
        )
        self.last_sentiment_linear = None

        # paraphrase
        self.dropout_paraphrase = nn.ModuleList(
            [
                nn.Dropout(config.hidden_dropout_prob)
                for _ in range(config.n_hidden_layers + 1)
            ]
        )
        self.paraphrase_linear = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.hidden_size)
                for _ in range(config.n_hidden_layers)
            ]
            + [nn.Linear(config.hidden_size, 1)]
        )

        # similarity
        self.dropout_similarity = nn.ModuleList(
            [
                nn.Dropout(config.hidden_dropout_prob)
                for _ in range(config.n_hidden_layers + 1)
            ]
        )
        self.similarity_linear = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.hidden_size)
                for _ in range(config.n_hidden_layers)
            ]
            + [nn.Linear(config.hidden_size, 1)]
        )

        if args.no_train_classifier:
            for param in self.sentiment_linear.parameters():
                param.requires_grad = False
            for param in self.paraphrase_linear.parameters():
                param.requires_grad = False
            for param in self.similarity_linear.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, task_id):

        if isinstance(self.bert, BertModelWithPAL):
            bert_output = self.bert(input_ids, attention_mask, task_id)["pooler_output"]
        else:
            _, pooler_output, all_sequences, _ = self.bert(input_ids, attention_mask)
            bert_output = all_sequences["pooler_output"]

        return bert_output

    def last_layers_sentiment(self, x):
        for i in range(len(self.sentiment_linear) - 1):
            x = self.dropout_sentiment[i](x)
            x = F.relu(self.sentiment_linear[i](x))

        x = self.dropout_sentiment[-1](x)
        logits = self.sentiment_linear[-1](x)
        # logits = F.softmax(logits, dim=1)
        return logits

    def predict_sentiment(self, input_ids, attention_mask):
        x = self.forward(input_ids, attention_mask, task_id=0)
        x = self.last_layers_sentiment(x)

        if self.last_sentiment_linear is not None:
            x = self.last_sentiment_linear(x)

        return x

    def get_similarity_paraphrase_embeddings(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, task_id
    ):
        sep_token_id = torch.tensor(
            [self.tokenizer.sep_token_id], dtype=torch.long, device=input_ids_1.device
        )
        batch_sep_token_id = sep_token_id.repeat(input_ids_1.shape[0], 1)

        input_id = torch.cat(
            (input_ids_1, batch_sep_token_id, input_ids_2, batch_sep_token_id), dim=1
        )
        attention_mask = torch.cat(
            (
                attention_mask_1,
                torch.ones_like(batch_sep_token_id),
                attention_mask_2,
                torch.ones_like(batch_sep_token_id),
            ),
            dim=1,
        )

        x = self.forward(input_id, attention_mask, task_id)
        return x

    def last_layers_paraphrase(self, x):
        for i in range(len(self.paraphrase_linear) - 1):
            x = self.dropout_paraphrase[i](x)
            x = F.relu(self.paraphrase_linear[i](x))
        x = self.dropout_paraphrase[-1](x)
        logits = self.paraphrase_linear[-1](x)
        # logits = torch.sigmoid(logits)
        return logits

    def predict_paraphrase(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        x = self.get_similarity_paraphrase_embeddings(
            input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, task_id=1
        )
        x = self.last_layers_paraphrase(x)
        return x

    def last_layers_similarity(self, x):
        for i in range(len(self.similarity_linear) - 1):
            x = self.dropout_similarity[i](x)
            x = F.relu(self.similarity_linear[i](x))
        x = self.dropout_similarity[-1](x)
        logits = self.similarity_linear[-1](x)
        # logits = torch.sigmoid(logits) *6 - 0.5 # scale to [-0.5, 0.5]

        # if not self.training:
        #     logits = torch.clamp(logits, 0, 5)
        return logits

    def predict_similarity(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        x = self.get_similarity_paraphrase_embeddings(
            input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, task_id=2
        )
        x = self.last_layers_similarity(x)
        return x


class ObjectsGroup:
    def __init__(self, model, optimizer, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.loss_sum = 0


class Scheduler:
    def __init__(self, dataloaders, reset=True):
        self.dataloaders = dataloaders
        self.names = list(dataloaders.keys())
        if reset:
            self.reset()

    def reset(self):
        self.sst_iter = iter(self.dataloaders["sst"])
        self.para_iter = iter(self.dataloaders["para"])
        self.sts_iter = iter(self.dataloaders["sts"])
        self.steps = {"sst": 0, "para": 0, "sts": 0}

    def get_SST_batch(self):
        try:
            return next(self.sst_iter)
        except StopIteration:
            self.sst_iter = cycle(self.dataloaders["sst"])
            return next(self.sst_iter)

    def get_Paraphrase_batch(self):
        try:
            return next(self.para_iter)
        except StopIteration:
            self.para_iter = cycle(self.dataloaders["para"])
            return next(self.para_iter)

    def get_STS_batch(self):
        try:
            return next(self.sts_iter)
        except StopIteration:
            self.sts_iter = cycle(self.dataloaders["sts"])
            return next(self.sts_iter)

    def get_batch(self, name):
        if name == "sst":
            return self.get_SST_batch()
        elif name == "para":
            return self.get_Paraphrase_batch()
        elif name == "sts":
            return self.get_STS_batch()
        else:
            raise ValueError(f"Invalid batch name: {name}")

    def process_named_batch(self, objects_group, args, name, apply_optimization):
        batch = self.get_batch(name)
        process_fn, gradient_accumulations = None, 0
        if name == "sst":
            process_fn = process_sentiment_batch
            gradient_accumulations = args.gradient_accumulations_sst
        elif name == "para":
            process_fn = process_paraphrase_batch
            gradient_accumulations = args.gradient_accumulations_para
        elif name == "sts":
            process_fn = process_similarity_batch
            gradient_accumulations = args.gradient_accumulations_sts
        else:
            raise ValueError(f"Invalid batch name: {name}")

        loss_of_batch = 0
        for _ in range(gradient_accumulations):
            loss_of_batch += process_fn(batch, objects_group, args)

        self.steps[name] += 1
        if apply_optimization:
            step_optimizer(objects_group, args, step=self.steps[name])

        return loss_of_batch


class RandomScheduler(Scheduler):
    def __init__(self, dataloaders):
        super().__init__(dataloaders, reset=True)

    def process_one_batch(self, epoch, num_epochs, objects_group, args):
        name = random.choice(self.names)
        return name, self.process_named_batch(objects_group, args, name)


class RoundRobinScheduler(Scheduler):
    def __init__(self, dataloaders):
        super().__init__(dataloaders, reset=False)
        self.reset()

    def reset(self):
        self.index = 0
        return super().reset()

    def process_one_batch(self, epoch, num_epochs, objects_group, args):
        name = self.names[self.index]
        self.index = (self.index + 1) % len(self.names)
        return name, self.process_named_batch(objects_group, args, name)


class PalScheduler(Scheduler):
    def __init__(self, dataloaders):
        super().__init__(dataloaders, reset=False)
        self.sizes = np.array([len(dataloaders[dataset]) for dataset in self.names])
        self.reset()

    def process_one_batch(
        self, epoch, num_epochs, objects_group, args, apply_optimization=True
    ):
        alpha = 0.2
        if num_epochs > 1:
            alpha = 1 - 0.8 * (epoch - 1) / (num_epochs - 1)
        probs = self.sizes**alpha
        probs /= np.sum(probs)
        name = np.random.choice(self.names, p=probs)
        return name, self.process_named_batch(
            objects_group, args, name, apply_optimization=apply_optimization
        )

    def process_several_batches_with_control(
        self, epoch, num_epochs, objects_group, args, num_batches
    ):
        schedule = ["sst", "para", "sts"]
        alpha = 0.2
        if num_epochs > 1:
            alpha = 1 - 0.8 * (epoch - 1) / (num_epochs - 1)
        probs = self.sizes**alpha
        probs /= np.sum(probs)
        probs_biased = (probs * num_batches - 1) / (num_batches - 3)
        probs_biased = np.clip(probs_biased, 0.025, 1)
        probs_biased /= np.sum(probs_biased)
        schedule += np.random.choice(
            self.names, size=num_batches - 3, p=probs_biased
        ).tolist()
        random.shuffle(schedule)

        losses = []
        for task in schedule:
            loss = self.process_named_batch(
                objects_group, args, task, apply_optimization=False
            )
            losses.append(loss)
        return schedule, losses


def process_sentiment_batch(batch, objects_group, args):
    device = args.device
    model, scaler = objects_group.model, objects_group.scaler

    with autocast() if args.use_amp else nullcontext():
        b_ids, b_mask, b_labels = (
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["labels"].to(device),
        )
        embeddings = model.forward(b_ids, b_mask, task_id=0)
        logits = model.last_layers_sentiment(embeddings)

        loss = (
            F.cross_entropy(logits, b_labels.view(-1), reduction="sum")
            / args.batch_size
        )
        loss_value = loss.item()

        if args.use_smart_regularization:
            smart_regularization(
                loss_value,
                args.smart_weight_regularization,
                embeddings,
                logits,
                model.last_layers_sentiment,
            )
        objects_group.loss_sum += loss_value

        if args.projection == "none":
            if args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        return loss


def process_paraphrase_batch(batch, objects_group, args):
    device = args.device
    model, scaler = objects_group.model, objects_group.scaler

    with autocast() if args.use_amp else nullcontext():
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
            batch["input_ids_1"].to(device),
            batch["attention_mask_1"].to(device),
            batch["input_ids_2"].to(device),
            batch["attention_mask_2"].to(device),
            batch["labels"].to(device),
        )
        embeddings = model.get_similarity_paraphrase_embeddings(
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, task_id=1
        )
        logits = model.last_layers_paraphrase(embeddings)

        loss = (
            F.binary_cross_entropy_with_logits(
                logits.view(-1), b_labels.float(), reduction="sum"
            )
            / args.batch_size
        )
        loss_value = loss.item()

        if args.use_smart_regularization:
            smart_regularization(
                loss_value,
                args.smart_weight_regularization,
                embeddings,
                logits,
                model.last_layers_paraphrase,
            )
        objects_group.loss_sum += loss_value

        if args.projection == "none":
            if args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        return loss


def process_similarity_batch(batch, objects_group, args):
    device = args.device
    model, scaler = objects_group.model, objects_group.scaler

    with autocast() if args.use_amp else nullcontext():
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
            batch["input_ids_1"].to(device),
            batch["attention_mask_1"].to(device),
            batch["input_ids_2"].to(device),
            batch["attention_mask_2"].to(device),
            batch["labels"].to(device),
        )
        embeddings = model.get_similarity_paraphrase_embeddings(
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, task_id=2
        )
        logits = model.last_layers_similarity(embeddings)

        loss = (
            F.mse_loss(logits.view(-1), b_labels.view(-1), reduction="sum")
            / args.batch_size
        )
        loss_value = loss.item()

        if args.use_smart_regularization:
            smart_regularization(
                loss_value,
                args.smart_weight_regularization,
                embeddings,
                logits,
                model.last_layers_similarity,
            )
        objects_group.loss_sum += loss_value

        if args.projection == "none":
            if args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        return loss


def step_optimizer(objects_group, args, step, total_nb_batches):
    optimizer, scaler = objects_group.optimizer, objects_group.scaler
    if args.use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad()
    loss_value = objects_group.loss_sum
    objects_group.loss_sum = 0
    torch.cuda.empty_cache()
    if TQDM_DISABLE:
        str_total_nb_batches = (
            "?" if total_nb_batches is None else str(total_nb_batches)
        )
        print(f"batch {step}/{str_total_nb_batches} STS - Loss: {loss_value:.5f}")
    return loss_value


def finish_training_batch(
    objects_group, args, step, gradient_accumulations, total_nb_batches=None
):
    if step % gradient_accumulations == 0:
        step_optimizer(objects_group, args, step, total_nb_batches)
        return True
    return False


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
    print(f"Saving the model to {filepath}.")
    torch.save(save_info, filepath)


def load_model(filepath, model, optimizer, use_gpu, combined_models=False):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with torch.no_grad():
        save_info = torch.load(
            filepath, map_location=torch.device("cuda") if use_gpu else "cpu"
        )
        print(f"Loading the model from {filepath}.")
        model.load_state_dict(save_info["model"])
        optimizer.load_state_dict(save_info["optim"])
        args = save_info["args"]
        args.use_gpu = use_gpu
        config = save_info["model_config"]
        random.setstate(save_info["system_rng"])
        np.random.set_state(save_info["numpy_rng"])
        torch.random.set_rng_state(save_info["torch_rng"])

    return model, optimizer, args, config


def train_multitask(args):
    if isinstance(args, dict):
        args = SimpleNamespace(**args)

    train_all_datasets = True
    n_datasets = 3

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
    sst_dev_data, num_labels, para_dev_data, sts_dev_data, etpc_dev_data = (
        load_multitask_data(
            args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="train"
        )
    )

    sst_train_dataloader = None
    sst_dev_dataloader = None
    para_train_dataloader = None
    para_dev_dataloader = None
    sts_train_dataloader = None
    sts_dev_dataloader = None
    total_num_batches = 0

    sst_train_data = SentenceClassificationDataset(
        sst_train_data, args, override_length=args.samples_per_epoch
    )
    sst_dev_data = SentenceClassificationDataset(
        sst_dev_data, args, override_length=10 if args.smoketest else None
    )

    sst_train_dataloader = DataLoader(
        sst_train_data,
        shuffle=True,  # check the shuffling later
        batch_size=args.batch_size,
        collate_fn=sst_train_data.collate_fn,
        num_workers=2,
    )
    sst_dev_dataloader = DataLoader(
        sst_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sst_dev_data.collate_fn,
        num_workers=2,
    )

    para_train_data = SentencePairDataset(
        para_train_data, args, override_length=args.samples_per_epoch
    )
    para_dev_data = SentencePairDataset(
        para_dev_data, args, override_length=10 if args.smoketest else None
    )

    para_train_dataloader = DataLoader(
        para_train_data,
        shuffle=True,
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

    sts_train_data = SentencePairDataset(
        sts_train_data, args, isRegression=True, override_length=args.samples_per_epoch
    )
    sts_dev_data = SentencePairDataset(
        sts_dev_data,
        args,
        isRegression=True,
        override_length=10 if args.smoketest else None,
    )

    sts_train_dataloader = DataLoader(
        sts_train_data,
        shuffle=True,
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

    total_num_batches += math.ceil(args.samples_per_epoch / args.batch_size)
    # Init model

    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "num_labels": num_labels,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
        "hidden_size": BERT_HIDDEN_SIZE,
        "n_hidden_layers": args.n_hidden_layers,
        'model_name': args.model_name
    }

    save_params_dir = args.config_save_dir + args.name + ".json"
    train_params = pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)})

    if args.no_tensorboard:
        with open(save_params_dir, "w") as f:
            json.dump(train_params, f)

    config = SimpleNamespace(**config)

    separator = "-" * 30
    print(separator)
    print("    Multitask BERT Model Configuration")
    print(separator)
    print(pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)}))
    print(separator)

    # Print Git info
    branch = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("utf-8")
        .strip()
    )
    commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"])
        .decode("utf-8")
        .strip()[:8]
    )
    is_modified = (
        "(!)"
        if subprocess.check_output(["git", "status", "--porcelain"])
        .decode("utf-8")
        .strip()
        else ""
    )

    # Print Git info
    print(f"Git Branch: {branch}")
    print(f"Git Hash: {commit} {is_modified}")
    print("-" * 60)  # Adjust as needed
    print(f"Command: {' '.join(sys.argv)}")
    print(separator)

    model = MultitaskBERT(config)
    bert_config = BertConfig()

    if args.use_pal:
        BertModelWithPAL.from_BertModel(model.bert, bert_config, train_pal=True)

    device = torch.device("cpu")
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs", file=sys.stderr)
            model = nn.DataParallel(model).module
    model = model.to(device)

    lr = args.lr
    hess_interval = args.hess_interval
    ctx = (
        nullcontext()
        if not args.use_gpu
        else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    )

    if args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sophiah":
        optimizer = SophiaH(
            model.parameters(),
            lr=lr,
            eps=1e-12,
            rho=args.rho,
            betas=(0.985, 0.99),
            weight_decay=args.weight_decay,
            update_period=hess_interval,
        )
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")

    scaler = None if not args.use_amp else GradScaler()

    if args.projection == "pcgrad":
        optimizer = (
            PCGrad(optimizer)
            if not args.use_amp
            else PCGradAMP(optimizer=optimizer, num_tasks=3, scaler=scaler)
        )
    elif args.projection == "vaccine":
        optimizer = GradVacAMP(
            optimizer=optimizer,
            num_tasks=3,
            scaler=scaler,
            DEVICE=device,
            beta=args.beta_vaccine,
        )

    best_dev_acc = 0
    best_dev_accuracies = {"sst": 0, "para": 0, "sts": 0}
    best_dev_rel_improv = 0

    # Save to tensorboard
    if args.no_tensorboard:
        writer = SummaryWriter(args.log_dir)

        s = 'python train_multitask_pal.py'
        for arg in vars(args):
            value = getattr(args, arg)
            if type(value) == bool:
                if value:
                    s += f' --{arg}'
            else:
                s += f' --{arg} {value}'
        print('\n' + 'Command to recreate this run: ' + s + '\n')

        with open(args.logdir + 'command.txt', 'w') as f:
            f.write(s)

    # Infos about the model parameters
    print("\n" + "-" * get_term_width())
    print("number of learnable parameters:", count_learnable_parameters(model))
    print("total number of parameters:", count_parameters(model))
    print("-" * get_term_width() + "\n")

    # Package Objects
    objects_group = ObjectsGroup(model, optimizer, scaler)
    args.device = device
    dataloaders = {
        "sst": sst_train_dataloader,
        "para": para_train_dataloader,
        "sts": sts_train_dataloader,
    }
    scheduler = None

    if args.task_scheduler == "round_robin":
        scheduler = RoundRobinScheduler(dataloaders)
    elif args.task_scheduler == "pal":
        scheduler = PalScheduler(dataloaders)
    elif args.task_scheduler == "random":
        scheduler = RandomScheduler(dataloaders)
    elif args.task_scheduler in ["sst", "para", "sts"]:
        scheduler = RandomScheduler(dataloaders)
        task = args.task_scheduler
        n_batches = 0
        best_dev_acc = -np.inf

        for epoch in range(args.epochs):
            model.train()
            for i in tqdm(
                range(args.num_batches_per_epoch),
                desc=task + " epoch " + str(epoch),
                disable=TQDM_DISABLE,
                smoothing=0,
            ):
                loss = scheduler.process_named_batch(
                    objects_group, args, name=task, apply_optimization=True
                )
                n_batches += 1
                if not args.tensorboard:
                    writer.add_scalar(
                        "Loss " + task, loss.item(), args.batch_size * n_batches
                    )
                    writer.add_scalar(
                        "Specific Loss " + task,
                        loss.item(),
                        args.batch_size * n_batches,
                    )

            dev_acc = 0
            args.task = task
            if task == "sst":
                dev_acc, _, _, _, _ = model_eval_sentiment(
                    sst_dev_dataloader, model, device
                )

            elif task == "para":
                dev_acc, _, _, _, _ = model_eval_paraphrase(
                    para_dev_dataloader, model, device
                )

            elif task == "sts":
                dev_acc, _, _, _, _ = model_eval_sts(sts_dev_dataloader, model, device)

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                save_model(model, optimizer, args, config, args.filepath)

            if not args.no_tensorboard:
                writer.add_scalar(
                    "Dev Accuracy " + task, dev_acc, args.batch_size * n_batches
                )
                writer.add_scalar("Specific Dev Accuracy " + task, dev_acc, epoch)

            print(f"Dev Accuracy {task}: {dev_acc}")
            print(f"Best Dev Accuracy {task}: {best_dev_acc}")
        return

    if args.option == "optimize":
        print("RUN KERNEL OPTIMIZATION FOR SST")
        linear = nn.Linear(5, 5)
        W = np.eye(5)
        B = np.array([0, 0, 0, 0, 0])
        linear.weight.data = torch.from_numpy(W).float()
        linear.bias.data = torch.from_numpy(B).float()
        linear.to(device)
        optimizer = AdamW(linear.parameters(), lr=lr)

        model.last_sentiment_linear = linear
        dev_acc, _, _, _, _ = model_eval_sentiment(sst_dev_dataloader, model, device)

        for epoch in range(args.epochs):
            model.last_sentiment_linear = None
            model.eval()
            for batch in tqdm(
                sst_train_dataloader,
                desc="Kernel Optimization",
                disable=TQDM_DISABLE,
                smoothing=0,
            ):
                b_ids, b_mask, b_labels = (
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["labels"].to(device),
                )
                embeddings = model.forward(b_ids, b_mask, task_id=0)
                logits = model.last_layers_sentiment(embeddings)
                logits = linear(logits)
                loss = F.cross_entropy(logits, b_labels)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            model.last_sentiment_linear = linear
            dev_acc, _, _, _, _ = model_eval_sentiment(
                sst_dev_dataloader, model, device
            )
            print(f"Dev Accuracy SST: {dev_acc}")

        print("weights of linear layer:", linear.weight.data)
        print("bias of linear layer:", linear.bias.data)
        return

    train_loss_logs_epochs = {"sst": [], "para": [], "sts": []}
    dev_acc_logs_epochs = {"sst": [], "para": [], "sts": []}

    if args.option == "individual_pretrain":
        n_batches = 0
        num_batches_per_epoch = (
            args.num_batches_per_epoch
            if args.num_batches_per_epoch > 0
            else len(sst_train_dataloader)
        )

        infos = {
            "sst": {
                "num_batches": num_batches_per_epoch,
                "eval_fn": model_eval_sentiment,
                "dev_dataloader": sst_dev_dataloader,
                "best_dev_acc": 0,
                "best_model": None,
                "layer": model.sentiment_linear,
                "optimizer": AdamW(model.parameters(), lr=lr),
                "last_improv": -1,
                "first": True,
                "first_loss": True,
            },
            "para": {
                "num_batches": num_batches_per_epoch,
                "eval_fn": model_eval_paraphrase,
                "dev_dataloader": para_dev_dataloader,
                "best_dev_acc": 0,
                "best_model": None,
                "layer": model.paraphrase_linear,
                "optimizer": AdamW(model.parameters(), lr=lr),
                "last_improv": -1,
                "first": True,
                "first_loss": True,
            },
            "sts": {
                "num_batches": num_batches_per_epoch,
                "eval_fn": model_eval_sts,
                "dev_dataloader": sts_dev_dataloader,
                "best_dev_acc": 0,
                "best_model": None,
                "layer": model.similarity_linear,
                "optimizer": AdamW(model.parameters(), lr=lr),
                "last_improv": -1,
                "first": True,
                "first_loss": True,
            },
        }
        total_num_batches = {"sst": 0, "para": 0, "sts": 0}

        for epoch in range(args.epochs):
            for task in ["sst", "sts", "para"]:
                if epoch - infos[task]["last_improv"] > args.patience:
                    print(f"Early stopping for {task}")
                    continue
                model.train()
                objects_group.optimizer = infos[task]["optimizer"]
                for i in tqdm(
                    range(infos[task]["num_batches"]),
                    desc=task + " epoch " + str(epoch),
                    disable=TQDM_DISABLE,
                    smoothing=0,
                ):
                    loss = scheduler.process_named_batch(
                        objects_group=objects_group,
                        args=args,
                        name=task,
                        apply_optimization=True,
                    )
                    total_num_batches[task] += 1
                    n_batches += 1
                    if not args.no_tensorboard:
                        if infos[task]["first"]:
                            writer.add_scalar("Loss " + task, loss.item(), 0)
                            infos[task]["first"] = False
                        writer.add_scalar(
                            "Loss " + task, loss.item(), args.batch_size * n_batches
                        )
                        writer.add_scalar(
                            "Specific Loss " + task,
                            loss.item(),
                            args.batch_size * total_num_batches[task],
                        )
                dev_acc, _, _, _, _ = infos[task]["eval_fn"](
                    infos[task]["dev_dataloader"], model, device
                )
                if dev_acc > infos[task]["best_dev_acc"]:
                    infos[task]["best_dev_acc"] = dev_acc
                    infos[task]["best_model"] = copy.deepcopy(
                        infos[task]["layer"].state_dict()
                    )
                    infos[task]["last_improv"] = epoch

                if args.no_tensorboard:
                    writer.add_scalar("[EPOCH] Dev Accuracy " + task, dev_acc, epoch)
                    if infos[task]["first_loss"]:
                        infos[task]["first_loss"] = False
                        writer.add_scalar("Dev Accuracy " + task, dev_acc, 0)
                    writer.add_scalar(
                        "Dev Accuracy " + task, dev_acc, args.batch_size * n_batches
                    )

                print(f"Dev Accuracy {task}: {dev_acc}")
                print(f"Best Dev Accuracy {task}: {infos[task]['best_dev_acc']}")

        # load best model for each task
        for task in infos.keys():
            if infos[task]["best_model"] is not None:
                infos[task]["layer"].load_state_dict(infos[task]["best_model"])

        print("Evaluation Multitask")
        (paraphrase_accuracy, _, _, sentiment_accuracy, _, _, sts_corr, _, _) = (
            model_eval_multitask(
                sst_dev_dataloader,
                para_dev_dataloader,
                sts_dev_dataloader,
                model,
                device,
                writer=writer,
                epoch=0,
                tensorboard=args.no_tensorboard,
            )
        )

        print(f"Dev Acc Para: {paraphrase_accuracy}")
        print(f"DeV Acc SST: {sentiment_accuracy}")
        print(f"DeV Acc STS: {sts_corr}")
        save_model(model, optimizer, args, config, args.filepath)
        print("Model saved to ", args.filepath)
        return

    # Finetuning
    num_batches_per_epoch = args.num_batches_per_epoch
    if num_batches_per_epoch <= 0:
        num_batches_per_epoch = int(len(sst_train_dataloader) / args.gradient_accumulations_sst) + \
                                 int(len(para_train_dataloader) / args.gradient_accumulations_para) + \
                                    int(len(sts_train_dataloader) / args.gradient_accumulations_sts)

    last_improv = -1
    n_batches = 0
    total_num_batches = {"sst": 0, "para": 0, "sts": 0}

    if not args.no_tensorboard:
        for name in ['sst', 'sts', 'para']:
            loss = scheduler.process_named_batch(objects_group, args, name, apply_optimization=False)
            writer.add_scalar("Loss " + name, loss.item(), 0)
            writer.add_scalar("Specific Loss " + name, loss.item(), 0)

    for epoch in range(args.epochs):
        model.train()
        train_loss = {"sst": 0, "para": 0, "sts": 0}
        num_batches = {"sst": 0, "para": 0, "sts": 0}

        if args.projection != 'none':
            if args.task_scheduler == 'pal':
                if args.combine_strategy == 'none':
                    raise ValueError("PAL scheduler requires a combined strategy")
                elif args.combine_strategy == 'force':
                    nb_batches_per_update = 16
                    for i in tqdm(range(int(num_batches_per_epoch / nb_batches_per_update)), desc= f'Train {epoch}', disable= TQDM_DISABLE, smoothing= 0):
                        losses = {'sst': 0, 'para': 0, 'sts': 0}
                        for task in ['sst', 'para', 'sts']:
                            loss = scheduler.process_named_batch(objects_group, args, task, apply_optimization=False)
                            losses[task] += loss
                            num_batches[task] += 1
                            n_batches += 1
                            if args.no_tensorboard:
                                writer.add_scalar("Loss " + task, loss.item(), args.batch_size * n_batches)
                                writer.add_scalar("Specific Loss " + task, loss.item(), args.batch_size * total_num_batches[name])

                        for j in range(nb_batches_per_update -3):
                            task, loss = scheduler.process_one_batch(epoch = epoch + 1, num_epochs= args.epochs, objects_group= objects_group, args= args, apply_optimization= False)
                            losses[task] += loss    #.item()
                            num_batches[task] += 1
                            n_batches += 1
                            if args.no_tensorboard:
                                writer.add_scalar("Loss " + task, losses[task].item(), args.batch_size * n_batches)
                                writer.add_scalar("Specific Loss " + task, losses[task].item(), args.batch_size * total_num_batches[name])

                        losses_opt = [losses[task] / num_batches[task] for task in ['sst', 'para', 'sts']]
                        optimizer.backward(losses_opt)
                        optimizer.step()

                elif args.combine_strategy == 'encourage':
                    nb_batches_per_update = 16
                    alpha = 0.2 + 0.8 * epoch / (args.epochs -1)
                    for i in tqdm(range(int(num_batches_per_epoch / nb_batches_per_update)), desc= f'Train {epoch}', disable= TQDM_DISABLE, smoothing= 0):
                        losses = {'sst': 0, 'para': 0, 'sts': 0}
                        tasks, losses_tasks = scheduler.process_several_batches_with_control(epoch= epoch + 1, num_epochs= args.epochs, objects_group= objects_group, args= args, num_batches= nb_batches_per_update)
                        for j, task in enumerate(tasks):
                            num_batches[task] += 1
                            losses[task] += losses_tasks[j]
                            n_batches += 1
                            if args.no_tensorboard:
                                writer.add_scalar("Loss " + task, losses[task][j].item(), args.batch_size * n_batches)
                                writer.add_scalar("Specific Loss " + task, losses[task][j].item(), args.batch_size * total_num_batches[name])

                        losses = [losses[task] / num_batches[task] **2 for task in ['sst', 'para', 'sts']]
                        optimizer.backward(losses)
                        optimizer.step()

            else:
                # Gradient Surgery / Vaccine without scheduler
                for i in tqdm (range(int(num_batches_per_epoch / 3)), desc= f'Train {epoch}', disable= TQDM_DISABLE, smoothing= 0):
                    losses = []
                    for j , name in enumerate(['sst', 'para', 'sts']):
                        losses.append(scheduler.process_named_batch(objects_group, args, name, apply_optimization= False))
                        n_batches += 1
                        train_loss[name] += losses[-1].item()
                        num_batches[name] += 1
                        total_num_batches[name] += 1
                        if args.no_tensorboard:
                            writer.add_scalar("Loss " + name, losses[-1].item(), args.batch_size * n_batches)
                            writer.add_scalar("Specific Loss " + name, losses[-1].item(), args.batch_size * total_num_batches[name])

                    optimizer.backward(losses)
                    optimizer.step()

        else:
            # Scheduler without projection
            for i in tqdm(range(num_batches_per_epoch), desc= f'Train {epoch}', disable= TQDM_DISABLE, smoothing=0):
                task, loss = scheduler.process_one_batch(epoch= epoch + 1, num_epochs= args.epochs, objects_group= objects_group, args= args)
                n_batches += 1
                train_loss[task] += loss.item()
                num_batches[task] += 1
                total_num_batches[task] += 1
                if args.no_tensorboard:
                    writer.add_scalar("Loss " + task, loss.item(), args.batch_size * n_batches)
                    writer.add_scalar("Specific Loss " + task, loss.item(), args.batch_size * total_num_batches[task])

        # Compute average train loss
        for task in train_loss:
            train_loss[task] = 0 if num_batches[task] == 0 else train_loss[task] / num_batches[task]
            train_loss_logs_epochs[task].append(train_loss[task])

        # Evaluation on dev set
        (paraphrase_accuracy, _, _,
        sentiment_accuracy, _, _,
        sts_corr, _, _) = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device, writer= writer, epoch= epoch, tensorboard= args.no_tensorboard)

        # Keep track of the accuracies for each task and each epoch
        dev_acc_logs_epochs["sst"].append(sentiment_accuracy)
        dev_acc_logs_epochs["para"].append(paraphrase_accuracy)
        dev_acc_logs_epochs["sts"].append(sts_corr)

        # mean of the accuracies
        mean_dev_acc = (sentiment_accuracy + paraphrase_accuracy + sts_corr) / 3

        # Write to tensorboard
        if not args.no_tensorboard:
            writer.add_scalar("Dev Accuracy SST", sentiment_accuracy, epoch)
            writer.add_scalar("Dev Accuracy Para", paraphrase_accuracy, epoch)
            writer.add_scalar("Dev Accuracy STS", sts_corr, epoch)
            writer.add_scalar("Mean Dev Accuracy", mean_dev_acc, epoch)
            writer.add_scalar('Num Batches SST', num_batches['sst'], epoch)
            writer.add_scalar('Num Batches Para', num_batches['para'], epoch)
            writer.add_scalar('Num Batches STS', num_batches['sts'], epoch)
            if epoch ==0:
                writer.add_scalar('Dev Accuracy SST', sentiment_accuracy, 0)
                writer.add_scalar('Dev Accuracy Para', paraphrase_accuracy, 0)
                writer.add_scalar('Dev Accuracy STS', sts_corr, 0)
                writer.add_scalar('Mean Dev Accuracy', mean_dev_acc, 0)
                writer.add_scalar('Num Batches SST', num_batches['sst'], 0)
                writer.add_scalar('Num Batches Para', num_batches['para'], 0)
                writer.add_scalar('Num Batches STS', num_batches['sts'], 0)
            writer.add_scalar('Dev Accuracy SST', sentiment_accuracy, args.batch_size * n_batches)
            writer.add_scalar('Dev Accuracy Para', paraphrase_accuracy, args.batch_size * n_batches)
            writer.add_scalar('Dev Accuracy STS', sts_corr, args.batch_size * n_batches)
            writer.add_scalar('Mean Dev Accuracy', mean_dev_acc, args.batch_size * n_batches)
            writer.add_scalar('Num Batches SST', num_batches['sst'], args.batch_size * n_batches)
            writer.add_scalar('Num Batches Para', num_batches['para'], args.batch_size * n_batches)
            writer.add_scalar('Num Batches STS', num_batches['sts'], args.batch_size * n_batches)

        # Save the model if the mean accuracy is the best
        if mean_dev_acc > best_dev_acc:
            best_dev_acc = mean_dev_acc
            best_dev_accuracies = {
                "sst": sentiment_accuracy,
                "para": paraphrase_accuracy,
                "sts": sts_corr,
            }
            save_model(model, optimizer, args, config, args.filepath)
            print("Model saved to ", args.filepath)
            last_improv = epoch

        print(f'Num batches SST: {num_batches["sst"]}')
        print(f'Num batches Para: {num_batches["para"]}')
        print(f'Num batches STS: {num_batches["sts"]}')
        print(f"Train Loss SST: {train_loss['sst']}")
        print(f"Train Loss Para: {train_loss['para']}")
        print(f"Train Loss STS: {train_loss['sts']}")
        print(f"Dev Accuracy SST: {sentiment_accuracy}")
        print(f"Dev Accuracy Para: {paraphrase_accuracy}")
        print(f"Dev Accuracy STS: {sts_corr}")
        print(f"Mean Dev Accuracy: {mean_dev_acc}")
        print(f"Best Dev Accuracy: {best_dev_acc}")
        print(f'best_dev_accuracies: {best_dev_accuracies['sst']}')
        print(f'best_dev_accuracies: {best_dev_accuracies['para']}')
        print(f'best_dev_accuracies: {best_dev_accuracies['sts']}')
        print(f"Last improvement: {last_improv}")
        print("")

        # apply patience regularization
        if epoch - last_improv >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Write train loss logs epochs and dev acc logs epochs to files
    if args.save_loss_acc_logs:
        with open(args.logdir + ' /train_loss.txt', 'w') as f:
            for key, value in train_loss_logs_epochs.items():
                f.write(f'{key}: {value}\n')
        with open(args.logdir + '/dev_acc.txt', 'w') as f:
            for key, value in dev_acc_logs_epochs.items():
                f.write(f'{key}: {value}\n')


def test_model(args):
    with torch.no_grad():
        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
        saved = torch.load(args.filepath)
        config = saved["model_config"]
        model = MultitaskBERT(config)
        bert_config = BertConfig()
        if args.use_pal:
            BertModelWithPAL.from_BertModel(model.bert, bert_config, train_pal=True)

        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")
        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()

    # Dataset paths
    parser.add_argument("--sst_train", type=str, default="data/sst-sentiment-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/sst-sentiment-dev.csv")
    parser.add_argument(
        "--sst_test", type=str, default="data/sst-sentiment-test-student.csv"
    )

    parser.add_argument(
        "--para_train", type=str, default="data/quora-paraphrase-train.csv"
    )
    parser.add_argument(
        "--para_dev", type=str, default="data/quora-paraphrase-dev.csv"
    )
    parser.add_argument(
        "--para_test", type=str, default="data/quora-paraphrase-test-student.csv"
    )

    parser.add_argument(
        "--sts_train", type=str, default="data/sts-similarity-train.csv"
    )
    parser.add_argument("--sts_dev", type=str, default="data/sts-similarity-dev.csv")
    parser.add_argument(
        "--sts_test", type=str, default="data/sts-similarity-test-student.csv"
    )

    parser.add_argument("--no_tensorboard", action='store_true', help="Dont log to tensorboard")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument(
        "--option",
        type=str,
        choices=("pretrain", "finetune", 'test', 'individual_pretrain', 'optimize'),
        default="finetune",
    )
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--model_name", choices=('bert-base', 'roberta-base', 'bert-large'), default="bert-base")

    parser.add_argument(
        "--sst_dev_out",
        type=str,
        default=(
            f"predictions/sst-sentiment-dev-output.csv")
            )
    parser.add_argument(
        "--sst_test_out",
        type=str,
        default=(
            f"predictions/sst-sentiment-test-output.csv")
    )

    parser.add_argument(
        "--para_dev_out",
        type=str,
        default=(
            "predictions/quora-paraphrase-dev-output.csv")
    )
    parser.add_argument(
        "--para_test_out",
        type=str,
        default=(
            "predictions/quora-paraphrase-test-output.csv")
    )
    parser.add_argument(
        "--sts_dev_out",
        type=str,
        default=(
            "predictions/sts-similarity-dev-output.csv")
    )
    parser.add_argument(
        "--sts_test_out",
        type=str,
        default=(
            "predictions/sts-similarity-test-output.csv"),
    )

    parser.add_argument("--smoketest", action="store_false", help="Run a smoke test")
    args, _ = parser.parse_known_args()

    parser.add_argument("--epochs", type=int, default=10 if not args.smoketest else 1)
    parser.add_argument("--samples_per_epoch", type=int, default= None if not args.smoketest else 10)

    parser.add_argument("--save_loss_acc_logs", type=bool, default=False)
    parser.add_argument("--batch_size", help='This is the simulated batch size using gradient accumulations', type=int, default=128)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2)
    parser.add_argument("--n_hidden_layers", type=int, default=2, help="Number of hidden layers for the classifier")
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--num_batches_per_epoch", type=int, default=-1)
    parser.add_argument("--task_scheduler", type=str, choices=('random', 'round_robin', 'pal', 'para', 'sts', 'sst'), default="round_robin")

    # Optimizations
    parser.add_argument("--use_pal", action='store_true', help="Use additionnal PAL in BERT layers")
    parser.add_argument("--no_train_classifier", action='store_false')
    parser.add_argument("--combine_strategy", type=str, choices=('none', 'encourage', 'force'), default="force")
    parser.add_argument("--use_amp", action='store_true')
    parser.add_argument("--max_batch_size_sst", type=int, default=32)
    parser.add_argument("--max_batch_size_para", type=int, default=16)
    parser.add_argument("--max_batch_size_sts", type=int, default=32)
    parser.add_argument("--projection", type=str, choices=('none', 'pcgrad', 'vaccine'), default="pcgrad")
    parser.add_argument("--beta_vaccine", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, help="Number maximum of epochs without improvement", default=5)
    parser.add_argument("--use_smart_regularization", action='store_true')
    parser.add_argument("--smart_weight_regularization", type=float, default=1e-2)

    parser.add_argument('--no_tensorboard', action='store_false')
    parser.add_argument("--tensorboard_subfolder", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="logs/")



    args = parser.parse_args()
    # TODO: apply data augmentation for the low represtented classes

    # Make sure that the actual batch sizes are not too large
    args.gradient_accumulations_sst = int(np.ceil(args.batch_size / args.max_batch_size_sst))
    args.gradient_accumulations_para = int(np.ceil(args.batch_size / args.max_batch_size_para))
    args.gradient_accumulations_sts = int(np.ceil(args.batch_size / args.max_batch_size_sts))
    args.batch_size_sst = args.batch_size // args.gradient_accumulations_sst
    args.batch_size_para = args.batch_size // args.gradient_accumulations_para
    args.batch_size_sts = args.batch_size // args.gradient_accumulations_sts

    if args.use_amp and not args.use_gpu:
        raise ValueError("AMP requires a GPU")

    # If we are in testing mode, we do not need to train the model
    if args.option == "test":
        if args.lr != 1e-5:
            print("WARNING: " + "Testing mode does not train the model, so the learning rate is not used")
        if args.epochs != 1:
            print("WARNING: " + "Testing mode does not train the model, so the number of epochs is not used")
        if args.num_batches_per_epoch != -1:
            print("WARNING: " + "Testing mode does not train the model, so num_batches_per_epoch is not used")
        if args.task_scheduler != "round_robin":
            print("WARNING: " + "Testing mode does not train the model, so task_scheduler is not used")
        if args.projection != "none":
            print("WARNING: " + "Testing mode does not train the model, so projection is not used")
        if args.hidden_dropout_prob != 0.3:
            print("WARNING: " + "Testing mode does not train the model, so hidden_dropout_prob is not used")
        if args.beta_vaccine != 1e-2:
            print("WARNING: " + "Testing mode does not train the model, so beta_vaccine is not used")
        if args.patience != 5:
            print("WARNING: " + "Testing mode does not train the model, so patience is not used")
        if args.use_amp:
            print("WARNING: " + "Testing mode does not train the model, so use_amp is not used")

    else:
        if args.projection != 'vaccine' and args.beta_vaccine != 1e-2:
            print("WARNING: " + "Beta vaccine is only used when Vaccine is used")
        if args.task_scheduler != 'none' and args.task.scheduler != 'round_robin':
            if args.combine_strategy == 'none':
                print("WARNING: " + "Combine strategy should be specified when using projection and PAL")
                raise ValueError("Combine strategy should be specified when using projection and PAL")
            print("[EXPERIMENTAL] PCGRAD & Vaccine use combine strategy")

    return args

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    args.name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.option}-{args.epochs}-{args.samples_per_epoch}-{args.batch_size}-{args.optimizer}-{args.lr}-{args.scheduler}-{args.hpo}-{args.task}"
    args.filepath = f"models1/multitask_classifier/{args.name}.pt"  # save path

    train_multitask(args)
    test_model(args)
