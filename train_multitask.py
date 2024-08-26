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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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

from evaluation import model_eval_multitask, test_model_multitask
from optimizer import AdamW, SophiaG, SophiaH
from contextlib import nullcontext
from transformers import get_linear_schedule_with_warmup
import math
import warnings
import json
import uuid
from filelock import FileLock

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

        elif args.model_name == "bert-large-uncased":
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
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # initialize params
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True

        self.config = config
        self.attention_layer = AttentionLayer(BERT_HIDDEN_SIZE)

        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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

    def forward(self, input_ids, attention_mask, task_id):

        # if isinstance(self.bert, BertModelWithPAL):
        #     bert_output = self.bert(input_ids, attention_mask, task_id)[
        #         "last_hidden_state"
        #     ]
        # else:
        # _, pooler_output, all_sequences, _ = self.bert(input_ids, attention_mask)
        # bert_output = all_sequences["last_hidden_state"]

        bert_output = self.bert(input_ids, attention_mask)["last_hidden_state"]
        bert_output = self.attention_layer(bert_output)
        bert_output = self.dropout(bert_output)

        if self.config.pooling == None:
            pooled_output = bert_output

        elif self.config.pooling == "max":
            pooled_output, _ = torch.max(bert_output, dim=1)
            # pooled_output = nn.MaxPool1d(kernel_size=self.config.max_position_embeddings)(bert_output).squeeze(-1)

        elif self.config.pooling == "mean":
            pooled_output = torch.mean(bert_output, dim=1)
            # pooled_output = nn.AvgPool1d(kernel_size= self.config.max_position_embeddings)(bert_output).squeeze(-1)

        return pooled_output

    def predict_sentiment(self, input_ids, attention_mask):
        bert_embeddings = self.forward(input_ids, attention_mask, task_id=2)
        sentiment_logits = F.relu(self.sentiment_linear(bert_embeddings))
        sentiment_logits = F.relu(self.sentiment_linear1(sentiment_logits))
        sentiment_logits = F.relu(self.sentiment_linear2(sentiment_logits))
        sentiment_logits = self.sentiment_classifier(sentiment_logits)
        return sentiment_logits

    def predict_paraphrase_train(
        self, input_ids1, attention_mask1, input_ids2, attention_mask2, task_id=1
    ):

        bert_embeddings1 = self.forward(input_ids1, attention_mask1, task_id=task_id)
        bert_embeddings2 = self.forward(input_ids2, attention_mask2, task_id=task_id)

        # apply relu later
        combined_bert_embeddings1 = self.paraphrase_linear(bert_embeddings1)
        combined_bert_embeddings2 = self.paraphrase_linear(bert_embeddings2)

        abs_diff = torch.abs(combined_bert_embeddings1 - combined_bert_embeddings2)
        abs_sum = torch.abs(combined_bert_embeddings1 + combined_bert_embeddings2)
        concatenated_features = torch.cat((abs_diff, abs_sum), dim=1)

        paraphrase_logits = F.relu(self.paraphrase_linear1(concatenated_features))
        paraphrase_logits = F.relu(self.paraphrase_linear2(paraphrase_logits))
        paraphrase_logits = self.paraphrase_classifier(paraphrase_logits)
        return paraphrase_logits

    def predict_paraphrase(
        self, input_ids1, attention_mask1, input_ids2, attention_mask2
    ):
        paraphrase_logits = self.predict_paraphrase_train(
            input_ids1, attention_mask1, input_ids2, attention_mask2, task_id=1
        )
        return paraphrase_logits.argmax(dim=-1)

    def predict_similarity(
        self, input_ids1, attention_mask1, input_ids2, attention_mask2
    ):
        bert_embeddings1 = self.forward(input_ids1, attention_mask1, task_id=0)
        bert_embeddings2 = self.forward(input_ids2, attention_mask2, task_id=0)

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

    if not config.combined_models:
        print(f"Saving the model to {filepath}.")
        torch.save(save_info, filepath)

    else:
        lock_path = filepath.split(".")[0] + ".lock"
        lock = FileLock(lock_path)

        try:
            with lock:
                with open(filepath, "wb") as file:
                    torch.save(save_info, file)
                    print(f"Saving the model to {filepath}.")

        except Exception as e:
            print(f"Error saving model to {filepath}")
            print(e)


def load_model(filepath, model, optimizer, use_gpu, combined_models=False):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if not combined_models:
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

    else:
        lock_path = filepath.split(".")[0] + ".lock"
        lock = FileLock(lock_path)

        try:
            with lock:
                with open(filepath, "rb") as file:
                    print(f"Loading the model from {file}.")
                    save_info = torch.load(
                        file, map_location=torch.device("cuda") if use_gpu else "cpu"
                    )
                    model.load_state_dict(save_info["model"])
                    optimizer.load_state_dict(save_info["optim"])
                    args = save_info["args"]
                    args.use_gpu = use_gpu
                    config = save_info["model_config"]
                    random.setstate(save_info["system_rng"])
                    np.random.set_state(save_info["numpy_rng"])
                    torch.random.set_rng_state(save_info["torch_rng"])

        except Exception as e:
            print(f"Error loading model from {filepath}")
            print(e)

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
    etpc_train_dataloader = None
    etpc_dev_dataloader = None
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
        sts_train_data, args, override_length=args.samples_per_epoch
    )
    sts_dev_data = SentencePairDataset(
        sts_dev_data, args, override_length=10 if args.smoketest else None
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
        "additional_inputs": args.additional_inputs,
        "hidden_size": BERT_HIDDEN_SIZE,
        "num_hidden_layers": N_HIDDEN_LAYERS,
        "max_position_embeddings": args.max_position_embeddings,
        "pooling": args.pooling,
        "layers": args.layers,
        "hidden_dim": HIDDEN_DIM,
        "max_length": args.max_length,
        "combined_models": args.combined_models,
    }

    save_params_dir = args.config_save_dir + args.name + ".json"
    train_params = pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)})

    if args.write_summary:
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
            optimizer, T_max=1, eta_min=1
        )
    else:
        scheduler = None

    best_dev_acc_para = 0
    best_dev_acc_sst = 0
    best_dev_acc_sts = 0

    if args.checkpoint:
        model, optimizer, _, config = load_model(
            args.checkpoint, model, optimizer, args.use_gpu, args.combined_models
        )
    path = (
        args.logdir
        + "/multitask_classifier/"
        + (f"{args.tensorboard_subfolder}/" if args.tensorboard_subfolder else "")
        + args.name
    )
    if args.write_summary:
        writer = SummaryWriter(log_dir=path)
        writer.add_hparams(
            vars(args),
            {},
            run_name="hparams",
        )

    if args.profiler:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(path + "_profiler"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        num_layers = N_HIDDEN_LAYERS

        # unfreeze layers in case of pretraining
        unfreeze = set()
        if args.unfreeze_interval and args.option == "pretrain":
            for name, param in model.named_parameters():
                if not name.startswith("bert_layers"):
                    continue
                layer_num = int(name.split(".")[1])
                unfreeze_up_to = num_layers - epoch // args.unfreeze_interval
                if layer_num >= unfreeze_up_to:
                    unfreeze.add(layer_num)
                else:
                    param.requires_grad = True

        if len(unfreeze) > 0:
            print(f"Unfreezing layers {unfreeze}", file=sys.stderr)

        for sts, para, sst in tqdm(
            zip(sts_train_dataloader, para_train_dataloader, sst_train_dataloader),
            total=total_num_batches,
            disable=TQDM_DISABLE,
            desc=f"train-{epoch}",
        ):

            optimizer.zero_grad()
            sts_loss, para_loss, sst_loss = 0, 0, 0

            # STS
            if train_all_datasets or args.sts:
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                    sts["token_ids_1"],
                    sts["attention_mask_1"],
                    sts["token_ids_2"],
                    sts["attention_mask_2"],
                    sts["labels"],
                )

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                with ctx:
                    sts_logits = model.predict_similarity(
                        b_ids_1, b_mask_1, b_ids_2, b_mask_2
                    )
                    b_labels = b_labels.to(torch.float32)
                    sts_loss = F.mse_loss(sts_logits, b_labels)

            # PARAPHRASE
            if train_all_datasets or args.para:
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                    para["token_ids_1"],
                    para["attention_mask_1"],
                    para["token_ids_2"],
                    para["attention_mask_2"],
                    para["labels"],
                )

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                with ctx:
                    para_logits = model.predict_paraphrase_train(
                        b_ids_1, b_mask_1, b_ids_2, b_mask_2
                    )
                    para_loss = F.cross_entropy(para_logits, b_labels.view(-1))

            # SST
            if train_all_datasets or args.sst:
                b_ids, b_mask, b_labels = (
                    sst["token_ids"],
                    sst["attention_mask"],
                    sst["labels"],
                )

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                with ctx:
                    sst_logits = model.predict_sentiment(b_ids, b_mask)
                    sst_loss = F.cross_entropy(sst_logits, b_labels.view(-1))

            # Combine Losses
            full_loss = sts_loss + para_loss + sst_loss
            full_loss.backward(
                create_graph=True if args.optimizer == "sophiah" else False
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()
            optimizer.zero_grad()
            train_loss += full_loss.item()
            num_batches += 1

            if args.scheduler == "cosine":
                scheduler.step(epoch + num_batches / total_num_batches)

            if args.write_summary:
                writer.add_scalar(
                    "Loss/train",
                    full_loss.item(),
                    epoch * total_num_batches + num_batches,
                )

            if args.profiler:
                prof.step()

        train_loss = train_loss / num_batches

        if args.write_summary:
            writer.add_scalar("Loss/train", train_loss, epoch)

        # Evaluate
        (
            para_train_acc,
            _,
            _,
            sst_train_acc,
            _,
            _,
            sts_train_acc,
            _,
            _,
            etpc_train_acc,
            _,
            _,
        ) = model_eval_multitask(
            sst_train_dataloader,
            para_train_dataloader,
            sts_train_dataloader,
            etpc_train_dataloader,
            model=model,
            device=device,
            task=args.task,
        )

        para_dev_acc, _, _, sst_dev_acc, _, _, sts_dev_acc, _, _, etpc_dev_acc, _, _ = (
            model_eval_multitask(
                sst_dev_dataloader,
                para_dev_dataloader,
                sts_dev_dataloader,
                etpc_dev_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )

        if args.write_summary:
            writer.add_scalar("para_acc/train/Epochs", para_train_acc, epoch)
            writer.add_scalar("para_acc/dev/Epochs", para_dev_acc, epoch)
            writer.add_scalar("sst_acc/train/Epochs", sst_train_acc, epoch)
            writer.add_scalar("sst_acc/dev/Epochs", sst_dev_acc, epoch)
            writer.add_scalar("sts_acc/train/Epochs", sts_train_acc, epoch)
            writer.add_scalar("sts_acc/dev/Epochs", sts_dev_acc, epoch)

        if (
            para_dev_acc > best_dev_acc_para
            and sst_dev_acc > best_dev_acc_sst
            and sts_dev_acc > best_dev_acc_sts
        ):
            best_dev_acc_para = para_dev_acc
            best_dev_acc_sst = sst_dev_acc
            best_dev_acc_sts = sts_dev_acc
            if args.hpo:
                args.filepath = f"ray_checkpoint/{session.get_trial_name()}-{epoch}.pt"

            save_model(model, optimizer, args, config, args.filepath)
            print("Model saved at", args.filepath)

        else:
            print(
                f"Model not saved since the metrics are did not exceed the starting point. Best Dev Acc: Para: {best_dev_acc_para:.3f}, SST: {best_dev_acc_sst:.3f}, STS: {best_dev_acc_sts:.3f}"
            )

        train_acc = para_train_acc + sst_train_acc + sts_train_acc
        dev_acc = para_dev_acc + sst_dev_acc + sts_dev_acc

        if args.hpo:
            session.report(
                {
                    "sst_dev_acc": sst_dev_acc,
                    "para_dev_acc": para_dev_acc,
                    "sts_dev_acc": sts_dev_acc,
                    "mean_dev_acc": dev_acc / 3,
                    "mean_train_acc": train_acc / 3,
                }
            )
        if args.scheduler == "plateau":
            scheduler.step(dev_acc)

        if args.write_summary:
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("acc/train/Epochs", train_acc, epoch)
            writer.add_scalar("acc/dev/Epochs", dev_acc, epoch)
        print(
            f"Epoch {epoch}: Avg Train Loss :: {train_loss/3 :.3f}, Avg Train Acc :: {train_acc/3:.3f}, Avg Dev Acc :: {dev_acc/3:.3f}"
        )


def test_model(args):
    with torch.no_grad():
        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
        if not args.combined_models:
            saved = torch.load(args.filepath)
            config = saved["model_config"]
            model = MultitaskBERT(config)
            bert_config = BertConfig()
            if args.use_pal:
                BertModelWithPAL.from_BertModel(model.bert, bert_config, train_pal=True)

            model.load_state_dict(saved["model"])
            model = model.to(device)
            print(f"Loaded model to test from {args.filepath}")

        else:
            lock_path = args.filepath.split(".")[0] + ".lock"
            lock = FileLock(lock_path)
            try:
                with lock:
                    with open(args.filepath, "rb") as file:
                        print(f"Loaded model to test from {file}")
                        saved = torch.load(file, map_location=device)
                        config = saved["model_config"]
                        model = MultitaskBERT(config)
                        model.load_state_dict(saved["model"])
                        model = model.to(device)

            except Exception as e:
                print(f"Error loading model from {e}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()

    # Training task
    parser.add_argument(
        "--task",
        type=str,
        help='choose between "sst","sts","qqp","etpc","multitask" to train for different tasks ',
        choices=("sst", "sts", "qqp", "etpc", "multitask"),
        default="multitask",
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

    parser.add_argument(
        "--etpc_train", type=str, default="data/etpc-paraphrase-train.csv"
    )
    parser.add_argument("--etpc_dev", type=str, default="data/etpc-paraphrase-dev.csv")
    parser.add_argument(
        "--etpc_test",
        type=str,
        default="data/etpc-paraphrase-detection-test-student.csv",
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
    parser.add_argument("--unfreeze_interval", type=int, default=None)
    parser.add_argument("--additional_inputs", action="store_false")
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--sts", action="store_true")
    parser.add_argument("--sst", action="store_true")
    parser.add_argument("--para", action="store_true")

    parser.add_argument("--logdir", type=str, default="logdir")
    parser.add_argument(
        "--config_save_dir", type=str, default="./config_params/bert/multitask/"
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        choices=("adamw", "sophiah"),
        default="adamw",
    )
    parser.add_argument(
        "--rho", type=float, default=0.05, help="rho for SophiaH optimizer"
    )
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument(
        "--hess_interval",
        type=int,
        default=10,
        help="Hessian update interval for SophiaH",
    )
    parser.add_argument("--smoketest", action="store_false", help="Run a smoke test")

    args, _ = parser.parse_known_args()

    parser.add_argument("--epochs", type=int, default=20 if not args.smoketest else 1)

    # hyper parameters
    parser.add_argument(
        "--batch_size",
        help="sst: 64 can fit a 12GB GPU",
        type=int,
        default=64 if not args.smoketest else 64,
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument(
        "--clip", type=float, default=0.25, help="value used gradient clipping"
    )
    parser.add_argument(
        "--samples_per_epoch", type=int, default=20_000 if not args.smoketest else 10
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
        default=(
            8e-5 * (1 / args.rho if args.optimizer == "sophiah" else 1)
            if args.option == "finetune"
            else 1e-3 * (1 / args.rho if args.optimizer == "sophiah" else 1)
        ),
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tensorboard_subfolder", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=("plateau", "cosine", "linear_warmup"),
    )
    parser.add_argument(
        "--hpo", action="store_false", help="Activate hyperparameter optimization"
    )
    parser.add_argument(
        "--hpo_trials",
        type=int,
        default=10 if not args.smoketest else 1,
        help="Number of trials for HPO",
    )

    parser.add_argument(
        "--pooling",
        default=None,
        choices=["mean", "max", None],
    )
    parser.add_argument(
        "--layers", default=None, type=int, nargs="+", help="Layers to train"
    )

    parser.add_argument(
        "--write_summary",
        action="store_false",
        help="Write summary to tensorboard",
    )

    args, _ = parser.parse_known_args()
    record_time = datetime.now().strftime("%m%d-%H%M")
    path = f"{args.option}-{args.epochs}-{args.samples_per_epoch}-{args.batch_size}-{args.optimizer}-{args.lr}-{args.scheduler}-{args.task}"

    predictions_path = f"./predictions/bert/multitask/{path}"
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path, exist_ok=True)

    # Output paths
    parser.add_argument(
        "--sst_dev_out",
        type=str,
        default=(
            f"predictions/bert/sst-sentiment-dev-output.csv"
            if not args.task == "multitask"
            else f"{predictions_path}/{record_time}_sst-sentiment-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sst_test_out",
        type=str,
        default=(
            f"predictions/bert/sst-sentiment-test-output.csv"
            if not args.task == "multitask"
            else f"{predictions_path}/{record_time}_sst-sentiment-test-output.csv"
        ),
    )

    parser.add_argument(
        "--quora_dev_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-dev-output.csv"
            if not args.task == "multitask"
            else f"{predictions_path}/{record_time}_quora-paraphrase-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--quora_test_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-test-output.csv"
            if not args.task == "multitask"
            else f"{predictions_path}/{record_time}_quora-paraphrase-test-output.csv"
        ),
    )

    parser.add_argument(
        "--sts_dev_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-dev-output.csv"
            if not args.task == "multitask"
            else f"{predictions_path}/{record_time}_sts-similarity-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sts_test_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-test-output.csv"
            if not args.task == "multitask"
            else f"{predictions_path}/{record_time}_sts-similarity-test-output.csv"
        ),
    )
    parser.add_argument(
        "--add_layers",
        action="store_false",
        help="Add extra layers to the model",
    )
    parser.add_argument(
        "--combined_models",
        action="store_false",
        help="Combine models for multitask learning",
    )
    parser.add_argument(
        "--improve_dir",
        type=str,
        default="./improve_dir",
        help="path to save the params",
    )
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        default=512,
        help="Max position embeddings",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="max length for the model",
    )
    parser.add_argument(
        "--use_pal",
        action="store_false",
        help="Use PAL for the model",
    )
    parser.add_argument(
        "--model_name",
        default="bert-large-uncased",
        help="Model name to use",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    if not os.path.exists("./logdir/multitas_classifier"):
        os.makedirs("./logdir/multitask_classifier", exist_ok=True)
    if not os.path.exists("./ray_checkpoint"):
        os.makedirs("./ray_checkpoint", exist_ok=True)
    if not os.path.exists("./config_params/bert/multitask"):
        os.makedirs("./config_params/bert/multitask", exist_ok=True)

    args = get_args()
    seed_everything(args.seed)
    args.name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.option}-{args.epochs}-{args.samples_per_epoch}-{args.batch_size}-{args.optimizer}-{args.lr}-{args.scheduler}-{args.hpo}-{args.task}"
    args.filepath = f"models1/multitask_classifier/{args.name}.pt"  # save path

    if args.hpo:

        def format_value(value):
            if isinstance(value, float):
                return "{:.2e}".format(value)
            return value

        import ray
        from ray import tune, air
        from ray.air import session
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune.search.optuna import OptunaSearch

        config = vars(args)
        tune_config = {
            "weight_decay": tune.choice([0.1, 0.01, 0.001, 0.0001, 0]),
            "hidden_dropout_prob": tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            "clip": tune.loguniform(0.01, 10),
            "lr": tune.choice([2e-5, 5e-8, 8e-5, 1e-6]),
        }
        config.update(tune_config)

        scheduler = ASHAScheduler(
            metric="mean_dev_acc",
            mode="max",
            max_t=args.epochs,
            grace_period=min(2, args.epochs),
            reduction_factor=2,
        )

        algo = OptunaSearch(
            metric=["sst_dev_acc", "para_dev_acc", "sts_dev_acc"], mode=["max"] * 3
        )

        ray.init(log_to_driver=False)

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(train_multitask),
                resources={"cpu": 16, "gpu": 2 if args.use_gpu else 0},
            ),
            tune_config=tune.TuneConfig(
                num_samples=args.hpo_trials,
                scheduler=scheduler,
                search_alg=algo,
                max_concurrent_trials=2,
                trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}_{'.'.join(f'{k}={format_value(v)}' for k, v in trial.evaluated_params.items())}",
            ),
            run_config=air.RunConfig(
                log_to_file="std.log",
                verbose=1,
            ),
            param_space=config,
        )
        results = tuner.fit()
        best_result = results.get_best_result("mean_dev_acc", "max")

        separator = "=" * 60
        print(separator)
        print("   Best Multitask BERT Model Configuration")
        print(separator)
        filtered_vars = {
            k: v
            for k, v in best_result.config.items()
            if "csv" not in str(v) and "pt" not in str(v)
        }
        print(pformat(filtered_vars))
        print(separator)
        print(f"Best mean_dev_acc: {best_result.metrics.get('mean_dev_acc', 'N/A')}")

        config["weight_decay"] = best_result.config["weight_decay"]
        config["hidden_dropout_prob"] = best_result.config["hidden_dropout_prob"]
        config["clip"] = best_result.config["clip"]
        config["lr"] = best_result.config["lr"]

    train_multitask(args)
    test_model(args)
