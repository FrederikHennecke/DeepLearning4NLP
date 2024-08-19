#!/usr/bin/env python3

"""
This module contains our Dataset classes and functions to load the 3 datasets we're using.

You should only need to call load_multitask_data to get the training and dev examples
to train your model.
"""


import csv

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from tokenizer import BertTokenizer
from pprint import pprint
import random


def preprocess_string(s):
    return " ".join(
        s.lower()
        .replace(".", " .")
        .replace("?", " ?")
        .replace(",", " ,")
        .replace("'", " '")
        .split()
    )


class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset, args, override_length=None):
        self.override_length = override_length
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def real_len(self):
        return len(self.dataset)

    def __len__(self):
        if self.override_length is None:
            return self.real_len()
        return self.override_length

    def __getitem__(self, idx):
        if self.override_length is not None:
            return random.choice(self.dataset)
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(
            sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.p.max_length,
        )
        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def chunk_input(self, input_ids, attention_mask, segment_length):
        batch_size, seq_length = input_ids.shape
        num_segments = (seq_length + segment_length - 1) // segment_length
        padded_length = num_segments * segment_length

        input_ids_padded = F.pad(input_ids, (0, padded_length - seq_length), value=0)
        attention_mask_padded = F.pad(
            attention_mask, (0, padded_length - seq_length), value=0
        )

        input_ids_segmented = input_ids_padded.view(
            batch_size, num_segments, segment_length
        )
        attention_mask_segmented = attention_mask_padded.view(
            batch_size, num_segments, segment_length
        )

        # input_ids = torch.cat([input_ids, torch.zeros(batch_size, segment_length - seq_length % segment_length)], dim=1)
        # attention_mask = torch.cat([attention_mask, torch.zeros(batch_size, segment_length - seq_length % segment_length)], dim=1)
        # input_ids = input_ids.view(batch_size, num_segments, segment_length)
        # attention_mask = attention_mask.view(batch_size, num_segments, segment_length)

        return input_ids_segmented, attention_mask_segmented

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)
        # token_ids, attention_mask = self.chunk_input(
        #     token_ids, attention_mask, self.p.segment_length
        # )

        batched_data = {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sents": sents,
            "sent_ids": sent_ids,
        }

        return batched_data


class SentenceClassificationTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def real_len(self):
        return len(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(
            sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.p.max_length,
        )
        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])

        return token_ids, attention_mask, sents, sent_ids

    def chunk_input(self, input_ids, attention_mask, segment_length):
        batch_size, seq_length = input_ids.shape
        num_segments = (seq_length + segment_length - 1) // segment_length
        padded_length = num_segments * segment_length

        input_ids_padded = F.pad(input_ids, (0, padded_length - seq_length), value=0)
        attention_mask_padded = F.pad(
            attention_mask, (0, padded_length - seq_length), value=0
        )

        input_ids_segmented = input_ids_padded.view(
            batch_size, num_segments, segment_length
        )
        attention_mask_segmented = attention_mask_padded.view(
            batch_size, num_segments, segment_length
        )

        return input_ids_segmented, attention_mask_segmented

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)
        # token_ids, attention_mask = self.chunk_input(
        #     token_ids, attention_mask, self.p.segment_length
        # )

        batched_data = {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "sents": sents,
            "sent_ids": sent_ids,
        }

        return batched_data


class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression=False, override_length=None):
        self.override_length = override_length
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def real_len(self):
        return len(self.dataset)

    def __len__(self):
        if self.override_length is None:
            return self.real_len()
        return self.override_length

    def __getitem__(self, idx):
        if self.override_length is not None:
            return random.choice(self.dataset)
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(
            sent1,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.p.max_length,
        )
        encoding2 = self.tokenizer(
            sent2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.p.max_length,
        )

        token_ids = torch.LongTensor(encoding1["input_ids"])
        attention_mask = torch.LongTensor(encoding1["attention_mask"])

        token_ids2 = torch.LongTensor(encoding2["input_ids"])
        attention_mask2 = torch.LongTensor(encoding2["attention_mask"])

        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)

        return (
            token_ids,
            attention_mask,
            token_ids2,
            attention_mask2,
            labels,
            sent_ids,
        )

    def chunk_input(
        self,
        input_ids,
        attention_mask,
        input_ids2,
        attention_mask2,
        segment_length,
    ):
        batch_size, seq_length = input_ids.shape
        batch_size2, seq_length2 = input_ids2.shape

        num_segments = (seq_length + segment_length - 1) // segment_length
        num_segments2 = (seq_length2 + segment_length - 1) // segment_length

        padded_length = num_segments * segment_length
        padded_length2 = num_segments2 * segment_length

        input_ids_padded = F.pad(input_ids, (0, padded_length - seq_length), value=0)
        attention_mask_padded = F.pad(
            attention_mask, (0, padded_length - seq_length), value=0
        )

        input_ids2_padded = F.pad(
            input_ids2, (0, padded_length2 - seq_length2), value=0
        )
        attention_mask2_padded = F.pad(
            attention_mask2, (0, padded_length2 - seq_length2), value=0
        )

        input_ids_segmented = input_ids_padded.view(
            batch_size, num_segments, segment_length
        )
        attention_mask_segmented = attention_mask_padded.view(
            batch_size, num_segments, segment_length
        )

        input_ids2_segmented = input_ids2_padded.view(
            batch_size2, num_segments2, segment_length
        )
        attention_mask2_segmented = attention_mask2_padded.view(
            batch_size2, num_segments2, segment_length
        )

        return (
            input_ids_segmented,
            attention_mask_segmented,
            input_ids2_segmented,
            attention_mask2_segmented,
        )

    def collate_fn(self, all_data):
        (
            token_ids,
            attention_mask,
            token_ids2,
            attention_mask2,
            labels,
            sent_ids,
        ) = self.pad_data(all_data)
        # (
        #     token_ids,
        #     attention_mask,
        #     token_ids2,
        #     attention_mask2,
        # ) = self.chunk_input(
        #     token_ids,
        #     attention_mask,
        #     token_ids2,
        #     attention_mask2,
        #     self.p.segment_length,
        # )

        batched_data = {
            "token_ids_1": token_ids,
            "attention_mask_1": attention_mask,
            "token_ids_2": token_ids2,
            "attention_mask_2": attention_mask2,
            "labels": labels,
            "sent_ids": sent_ids,
        }

        return batched_data


class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(
            sent1,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.p.max_length,
        )
        encoding2 = self.tokenizer(
            sent2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.p.max_length,
        )

        token_ids = torch.LongTensor(encoding1["input_ids"])
        attention_mask = torch.LongTensor(encoding1["attention_mask"])

        token_ids2 = torch.LongTensor(encoding2["input_ids"])
        attention_mask2 = torch.LongTensor(encoding2["attention_mask"])

        return (
            token_ids,
            attention_mask,
            token_ids2,
            attention_mask2,
            sent_ids,
        )

    def chunk_input(
        self,
        input_ids,
        attention_mask,
        input_ids2,
        attention_mask2,
        segment_length,
    ):
        batch_size, seq_length = input_ids.shape
        batch_size2, seq_length2 = input_ids2.shape

        num_segments = (seq_length + segment_length - 1) // segment_length
        num_segments2 = (seq_length2 + segment_length - 1) // segment_length

        padded_length = num_segments * segment_length
        padded_length2 = num_segments2 * segment_length

        input_ids_padded = F.pad(input_ids, (0, padded_length - seq_length), value=0)
        attention_mask_padded = F.pad(
            attention_mask, (0, padded_length - seq_length), value=0
        )

        input_ids2_padded = F.pad(
            input_ids2, (0, padded_length2 - seq_length2), value=0
        )
        attention_mask2_padded = F.pad(
            attention_mask2, (0, padded_length2 - seq_length2), value=0
        )

        input_ids_segmented = input_ids_padded.view(
            batch_size, num_segments, segment_length
        )
        attention_mask_segmented = attention_mask_padded.view(
            batch_size, num_segments, segment_length
        )

        input_ids2_segmented = input_ids2_padded.view(
            batch_size2, num_segments2, segment_length
        )
        attention_mask2_segmented = attention_mask2_padded.view(
            batch_size2, num_segments2, segment_length
        )

        return (
            input_ids_segmented,
            attention_mask_segmented,
            input_ids2_segmented,
            attention_mask2_segmented,
        )

    def collate_fn(self, all_data):
        (
            token_ids,
            attention_mask,
            token_ids2,
            attention_mask2,
            sent_ids,
        ) = self.pad_data(all_data)

        # (
        #     token_ids,
        #     attention_mask,
        #     token_ids2,
        #     attention_mask2,
        # ) = self.chunk_input(
        #     token_ids,
        #     attention_mask,
        #     token_ids2,
        #     attention_mask2,
        #     self.p.segment_length,
        # )

        batched_data = {
            "token_ids_1": token_ids,
            "attention_mask_1": attention_mask,
            "token_ids_2": token_ids2,
            "attention_mask_2": attention_mask2,
            "sent_ids": sent_ids,
        }

        return batched_data


def load_multitask_data(
    sst_filename, quora_filename, sts_filename, etpc_filename, split="train"
):
    sst_data = []
    num_labels = {}
    if split == "test":
        with open(sst_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent = record["sentence"].lower().strip()
                sent_id = record["id"].lower().strip()
                sst_data.append((sent, sent_id))
    else:
        with open(sst_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t", quoting=csv.QUOTE_NONE):
                try:
                    sent = record["sentence"].lower().strip()
                    sent_id = record["id"].lower().strip()
                    label = int(record["sentiment"].strip())
                    if label not in num_labels:
                        num_labels[label] = len(num_labels)
                    sst_data.append((sent, label, sent_id))
                except:
                    pprint(record)

    print(f"Loaded {len(sst_data)} {split} examples from {sst_filename}")

    quora_data = []
    if split == "test":
        with open(quora_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                quora_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        sent_id,
                    )
                )

    else:
        with open(quora_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                try:
                    sent_id = record["id"].lower().strip()
                    quora_data.append(
                        (
                            preprocess_string(record["sentence1"]),
                            preprocess_string(record["sentence2"]),
                            int(float(record["is_duplicate"])),
                            sent_id,
                        )
                    )
                except:
                    pass

    print(f"Loaded {len(quora_data)} {split} examples from {quora_filename}")

    sts_data = []
    if split == "test":
        with open(sts_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                sts_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        sent_id,
                    )
                )
    else:
        with open(sts_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                sts_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        float(record["similarity"]),
                        sent_id,
                    )
                )

    print(f"Loaded {len(sts_data)} {split} examples from {sts_filename}")

    etpc_data = []
    if split == "test":
        with open(etpc_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                etpc_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        sent_id,
                    )
                )

    else:
        with open(etpc_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                try:
                    sent_id = record["id"].lower().strip()
                    etpc_data.append(
                        (
                            preprocess_string(record["sentence1"]),
                            preprocess_string(record["sentence2"]),
                            list(
                                map(
                                    int,
                                    record["paraphrase_types"].strip("][").split(", "),
                                )
                            ),
                            sent_id,
                        )
                    )
                except:
                    pass

    print(f"Loaded {len(etpc_data)} {split} examples from {etpc_filename}")

    return sst_data, num_labels, quora_data, sts_data, etpc_data
