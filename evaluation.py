#!/usr/bin/env python3

"""
Model evaluation functions.

When training your multitask model, you will find it useful to run
model_eval_multitask to be able to evaluate your model on the 3 tasks in the
development set.

Before submission, your code needs to call test_model_multitask(args, model, device) to generate
your predictions. We'll evaluate these predictions against our labels on our end,
which is how the leaderboard will be updated.
The provided test_model() function in multitask_classifier.py **already does this for you**,
so unless you change it you shouldn't need to call anything from here
explicitly aside from model_eval_multitask.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data,
)
import os, json
from datetime import datetime

TQDM_DISABLE = True


# Perform model evaluation
def model_eval_multitask(
    sst_dataloader,
    quora_dataloader,
    sts_dataloader,
    etpc_dataloader,
    model,
    device,
    task,
):
    model.eval()  # switch to eval model, will turn off randomness like dropout

    with torch.no_grad():
        quora_y_true = []
        quora_y_pred = []
        quora_sent_ids = []

        # Evaluate paraphrase detection.
        if task == "qqp" or task == "multitask":
            for step, batch in enumerate(
                tqdm(quora_dataloader, desc="eval", disable=TQDM_DISABLE)
            ):
                (b_ids1, b_mask1, b_ids2, b_mask2, b_labels, b_sent_ids) = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
                    batch["sent_ids"],
                )

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)

                logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                y_hat = logits.sigmoid().round().flatten().cpu().numpy()
                b_labels = b_labels.flatten().cpu().numpy()

                quora_y_pred.extend(y_hat)
                quora_y_true.extend(b_labels)
                quora_sent_ids.extend(b_sent_ids)

        if task == "qqp" or task == "multitask":
            quora_accuracy = np.mean(np.array(quora_y_pred) == np.array(quora_y_true))
        else:
            print("qqp task not included in the training and evaluation")
            quora_accuracy = 0

        sts_y_true = []
        sts_y_pred = []
        sts_sent_ids = []

        # Evaluate semantic textual similarity.
        if task == "sts" or task == "multitask":
            for step, batch in enumerate(
                tqdm(sts_dataloader, desc="eval", disable=TQDM_DISABLE)
            ):
                (b_ids1, b_mask1, b_ids2, b_mask2, b_labels, b_sent_ids) = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
                    batch["sent_ids"],
                )

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)

                logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                y_hat = logits.flatten().cpu().numpy()
                b_labels = b_labels.flatten().cpu().numpy()

                sts_y_pred.extend(y_hat)
                sts_y_true.extend(b_labels)
                sts_sent_ids.extend(b_sent_ids)

        if task == "sts" or task == "multitask":
            pearson_mat = np.corrcoef(sts_y_pred, sts_y_true)
            sts_corr = pearson_mat[1][0]
        else:
            print("sts task not included in the training and evaluation")
            sts_corr = 0

        sst_y_true = []
        sst_y_pred = []
        sst_sent_ids = []

        # Evaluate sentiment classification.
        if task == "sst" or task == "multitask":
            for step, batch in enumerate(
                tqdm(sst_dataloader, desc="eval", disable=TQDM_DISABLE)
            ):
                b_ids, b_mask, b_labels, b_sent_ids = (
                    batch["token_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                    batch["sent_ids"],
                )

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)

                logits = model.predict_sentiment(b_ids, b_mask)
                y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
                b_labels = b_labels.flatten().cpu().numpy()

                sst_y_pred.extend(y_hat)
                sst_y_true.extend(b_labels)
                sst_sent_ids.extend(b_sent_ids)

        if task == "sst" or task == "multitask":
            sst_accuracy = np.mean(np.array(sst_y_pred) == np.array(sst_y_true))
        else:
            print("sst task not included in the training and evaluation")
            sst_accuracy = 0

        etpc_y_true = []
        etpc_y_pred = []
        etpc_sent_ids = []

        # Evaluate paraphrase type detection.
        if task == "etpc":
            for step, batch in enumerate(
                tqdm(etpc_dataloader, desc="eval", disable=TQDM_DISABLE)
            ):
                (b_ids1, b_mask1, b_ids2, b_mask2, b_labels, b_sent_ids) = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
                    batch["sent_ids"],
                )

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)

                logits = model.predict_paraphrase_types(
                    b_ids1, b_mask1, b_ids2, b_mask2
                )
                y_hat = logits.sigmoid().round().cpu().numpy()
                b_labels = b_labels.cpu().numpy()

                etpc_y_pred.extend(y_hat)
                etpc_y_true.extend(b_labels)
                etpc_sent_ids.extend(b_sent_ids)

        if task == "etpc":
            correct_pred = np.all(
                np.array(etpc_y_pred) == np.array(etpc_y_true), axis=1
            ).astype(int)
            etpc_accuracy = np.mean(correct_pred)
            etpc_y_pred = etpc_y_pred.tolist()
        else:
            print("etpc task not included in the multitask training and evaluation")
            etpc_accuracy = None
            etpc_sent_ids = None

        if task == "qqp" or task == "multitask":
            print(f"Paraphrase detection accuracy: {quora_accuracy:.3f}")
        if task == "sst" or task == "multitask":
            print(f"Sentiment classification accuracy: {sst_accuracy:.3f}")
        if task == "sts" or task == "multitask":
            print(f"Semantic Textual Similarity correlation: {sts_corr:.3f}")
        if task == "etpc":
            print(f"Paraphrase Type detection accuracy: {etpc_accuracy:.3f}")
        print("\n")

    model.train()  # switch back to train model

    return (
        quora_accuracy,
        quora_y_pred,
        quora_sent_ids,
        sst_accuracy,
        sst_y_pred,
        sst_sent_ids,
        sts_corr,
        sts_y_pred,
        sts_sent_ids,
        etpc_accuracy,
        etpc_y_pred,
        etpc_sent_ids,
    )


# Perform model evaluation in terms by averaging accuracies across tasks.
def model_eval_test_multitask(
    sst_dataloader,
    quora_dataloader,
    sts_dataloader,
    etpc_dataloader,
    model,
    device,
    task,
):
    model.eval()  # switch to eval model, will turn off randomness like dropout

    with torch.no_grad():
        quora_y_pred = []
        quora_sent_ids = []
        # Evaluate paraphrase detection.
        if task == "qqp" or task == "multitask":
            for step, batch in enumerate(
                tqdm(quora_dataloader, desc="eval", disable=TQDM_DISABLE)
            ):
                (b_ids1, b_mask1, b_ids2, b_mask2, b_sent_ids) = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["sent_ids"],
                )

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)

                logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                y_hat = logits.sigmoid().round().flatten().cpu().numpy()

                quora_y_pred.extend(y_hat)
                quora_sent_ids.extend(b_sent_ids)

        sts_y_pred = []
        sts_sent_ids = []

        # Evaluate semantic textual similarity.
        if task == "sts" or task == "multitask":
            for step, batch in enumerate(
                tqdm(sts_dataloader, desc="eval", disable=TQDM_DISABLE)
            ):
                (b_ids1, b_mask1, b_ids2, b_mask2, b_sent_ids) = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["sent_ids"],
                )

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)

                logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                y_hat = logits.flatten().cpu().numpy()

                sts_y_pred.extend(y_hat)
                sts_sent_ids.extend(b_sent_ids)

        sst_y_pred = []
        sst_sent_ids = []

        # Evaluate sentiment classification.
        if task == "sst" or task == "multitask":
            for step, batch in enumerate(
                tqdm(sst_dataloader, desc="eval", disable=TQDM_DISABLE)
            ):
                b_ids, b_mask, b_sent_ids = (
                    batch["token_ids"],
                    batch["attention_mask"],
                    batch["sent_ids"],
                )

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)

                logits = model.predict_sentiment(b_ids, b_mask)
                y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()

                sst_y_pred.extend(y_hat)
                sst_sent_ids.extend(b_sent_ids)

        etpc_y_pred = []
        etpc_sent_ids = []
        if task == "etpc":
            for step, batch in enumerate(
                tqdm(etpc_dataloader, desc="eval", disable=TQDM_DISABLE)
            ):
                (b_ids1, b_mask1, b_ids2, b_mask2, b_sent_ids) = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["sent_ids"],
                )

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)

                logits = model.predict_paraphrase_types(
                    b_ids1, b_mask1, b_ids2, b_mask2
                )
                y_hat = logits.sigmoid().round().cpu().numpy().astype(int).tolist()

                etpc_y_pred.extend(y_hat)
                etpc_sent_ids.extend(b_sent_ids)
        else:
            print("etpc task not included in the multitask training and evaluation")
            etpc_y_pred = None
            etpc_sent_ids = None

        return (
            quora_y_pred,
            quora_sent_ids,
            sst_y_pred,
            sst_sent_ids,
            sts_y_pred,
            sts_sent_ids,
            etpc_y_pred,
            etpc_sent_ids,
        )


def test_model_multitask(args, model, device, config):
    sst_test_data, _, quora_test_data, sts_test_data, etpc_test_data = (
        load_multitask_data(
            args.sst_test, args.quora_test, args.sts_test, args.etpc_test, split="test"
        )
    )

    sst_dev_data, _, quora_dev_data, sts_dev_data, etpc_dev_data = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="dev"
    )

    sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_test_dataloader = DataLoader(
        sst_test_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=sst_test_data.collate_fn,
    )
    sst_dev_dataloader = DataLoader(
        sst_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sst_dev_data.collate_fn,
    )

    quora_test_data = SentencePairTestDataset(quora_test_data, args)
    quora_dev_data = SentencePairDataset(quora_dev_data, args)

    quora_test_dataloader = DataLoader(
        quora_test_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=quora_test_data.collate_fn,
    )
    quora_dev_dataloader = DataLoader(
        quora_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=quora_dev_data.collate_fn,
    )

    sts_test_data = SentencePairTestDataset(sts_test_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_test_dataloader = DataLoader(
        sts_test_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=sts_test_data.collate_fn,
    )
    sts_dev_dataloader = DataLoader(
        sts_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sts_dev_data.collate_fn,
    )

    etpc_test_data = SentencePairTestDataset(etpc_test_data, args)
    etpc_dev_data = SentencePairDataset(etpc_dev_data, args)

    etpc_test_dataloader = DataLoader(
        etpc_test_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=etpc_test_data.collate_fn,
    )
    etpc_dev_dataloader = DataLoader(
        etpc_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=etpc_dev_data.collate_fn,
    )

    task = args.task

    (
        dev_quora_accuracy,
        dev_quora_y_pred,
        dev_quora_sent_ids,
        dev_sst_accuracy,
        dev_sst_y_pred,
        dev_sst_sent_ids,
        dev_sts_corr,
        dev_sts_y_pred,
        dev_sts_sent_ids,
        dev_etpc_accuracy,
        dev_etpc_y_pred,
        dev_etpc_sent_ids,
    ) = model_eval_multitask(
        sst_dev_dataloader,
        quora_dev_dataloader,
        sts_dev_dataloader,
        etpc_dev_dataloader,
        model,
        device,
        task,
    )

    (
        test_quora_y_pred,
        test_quora_sent_ids,
        test_sst_y_pred,
        test_sst_sent_ids,
        test_sts_y_pred,
        test_sts_sent_ids,
        test_etpc_y_pred,
        test_etpc_sent_ids,
    ) = model_eval_test_multitask(
        sst_test_dataloader,
        quora_test_dataloader,
        sts_test_dataloader,
        etpc_test_dataloader,
        model,
        device,
        task,
    )

    if task == "sst":
        print(f"dev sentiment acc :: {dev_sst_accuracy :.3f}")

        if dev_sst_accuracy > 0.522:
            print("Your score is higher than the baseline for sst task")
            print("saving params")
            if not os.path.exists(args.sst_improve_dir):
                os.makedirs(args.sst_improve_dir)

            with open(
                os.path.join(
                    args.sst_improve_dir,
                    f"sst_params_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
                ),
                "w",
            ) as f:
                json.dump(config, f)

            with open(args.sst_dev_out, "w+") as f:
                f.write("id,Predicted_Sentiment\n")
                for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                    f.write(f"{p}\t{s}\n")

            with open(args.sst_test_out, "w+") as f:
                f.write("id,Predicted_Sentiment\n")
                for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                    f.write(f"{p}\t{s}\n")
        else:
            print("your score is lower than the baseline for sst task")

    if task == "qqp":
        print(f"dev paraphrase acc :: {dev_quora_accuracy :.3f}")

        if dev_quora_accuracy > 0.806:
            print("Your score is higher than the baseline for qqp task")
            print("saving params")
            if not os.path.exists(args.qqp_improve_dir):
                os.makedirs(args.qqp_improve_dir)

            with open(
                os.path.join(
                    args.qqp_improve_dir,
                    f"qqp_params_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
                ),
                "w",
            ) as f:
                json.dump(config, f)

            with open(args.quora_dev_out, "w+") as f:
                f.write("id,Predicted_Is_Paraphrase\n")
                for p, s in zip(dev_quora_sent_ids, dev_quora_y_pred):
                    f.write(f"{p}\t{s}\n")

            with open(args.quora_test_out, "w+") as f:
                f.write("id,Predicted_Is_Paraphrase\n")
                for p, s in zip(test_quora_sent_ids, test_quora_y_pred):
                    f.write(f"{p}\t{s}\n")

        else:
            print("your score is lower than the baseline for qqp task")

    if task == "sts":
        print(f"dev sts corr :: {dev_sts_corr :.3f}")

        if dev_sts_corr > 0.414:
            print("Your score is higher than the baseline for sts task")
            print("saving params")
            if not os.path.exists(args.sts_improve_dir):
                os.makedirs(args.sts_improve_dir)

            with open(
                os.path.join(
                    args.sts_improve_dir,
                    f"sts_params_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
                ),
                "w",
            ) as f:
                json.dump(config, f)

            with open(args.sts_dev_out, "w+") as f:
                f.write("id,Predicted_Similarity\n")
                for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                    f.write(f"{p}\t{s}\n")

            with open(args.sts_test_out, "w+") as f:
                f.write("id,Predicted_Similarity\n")
                for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                    f.write(f"{p}\t{s}\n")
        else:
            print("your score is lower than the baseline for sts task")

    if task == "etpc":
        with open(args.etpc_dev_out, "w+") as f:
            print(f"dev etpc acc :: {dev_etpc_accuracy :.3f}")
            f.write("id,Predicted_Paraphrase_Types\n")
            for p, s in zip(dev_etpc_sent_ids, dev_etpc_y_pred):
                f.write(f"{p}\t{s}\n")

        with open(args.etpc_test_out, "w+") as f:
            f.write("id,Predicted_Paraphrase_Types\n")
            for p, s in zip(test_etpc_sent_ids, test_etpc_y_pred):
                f.write(f"{p}\t{s}\n")

    if task == "multitask":
        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sst_accuracy :.3f}")
            f.write("id,Predicted_Sentiment\n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p}\t{s}\n")

        with open(args.sst_test_out, "w+") as f:
            f.write("id,Predicted_Sentiment\n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p}\t{s}\n")

        with open(args.quora_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_quora_accuracy :.3f}")
            f.write("id,Predicted_Is_Paraphrase\n")
            for p, s in zip(dev_quora_sent_ids, dev_quora_y_pred):
                f.write(f"{p}\t{s}\n")

        with open(args.quora_test_out, "w+") as f:
            f.write("id,Predicted_Is_Paraphrase\n")
            for p, s in zip(test_quora_sent_ids, test_quora_y_pred):
                f.write(f"{p}\t{s}\n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write("id,Predicted_Similarity\n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p}\t{s}\n")

        with open(args.sts_test_out, "w+") as f:
            f.write("id,Predicted_Similarity\n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p}\t{s}\n")
