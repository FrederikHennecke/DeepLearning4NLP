import argparse
import random

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

from optimizer import AdamW
from bart_detection import get_args, seed_everything

TQDM_DISABLE = False


def transform_data(dataset, shuffle, max_length=256):
    """
    Turn the data to the format you want to use.
    Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
    Tokenize the sentence pair in the following format:
    sentence_1 + SEP + sentence_1 segment location + SEP + paraphrase types.
    Return Data Loader.
    """
    ### TODO
    # raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained("models/bart-large")
    # sentences1= dataset["sentence1"].tolist()
    # sentences1_segment_location = dataset["sentence1_segment_location"].tolist()
    # paraphrase_types = dataset["paraphrase_types"].tolist()

    combined_sentences = []
    for i in range(len(dataset)):
        # paraphrase_types = dataset["paraphrase_types"].tolist()
        # paraphrase_types = ", ".join(map(str, paraphrase_types))
        combined_sentence = (
            dataset.loc[i, "sentence1"]
            + " <location> "
            + dataset.loc[i, "sentence1_segment_location"]
            + " <type> "
            + dataset.loc[i, "paraphrase_types"]
        )

        combined_sentences.append(combined_sentence)

    has_target = "sentence2" in dataset.columns

    if has_target:
        target = dataset["sentence2"].tolist()
    else:
        target = None

    # print(f"combined_sentences: {combined_sentences[:5]}")

    input_encodings = tokenizer(
        combined_sentences,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = torch.tensor(input_encodings["input_ids"])
    attention_mask = torch.tensor(input_encodings["attention_mask"])
    # print(f"input_ids: {input_ids.shape}")
    # print(f"attention_mask: {attention_mask.shape}")

    if target:
        target_encodings = tokenizer(
            target,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_ids = torch.tensor(target_encodings["input_ids"])
        dataset = TensorDataset(input_ids, attention_mask, target_ids)

    else:
        dataset = TensorDataset(input_ids, attention_mask)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    return dataloader


def train_model(model, train_data, dev_data, device, tokenizer):
    """
    Train the model. Return and save the model.
    """
    ### TODO
    # raise NotImplementedError

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        total_examples = 0
        correct_preds = 0

        for batch in tqdm(train_data, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE):
            input_ids, attention_mask, target_ids = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=target_ids
            )
            loss = logits.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches

        model.eval()  # switch to eval model, will turn off randomness like dropout
        dev_loss = 0
        num_batches = 0
        with torch.no_grad():
            for batch in dev_data:
                input_ids, attention_mask, target_ids = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                target_ids = target_ids.to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=target_ids,
                )
                dev_loss += outputs.loss.item()
                num_batches += 1

        avg_dev_loss = dev_loss / num_batches
        print(
            f"Epoch {epoch+1:02} | Average Train Loss: {avg_train_loss:.4f} | Average Dev Loss: {avg_dev_loss:.4f}"
        )
        model.train()

    model.save_pretrained("models/bart_generation")
    tokenizer.save_pretrained("models/bart_generation")

    return model


def test_model(test_data, test_ids, device, model, tokenizer):
    """
    Test the model. Generate paraphrases for the given sentences (sentence1) and return the results
    in form of a Pandas dataframe with the columns 'id' and 'Generated_sentence2'.
    The data format in the columns should be the same as in the train dataset.
    Return this dataframe.
    """
    ### TODO
    # raise NotImplementedError

    model.to(device)
    model.eval()

    generated_sentences = []
    with torch.no_grad():
        for batch in tqdm(test_data, desc="test", disable=TQDM_DISABLE):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,  # WARN: tune max_length parameter
                num_beams=5,  # WARN: tune num_beams parameter
                early_stopping=True,
            )

            pred_text = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in outputs
            ]
            generated_sentences.extend(pred_text)

    df_gen_results = pd.DataFrame(
        {"id": test_ids, "Generated_sentence2": generated_sentences}
    )

    return df_gen_results


def evaluate_model(model, dev_data, device, tokenizer):
    """
    You can use your train/validation set to evaluate models performance with the BLEU score.
    """
    model.eval()
    bleu = BLEU()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in dev_data:
            input_ids, attention_mask, target_ids = (
                batch  # WARN! paraphrase type is inseted in the encoder in the transform_data function above, but how to extract them as target_ids to use here??
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)

            # Generate paraphrases
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )

            pred_text = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in outputs
            ]
            ref_text = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in target_ids
            ]

            predictions.extend(pred_text)
            references.extend([[r] for r in ref_text])

    model.train()

    # Calculate BLEU score
    bleu_score = bleu.corpus_score(predictions, references)
    return bleu_score.score


def finetune_paraphrase_generation(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = BartForConditionalGeneration.from_pretrained("models/bart-large")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("models/bart-large")

    train_dataset = pd.read_csv(
        args.etpc_train_filename,
        sep="\t",
        usecols=[
            "sentence1",
            "sentence1_segment_location",
            "paraphrase_types",
            "sentence2",
            "sentence2_segment_location",
        ],
    )
    print(f"train_dataset shape: {train_dataset.shape}")
    print(f"train_dataset: {train_dataset.head()}\n")

    dev_dataset = pd.read_csv(
        args.etpc_dev_filename,
        sep="\t",
        usecols=[
            "sentence1",
            "sentence1_segment_location",
            "paraphrase_types",
            "sentence2",
            "sentence2_segment_location",
        ],
    )
    print(f"dev_dataset shape: {dev_dataset.shape}")
    print(f"dev_dataset: {dev_dataset.head()}\n")

    test_dataset = pd.read_csv(
        args.etpc_test_filename,
        sep="\t",
        usecols=["id", "sentence1", "sentence1_segment_location", "paraphrase_types"],
    )
    print(f"test_dataset shape: {test_dataset.shape}")
    print(f"test_dataset: {test_dataset.head()}")

    # You might do a split of the train data into train/validation set here
    # ...

    train_data = transform_data(train_dataset, shuffle=True)
    dev_data = transform_data(dev_dataset, shuffle=False)
    test_data = transform_data(test_dataset, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model, train_data, dev_data, device, tokenizer)

    print("Training finished.")

    bleu_score = evaluate_model(model, dev_data, device, tokenizer)
    print(f"The BLEU-score of the model is: {bleu_score:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(test_data, test_ids, device, model, tokenizer)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-generation-test-output.csv",
        index=False,
        sep="\t",
    )


if __name__ == "__main__":
    args = get_args()
    args.etpc_test_filename = "data/etpc-paraphrase-generation-test-student.csv"
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
