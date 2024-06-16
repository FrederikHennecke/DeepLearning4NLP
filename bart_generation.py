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


def transform_data(dataset, batch_size, shuffle, max_length=256):
    """
    Turn the data to the format you want to use.
    Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
    Tokenize the sentence pair in the following format:
    sentence_1 + SEP + sentence_1 segment location + SEP + paraphrase types.
    Return Data Loader.
    """
    ### TODO
    # raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    combined_sentences = []
    for i in range(len(dataset)):
        combined_sentence = (
            dataset.loc[i, "sentence1"]
            + " [SEP] "
            + dataset.loc[i, "sentence1_segment_location"]
            + " [SEP] "
            + dataset.loc[i, "paraphrase_types"]
        )
        combined_sentences.append(combined_sentence)

    encodings = tokenizer(
        combined_sentences,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = torch.tensor(encodings["input_ids"])
    attention_mask = torch.tensor(encodings["attention_mask"])

    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )  # WARN: change batch size to 32
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
            # print(f"batch: {batch}")
            # break
            b_ids, b_mask = batch
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = logits.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches

        model.eval()  # switch to eval model, will turn off randomness like dropout
        dev_loss = 0
        with torch.no_grad():
            for batch in dev_data:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                # labels = batch["input_ids"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                dev_loss += outputs.loss.item()

        avg_dev_loss = dev_loss / len(dev_data)
        print(
            f"Epoch {epoch+1:02} | Average Train Loss: {avg_train_loss:.4f} | Average Dev Loss: {avg_dev_loss:.4f}"
        )
        model.train()

    model.save_pretrained("models/bart_generation.pt")
    tokenizer.save_pretrained("models/bart_generation.pt")

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


def evaluate_model(model, test_data, device, tokenizer):
    """
    You can use your train/validation set to evaluate models performance with the BLEU score.
    """
    model.eval()
    bleu = BLEU()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, labels = (
                batch  # WARN! paraphrase type is inseted in the encoder in the transform_data function above, but how to extract them as labels to use here??
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

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
                for g in labels
            ]

            predictions.extend(pred_text)
            references.extend([[r] for r in ref_text])

    model.train()

    # Calculate BLEU score
    bleu_score = bleu.corpus_score(predictions, references)
    return bleu_score.score


def finetune_paraphrase_generation(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    train_dataset = pd.read_csv(
        args.etpc_train_filename,
        sep="\t",
        usecols=["sentence1", "sentence1_segment_location", "paraphrase_types"],
    )
    print(f"train_dataset shape: {train_dataset.shape}")
    print(f"train_dataset: {train_dataset.head()}\n")

    dev_dataset = pd.read_csv(
        args.etpc_dev_filename,
        sep="\t",
        usecols=["sentence1", "sentence1_segment_location", "paraphrase_types"],
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

    train_data = transform_data(train_dataset, args.batch_size, shuffle=True)
    dev_data = transform_data(dev_dataset, args.batch_size, shuffle=False)
    test_data = transform_data(test_dataset, args.batch_size, shuffle=True)

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

    # For code testing
    args.epoch = 1
    args.lr = 10
    args.batch_size = 256
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
