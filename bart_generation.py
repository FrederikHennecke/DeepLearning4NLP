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


TQDM_DISABLE = False


def transform_data(dataset, max_length=256):
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
        dataset, batch_size=512, shuffle=True
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
            b_ids = batch["input_ids"].to(device)
            b_mask = batch["attention_mask"].to(device)
            b_labels = batch["input_ids"].to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask, b_labels)
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
                labels = batch["input_ids"].to(device)
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                dev_loss += outputs.loss.item()

        avg_dev_loss = dev_loss / len(dev_data)
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
            input_ids, attention_mask, labels = batch
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


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    return args


def finetune_paraphrase_generation(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    dev_dataset = pd.read_csv("data/etpc-paraphrase-dev.csv", sep="\t")
    test_dataset = pd.read_csv(
        "data/etpc-paraphrase-generation-test-student.csv", sep="\t"
    )

    # You might do a split of the train data into train/validation set here
    # ...

    train_data = transform_data(train_dataset)
    dev_data = transform_data(dev_dataset)
    test_data = transform_data(test_dataset)

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
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
