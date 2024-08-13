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
from bart_detection import get_args

TQDM_DISABLE = False


def transform_data(dataset, shuffle, max_length=256, target_encoding=True):
    """
    Turn the data to the format you want to use.
    Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
    Tokenize the sentence pair in the following format:
    sentence_1 + SEP + sentence_2 segment location + SEP + paraphrase types.
    Return DataLoader.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large", local_files_only=True
    )
    inputs = []
    targets = []

    for idx, row in dataset.iterrows():
        if target_encoding:
            input_text = row['sentence1'] + " SEP " + row['sentence1_segment_location'] + " SEP " + row['paraphrase_types']
            target_text = row['sentence2']
            inputs.append(input_text)
            targets.append(target_text)
        else:
            input_text = row['sentence1'] + " SEP " + row['sentence1_segment_location'] + " SEP " + row['paraphrase_types']
            inputs.append(input_text)

    encodings = tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    if target_encoding:
        target_encodings = tokenizer(targets, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        dataset = TensorDataset(encodings.input_ids, encodings.attention_mask, target_encodings.input_ids)
    else:
        dataset = TensorDataset(encodings.input_ids, encodings.attention_mask)

    dataloader = DataLoader(dataset, shuffle=shuffle)

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

    model.save_pretrained("models1/bart_generation_model")
    tokenizer.save_pretrained("models1/bart_generation_tokenizer")

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
    test_data is a Pandas Dataframe, the column "sentence1" contains all input sentence and
    the column "sentence2" contains all target sentences
    """
    model.eval()
    bleu = BLEU()
    predictions = []

    dataloader = transform_data(test_data, shuffle=False)
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Generate paraphrases
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )

            pred_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]

            predictions.extend(pred_text)

    inputs = test_data["sentence1"].tolist()
    references = test_data["sentence2"].tolist()

    model.train()
    # Calculate BLEU score
    bleu_score_reference = bleu.corpus_score(references, [predictions]).score
    # Penalize BLEU score if its to close to the input
    bleu_score_inputs = 100 - bleu.corpus_score(inputs, [predictions]).score

    print(f"BLEU Score: {bleu_score_reference}", f"Negative BLEU Score with input: {bleu_score_inputs}")

    # Penalize BLEU and rescale it to 0-100
    # If you perfectly predict all the targets, you should get an penalized BLEU score of around 52
    penalized_bleu = bleu_score_reference * bleu_score_inputs / 52
    print(f"Penalized BLEU Score: {penalized_bleu}")

    return penalized_bleu


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def finetune_paraphrase_generation(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large", local_files_only=True
    )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large", local_files_only=True
    )

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

    test_dataset = pd.read_csv(
        args.etpc_test_filename,
        sep="\t",
        usecols=["id", "sentence1", "sentence1_segment_location", "paraphrase_types"],
    )

    # You might do a split of the train data into train/validation set here
    # in the Main function

    train_data = transform_data(train_dataset, shuffle=True, target_encoding=True)
    dev_data = transform_data(dev_dataset, shuffle=False, target_encoding=True)
    test_data = transform_data(test_dataset, shuffle=False, target_encoding=False)

    model = train_model(model, train_data, dev_data, device, tokenizer)

    print("Training finished.")

    bleu_score = evaluate_model(model, dev_dataset, device, tokenizer)
    print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")

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
    args.etpc_train_filename = "data/etpc-paraphrase-train.csv"
    args.etpc_dev_filename = "data/etpc-paraphrase-dev.csv"
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
