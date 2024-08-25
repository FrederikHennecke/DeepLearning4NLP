import argparse
import random

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
from nltk.corpus import wordnet

from optimizer import AdamW

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

    SEP_token = "<SEP>"
    tokenizer.add_tokens([SEP_token])
    tokenizer.sep_token = SEP_token

    for idx, row in dataset.iterrows():
        if target_encoding:
            input_text = row['sentence1'] + SEP_token + row['sentence1_segment_location'] + SEP_token + row['paraphrase_types']
            target_text = row['sentence2']
            inputs.append(input_text)
            targets.append(target_text)
        else:
            input_text = row['sentence1'] + SEP_token + row['sentence1_segment_location'] + SEP_token + row['paraphrase_types']
            inputs.append(input_text)

    encodings = tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    if target_encoding:
        target_encodings = tokenizer(targets, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        dataset = TensorDataset(encodings.input_ids, encodings.attention_mask, target_encodings.input_ids)
    else:
        dataset = TensorDataset(encodings.input_ids, encodings.attention_mask)

    dataloader = DataLoader(dataset, shuffle=shuffle)

    return dataloader, tokenizer


def replace_with_synonyms(sentence, prob=0.3):
    """
    Replace words in a sentence with their synonyms with a given probability.

    :param sentence: The input sentence to process.
    :param prob: Probability of replacing a word with its synonym.
    :return: The sentence with some words replaced by their synonyms.
    """
    words = sentence.split()
    new_sentence = []

    for word in words:
        # Decide whether to replace this word
        if random.random() < prob:
            synonyms = wordnet.synsets(word)
            if synonyms:
                # Get the lemmas of the first synset
                lemmas = synonyms[0].lemmas()
                if lemmas:
                    # Choose a random synonym
                    synonym = random.choice(lemmas).name()
                    # Avoid replacing a word with itself
                    if synonym.lower() != word.lower():
                        new_sentence.append(synonym.replace('_', ' '))
                        continue
        new_sentence.append(word)

    return ' '.join(new_sentence)



def add_synonyms_to_dataframe(df, prob=0.3):
    """
    Replace words in a pandas DataFrame column with synonyms and append the new dataframe.

    :param df: The DataFrame containing the sentences.
    :param prob: Probability of replacing a word with its synonym.
    :return: dataframe appended with new sentences.
    """
    lst = []
    for _, row in df.iterrows():
        sentence = row.copy()
        sentence['sentence1'] = replace_with_synonyms(row['sentence1'], prob)
        lst.append(sentence)
    df_tmp = pd.DataFrame(lst)
    return pd.concat([df, df_tmp], ignore_index=True)



def add_noise_to_sentence(sentence, noise_level=0.2, tokenizer=None):
    """
    Adds noise to a sentence by randomly swapping, deleting, or replacing words.

    :param sentence: Input sentence.
    :param noise_level: Proportion of words to modify (default: 0.2).
    :param tokenizer: Tokenizer with a mask token for word replacement.
    :return: Noisy sentence with modified words.
    """
    words = sentence.split()
    num_words = len(words)
    num_noisy_words = int(noise_level * num_words)

    for _ in range(num_noisy_words):
        num_words = len(words)
        noise_type = random.randint(0, 2)
        idx = random.randint(0, num_words - 1)

        # swap words
        if noise_type == 0 and num_words > 1:
            idx2 = (idx + 1) % num_words
            words[idx], words[idx2] = words[idx2], words[idx]

        # delete words
        elif noise_type == 1:
            words.pop(idx)

        # replace words with token
        elif noise_type == 2:
            words[idx] = tokenizer.mask_token

    noisy_sentence = " ".join(words)
    return noisy_sentence


def custom_loss_function(logits, labels, input_ids, model, tokenizer,  similarity_weight=0.2, dissimilarity_weight=0.6, copy_penalty_weight=1.5):
    """
    Replace words in a pandas DataFrame column with synonyms and append the new dataframe.

    :param logits: Model output logits.
    :param labels: Ground truth token IDs.
    :param input_ids: Input token IDs.
    :param model: BART model instance.
    :param tokenizer: Tokenizer instance.
    :param similarity_weight: Weight for similarity loss (default: 0.2).
    :param dissimilarity_weight: eight for dissimilarity loss (default: 0.6).
    :param copy_penalty_weight: Weight for copy penalty loss (default: 1.5).
    :return: Combined loss value.
    """
    loss_fct = torch.nn.CrossEntropyLoss()
    cross_entropy_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    with torch.no_grad():
        input_embeddings = model.model.encoder(input_ids).last_hidden_state.mean(dim=1)
        output_ids = logits.argmax(dim=-1)
        output_embeddings = model.model.encoder(output_ids).last_hidden_state.mean(dim=1)
        
    # cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(input_embeddings, output_embeddings, dim=-1)
    similarity_loss = 1 - cosine_sim.mean()
    dissimilarity_loss = cosine_sim.mean()

    # penalty if sentences are simple copies
    input_ids_expanded = input_ids.unsqueeze(1).expand(-1, labels.size(1), -1)
    target_ids_expanded = labels.unsqueeze(2)
    match = (target_ids_expanded == input_ids_expanded).float()
    copy_penalty = match.mean(dim=2).sum(dim=1) / input_ids.size(1)
    copy_penalty = copy_penalty.mean()
    
    loss = cross_entropy_loss + similarity_weight * similarity_loss + dissimilarity_weight * dissimilarity_loss + copy_penalty_weight * copy_penalty

    return loss


def train_model(model, train_data, dev_data, device, tokenizer, args):
    """
    Train the model. Return and save the model.
    """
    ### TODO
    # raise NotImplementedError

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

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

            # Decode the input_ids to text, apply noise, then re-encode
            original_sentences = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            noisy_sentences = [add_noise_to_sentence(sentence, noise_level=args.noise, tokenizer=tokenizer) for sentence in original_sentences]
            noisy_encodings = tokenizer(noisy_sentences, padding=True, truncation=True, max_length=input_ids.size(1), return_tensors="pt")
            
            noisy_input_ids = noisy_encodings.input_ids.to(device)
            noisy_attention_mask = noisy_encodings.attention_mask.to(device)

            optimizer.zero_grad()
            logits = model(
                input_ids=noisy_input_ids, attention_mask=noisy_attention_mask, labels=target_ids
            )
            #loss = logits.loss
            loss = custom_loss_function(logits.logits , target_ids, input_ids, model=model, tokenizer=tokenizer, 
                                        similarity_weight=args.similarity_weight, dissimilarity_weight=args.dissimilarity_weight, copy_penalty_weight=args.copy_penalty_weight)
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

                # Calculate dev loss using the same custom loss function
                #loss = outputs.loss
                logits = outputs.logits
                loss = custom_loss_function(logits , target_ids, input_ids, model=model, tokenizer=tokenizer, 
                                        similarity_weight=args.similarity_weight, dissimilarity_weight=args.dissimilarity_weight, copy_penalty_weight=args.copy_penalty_weight)

                dev_loss += loss.item()
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

    dataloader, _ = transform_data(test_data, shuffle=False)
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Generate paraphrases
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,  # WARN: tune max_length parameter
                num_beams=5,  # WARN: tune num_beams parameter
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
    train_dataset = add_synonyms_to_dataframe(train_dataset, prob=args.synonym_prob)
    train_data, tokenizer = transform_data(train_dataset, shuffle=True, target_encoding=True)
    dev_data, _ = transform_data(dev_dataset, shuffle=False, target_encoding=True)
    test_data, _ = transform_data(test_dataset, shuffle=False, target_encoding=False)

    model.resize_token_embeddings(len(tokenizer))

    model = train_model(model, train_data, dev_data, device, tokenizer, args)

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--etpc_train", type=str, default="data/etpc-paraphrase-train.csv"
    )
    parser.add_argument("--etpc_dev", type=str, default="data/etpc-paraphrase-dev.csv")
    parser.add_argument(
        "--etpc_test",
        type=str,
        default="data/etpc-paraphrase-detection-test-student.csv",
    )
    parser.add_argument("--similarity_weight", type=float, default=0.)
    parser.add_argument("--dissimilarity_weight", type=float, default=0.)
    parser.add_argument("--copy_penalty_weight", type=float, default=0.)
    parser.add_argument("--noise", type=float, default=0.)
    parser.add_argument("--synonym_prob", type=float, default=0.)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.etpc_test_filename = "data/etpc-paraphrase-generation-test-student.csv"
    args.etpc_train_filename = "data/etpc-paraphrase-train.csv"
    args.etpc_dev_filename = "data/etpc-paraphrase-dev.csv"
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)

# python -u bart_generation.py --use_gpu --epochs 1 --lr 1e-5 --similarity_weight 0.2 --dissimilarity_weight 0.6 --copy_penalty_weight 1.5 --noise 0.2 --synonym_prob 0.3