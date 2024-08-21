import argparse
import random
import numpy as np
import pandas as pd
import csv
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel
from multitask_classifier import split_csv

from sklearn.metrics import matthews_corrcoef
from optimizer import AdamW
from sophia import SophiaG
from datasets import preprocess_string
import costum_loss

TQDM_DISABLE = False


class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=7):
        super(BartWithClassifier, self).__init__()

        self.bart = BartModel.from_pretrained(
            "facebook/bart-large", local_files_only=True
        )
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        # Use the BartModel to obtain the last hidden state
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        # Add an additional fully connected layer to obtain the logits
        logits = self.classifier(cls_output)

        # Return the probabilities
        probabilities = self.sigmoid(logits)
        return probabilities


def transform_data(
    dataset,
    batch_size,
    shuffle,
    max_length=512,
):
    """
    dataset: pd.DataFrame

    Turn the data to the format you want to use.

    1. Extract the sentences from the dataset. We recommend using the already split
    sentences in the dataset.
    2. Use the AutoTokenizer from_pretrained to tokenize the sentences and obtain the
    input_ids and attention_mask.
    3. Currently, the labels are in the form of [2, 5, 6, 0, 0, 0, 0]. This means that
    the sentence pair is of type 2, 5, and 6. Turn this into a binary form, where the
    label becomes [0, 1, 0, 0, 1, 1, 0]. Be careful that the test-student.csv does not
    have the paraphrase_types column. You should return a DataLoader without the labels.
    4. Use the input_ids, attention_mask, and binary labels to create a TensorDataset.
    Return a DataLoader with the TensorDataset. You can choose a batch size of your
    choice.
    """
    # raise NotImplementedError
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large", local_files_only=True
    )
    sentences1 = dataset["sentence1"].tolist()
    sentences2 = dataset["sentence2"].tolist()
    has_labels = "paraphrase_types" in dataset.columns

    if has_labels:
        labels = (
            dataset["paraphrase_types"]
            .apply(lambda x: list(map(int, x.strip("[]").split(", "))))
            .tolist()
        )
        binary_labels = [
            [1 if i in label else 0 for i in range(1, 8)] for label in labels
        ]  # number of labels = 7
    else:
        binary_labels = None

    encodings = tokenizer(
        sentences1,
        sentences2,
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    input_ids = torch.tensor(encodings["input_ids"])
    attention_mask = torch.tensor(encodings["attention_mask"])

    if binary_labels:
        labels_tensors = torch.tensor(binary_labels)
        dataset = TensorDataset(input_ids, attention_mask, labels_tensors)
    else:
        dataset = TensorDataset(input_ids, attention_mask)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def train_model(model, train_data, dev_data, device):
    """
    Train the model. You can use any training loop you want. We recommend starting with
    AdamW as your optimizer. You can take a look at the SST training loop for reference.
    Think about your loss function and the number of epochs you want to train for.
    You can also use the evaluate_model function to evaluate the
    model on the dev set. Print the training loss, training accuracy, and dev accuracy at
    the end of each epoch.

    Return the trained model.
    """
    ### TODO
    # raise NotImplementedError

    model = model.to(device) 
    optimizer = AdamW(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor([1., args.type_2, 1., 1., 1., args.type_6, args.type_7], device=device)
    loss_fun = costum_loss.CustomLoss(pos_weight=pos_weight)
    

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        total_examples = 0
        correct_preds = 0

        for batch in tqdm(train_data, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE):
            # print(f"batch: {batch}")
            b_ids, b_mask, b_labels = batch

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = loss_fun(logits, b_labels.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            total_examples += b_labels.size(0) * b_labels.size(
                1
            )  # total number of examples per batch
            preds = logits.round()
            correct_preds += (preds == b_labels).sum().item()

        avg_train_loss = train_loss / num_batches
        train_accuracy = correct_preds / total_examples
        dev_accuracy, matthews_coefficient  = evaluate_model(model, dev_data, device)
        print(
            f"Epoch {epoch+1:02} | Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | Dev Accuracy: {dev_accuracy:.4f} | matthews_coefficient: {matthews_coefficient:.4f}" 
        )

    return model


def test_model(model, test_data, test_ids, device):
    """
    Test the model. Predict the paraphrase types for the given sentences and return the results in form of
    a Pandas dataframe with the columns 'id' and 'Predicted_Paraphrase_Types'.
    The 'Predicted_Paraphrase_Types' column should contain the binary array of your model predictions.
    Return this dataframe.
    """
    ### TODO

    # raise NotImplementedError
    model.to(device)
    model.eval()
    all_preds = []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(test_data, desc="test", disable=TQDM_DISABLE):
            b_ids, b_mask = batch
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            logits = model(b_ids, b_mask)
            preds = logits.round().cpu().numpy()
            all_preds.extend(preds)

            logits = logits.cpu().numpy()
            all_logits.extend(logits)

    pred_paraphrase_types = [[int(x) for x in pred] for pred in all_preds]

    logit_paraphrase_types = [[float(x) for x in logit] for logit in all_logits]
    
    df_test_results = pd.DataFrame(
        {"id": test_ids, "Predicted_Paraphrase_Types": pred_paraphrase_types, "logits": logit_paraphrase_types}
    )
    return df_test_results


def evaluate_model(model, test_data, device):
    """
    This function measures the accuracy of our model's prediction on a given train/validation set
    We measure how many of the seven paraphrase types the model has predicted correctly for each data point.
    So, if the models prediction is [1,1,0,0,1,1,0] and the true label is [0,0,0,0,1,1,0], this predicition
    has an accuracy of 5/7, i.e. 71.4% .
    """
    all_pred = []
    all_labels = []
    model.eval()  # switch to eval model, will turn off randomness like dropout

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            predicted_labels = (outputs > 0.5).int()

            all_pred.append(predicted_labels)
            all_labels.append(labels)

    all_predictions = torch.cat(all_pred, dim=0)
    all_true_labels = torch.cat(all_labels, dim=0)

    true_labels_np = all_true_labels.cpu().numpy()
    predicted_labels_np = all_predictions.cpu().numpy()

    # Compute the accuracy for each label
    accuracies = []
    matthews_coefficients = []
    for label_idx in range(true_labels_np.shape[1]):
        correct_predictions = np.sum(
            true_labels_np[:, label_idx] == predicted_labels_np[:, label_idx]
        )
        total_predictions = true_labels_np.shape[0]
        label_accuracy = correct_predictions / total_predictions
        accuracies.append(label_accuracy)

        #compute Matthwes Correlation Coefficient for each paraphrase type
        matth_coef = matthews_corrcoef(true_labels_np[:,label_idx], predicted_labels_np[:,label_idx])
        matthews_coefficients.append(matth_coef)

    # Calculate the average accuracy over all labels
    accuracy = np.mean(accuracies)
    matthews_coefficient = np.mean(matthews_coefficients)
    model.train()
    return accuracy, matthews_coefficient


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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    # parser.add_argument("--lr_steps", type=int, default=10)
    parser.add_argument(
        "--etpc_train", type=str, default="data/etpc-paraphrase-train.csv"
    ) 
    parser.add_argument("--etpc_dev", type=str, default="data/etpc-paraphrase-dev.csv")
    parser.add_argument(
        "--etpc_test",
        type=str,
        default="data/etpc-paraphrase-detection-test-student.csv",
    )
    parser.add_argument("--type_2", type=float, default=0.3)
    parser.add_argument("--type_6", type=float, default=0.3)
    parser.add_argument("--type_7", type=float, default=0.3)
    
    args = parser.parse_args()
    return args


def finetune_paraphrase_detection(args):
    model = BartWithClassifier()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model.to(device)

    train_dataset = pd.read_csv(
        args.etpc_train,
        sep="\t",
        usecols=["sentence1", "sentence2", "paraphrase_types"],
    )

    dev_dataset = pd.read_csv(
        args.etpc_dev,
        sep="\t",
        usecols=["sentence1", "sentence2", "paraphrase_types"],
    )

    test_dataset = pd.read_csv(
        args.etpc_test,
        sep="\t",
        usecols=["id", "sentence1", "sentence2"],
    )

    # TODO You might do a split of the train data into train/validation set here
    # (or in the csv files directly)

    # Already Done before!

    train_data = transform_data(train_dataset, args.batch_size, shuffle=True)
    dev_data = transform_data(dev_dataset, args.batch_size, shuffle=False)
    test_data = transform_data(test_dataset, args.batch_size, shuffle=False)

    model = train_model(model, train_data, dev_data, device)

    print("Training finished.")

    accuracy, matthews_corr = evaluate_model(model, train_data, device)
    print(f"The accuracy of the model is: {accuracy:.3f}")
    print(f"Matthews Correlation Coefficient of the model is: {matthews_corr:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(model, test_data, test_ids, device)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-detection-test-output.csv",
        index=False,
        sep="\t",
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)