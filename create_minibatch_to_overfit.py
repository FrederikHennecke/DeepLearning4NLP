import pandas as pd

# Load the etcp-paraphrase train dataset
dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")

# Sample a minibatch of 64 examples
minibatch = dataset.sample(n=64)

# Store the minibatch in a new CSV document
minibatch.to_csv("data/etpc-paraphrase-train-minibatch.csv", sep="\t", index=False)