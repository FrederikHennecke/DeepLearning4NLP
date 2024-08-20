# load paraphrase data

import pandas as pd
import os


dataset = pd.read_csv(
    'data/etpc-paraphrase-train.csv',
    sep="\t",
    usecols=[
        "sentence1",
        "sentence1_segment_location",
        "paraphrase_types",
        "sentence2",
        "sentence2_segment_location",
    ],
)

has_labels = "paraphrase_types" in dataset.columns

if has_labels:
    labels = (
        dataset["paraphrase_types"]
        .apply(lambda x: list(map(int, x.strip("[]").split(", "))))
        # .tolist()
    )
    binary_labels = [
        [1 if i in label else 0 for i in range(1, 8)] for label in labels
    ]  # number of labels = 7
else:
    binary_labels = None
# print the unique paraphrase_types


# make the list binary_labels of lists into a dataframe with one column named "types"
binary_labels = pd.DataFrame(binary_labels)

# print the unique rows of the binary_labels

print(binary_labels.shape)

# Find unique rows and count occurrences
unique_rows_with_counts = binary_labels.groupby([0, 1, 2, 3, 4, 5, 6]).size().reset_index(name='count')

# Count unique rows
unique_count = unique_rows_with_counts.shape[0]

# Print the count and the unique rows with their counts
print(f"Number of unique rows: {unique_count}")
print("Unique rows with counts:")
print(unique_rows_with_counts)

# sum the columns of the binary_labels
sums = binary_labels.sum(axis=0)


# print the sum of the columns
print(sums)
# save the dataset
# unique_rows_with_counts.to_csv("data/inspect-etpc-paraphrase-train.csv", index=False)