import csv


def read_csv_and_get_sentence1_values(file_path, string='sentence1'):
    sentence1_values = []

    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')

        for row in reader:
            sentence1_values.append(row[string])

    return sentence1_values


cnt = 0
# Example usage
file_path1 = 'data/etpc-paraphrase-generation-test-student.csv'
sentence1_values1 = read_csv_and_get_sentence1_values(file_path1)

file_path2 = 'predictions/bart/etpc-paraphrase-generation-test-output.csv'
sentence1_values2 = read_csv_and_get_sentence1_values(file_path2, string="Generated_sentence2")

for i in range(len(sentence1_values2)):
    if sentence1_values1[i] != sentence1_values2[i]:
        print(sentence1_values1[i], end="\n")
        print(sentence1_values2[i], end="\n--------------\n")
        cnt += 1

print(f"number of changed sentences: {cnt}")
