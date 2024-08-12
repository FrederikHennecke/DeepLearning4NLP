# Creates a zip file for submission.

import os
import zipfile

# Collect predictions
predictions_dir = "predictions/bert/multitask/train_multitask_alldata_STS_QQP_improve"
required_files = []

for files in os.listdir(predictions_dir):
    required_files += [os.path.join(predictions_dir, files)]

print(f"required_files: {required_files}")
print(f"len(required_files): {len(required_files)}")


def main():
    aid = "alldata_multitask_STS_QQP_improvement"
    with zipfile.ZipFile(f"{aid}.zip", "w") as zz:
        for file in required_files:
            zz.write(file, os.path.join(".", file))
    print(f"Submission zip file created: {aid}.zip")


if __name__ == "__main__":
    main()
