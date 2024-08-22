# Creates a zip file for submission.

import os
import zipfile

# Collect predictions
predictions_dir = "./logs/predictions/"
required_files = []

for files in os.listdir(predictions_dir):
    required_files += [os.path.join(predictions_dir, files)]

print(f"required_files: {required_files}")
print(f"len(required_files): {len(required_files)}")


def main():
    aid = "multitask-vaccine-smart-regularization"
    with zipfile.ZipFile(f"{aid}.zip", "w") as zz:
        for file in required_files:
            zz.write(file, os.path.join("predictions-zip", file))
    print(f"Submission zip file created: {aid}.zip")


if __name__ == "__main__":
    main()
