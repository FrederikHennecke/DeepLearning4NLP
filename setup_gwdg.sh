#!/bin/bash

# Function to check if conda is installed
check_conda_installed() {
    if command -v conda &> /dev/null; then
        echo "Conda is already installed."
    else
        echo "Conda is not installed. Installing Miniconda..."
        install_miniconda
    fi
}

# Function to install Miniconda
install_miniconda() {
    echo "Downloading Miniconda installer..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest.sh
    echo "Running Miniconda installer..."
    bash Miniconda3-latest.sh -b -p $HOME/miniconda
    echo "Initializing Miniconda..."
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init
    source ~/.bashrc
}

# Function to check if conda environment exists
check_conda_env() {
    if conda env list | grep -q "dnlp"; then
        echo "Conda environment 'dnlp' already exists."
    else
        echo "Conda environment 'dnlp' does not exist. Creating environment..."
        # conda create -n dnlp python=3.10 -y
        wget https://github.com/FrederikHennecke/DeepLearning4NLP/blob/main/dnlp.yml
        conda env create -f dnlp.yml
    fi
}

# Main script execution
check_conda_installed
check_conda_env

set -e

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

echo "Activating conda environment 'dnlp'..."
module load anaconda3
source activate dnlp

echo $CONDA_DEFAULT_ENV

## Download spacy POS and NER tags
pip install "spacy[cuda-autodetect]>=3.5"
python -m spacy download en_core_web_sm

## Download model on login-node
python - <<EOF
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
EOF

python - <<EOF
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased')
EOF

python - <<EOF
from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
EOF


python - <<EOF
from transformers import AutoTokenizer, AutoModel, BartModel
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
model = BartModel.from_pretrained('facebook/bart-large')
EOF

python -c "from tokenizer import BertTokenizer; from bert import BertModel; BertTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"
python -c "from tokenizer import BertTokenizer; from bert import BertModel; BertTokenizer.from_pretrained('bert-large-uncased'); BertModel.from_pretrained('bert-large-uncased')"
# python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-large-uncased'); from transformers import BertModel; BertModel.from_pretrained('bert-large-uncased')"


python -c "from transformers import RobertaTokenizer; RobertaTokenizer.from_pretrained('roberta-base'); from transformers import RobertaModel; RobertaModel.from_pretrained('roberta-base')"
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('facebook/bart-large'); from transformers import BartModel; BartModel.from_pretrained('facebook/bart-large')"
