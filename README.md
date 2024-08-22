# DNLP SS24 Final Project – BERT for Multitask Learning and BART for Paraphrasing

- **Group name:** BERT's Buddies

- **Group code:** G04

- **Group repository:** [FrederikHennecke/DeepLearning4NLP](https://github.com/FrederikHennecke/DeepLearning4NLP)

- **Tutor responsible:** Yasir

- **Group team leader:** Hennecke, Frederik

- **Group member 1:** Hennecke, Frederik

- **Group member 2:** Aly, Mohamed [Email](mohamed.aly@stud.uni-goettingen.de), [Github](https://github.com/maly-phy)

- **Group member 3:** Tillmann, Arne

## Introduction

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Final](https://img.shields.io/badge/Status-Final-purple.svg)](https://https://img.shields.io/badge/Status-Final-blue.svg)
[![Black Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://black.readthedocs.io/en/stable/)
[![AI-Usage Card](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](./AI-Usage-Card.pdf/)

This repository our official implementation of the Multitask BERT project for the Deep Learning for Natural Language
Processing course at the University of Göttingen.

A pretrained
BERT ([BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805))
model was used as the basis for our experiments. The model was fine-tuned on the three tasks using a multitask learning
approach. The model was trained on the three tasks simultaneously, with a single shared BERT encoder and three separate
task-specific classifiers.

## Requirements

To install requirements and all dependencies using conda, run:

```sh
bash setup_gwdg.sh
```

This will download and install miniconda on your machine, create the project's conda environment and activate it. It will also download the models used through the project and the POS and NER tags from spacy.

## Training

To train the multitask BERT, you need to activate the environment and run

```sh
python -u train_multitask_pal.py --option finetune --use_gpu --local_files_only --smoketest --no_tensorboard --no_train_classifier --use_smart_regularization --use_pal --use_amp
```

Alternatively, you can run the `run_train.sh` script which also has the option to run on the GWDG cluster and submit jobs for training. You can also choose different scripts to run from that file and set the parameters.

There are lots of parameters to set. To see all of them, run `python <filename> --help`. Theses are most important ones in the `train_multitask.py` and `train_multitask_pal.py` file. Please note that parameters from branches other than the main are included as well:

| **Parameter**                           | **Description**                                                                                                                                            |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--additional_input`                    | Activates the usage for POS and NER tags for the input of BERT                                                                                             |
| `--batch_size`                          | Batch size.                                                                                                                                                |
| `--clip`                                | Gradient clipping value.                                                                                                                                   |
| `--epochs`                              | Number of epochs.                                                                                                                                          |
| `--hidden_dropout_prob`                 | Dropout probability for hidden layers.                                                                                                                     |
| `--hpo_trials`                          | Number of trials for hyperparameter optimization.                                                                                                          |
| `--hpo`                                 | Activate hyperparameter optimization.                                                                                                                      |
| `--lr`                                  | Learning rate.                                                                                                                                             |
| `--optimizer`                           | Optimizer to use. Options are `AdamW` and `SophiaH`.                                                                                                       |
| `--option`                              | Determines if BERT parameters are frozen (`pretrain`) or updated (`finetune`).                                                                             |
| `--samples_per_epoch`                   | Number of samples per epoch.                                                                                                                               |
| `--scheduler`                           | Learning rate scheduler to use. Options are `plateau`, `cosine`, `linear_warmup` or `None` for non schedulers                                              |
| `--unfreeze_interval`                   | Number of epochs until the next BERT layer is unfrozen                                                                                                     |
| `--use_gpu`                             | Whether to use the GPU.                                                                                                                                    |
| `--weight_decay`                        | Weight decay for optimizer.                                                                                                                                |
| `--smoketest`                           | To test the code implementation and debug it before the actual run                                                                                         |
| `--pooling`                             | Add a pooling layer before the classifier                                                                                                                  |
| `--layers`                              | Select more layers from the model to train                                                                                                                 |
| `--add_layers`                          | Add more linear layers to train before the classifier for the SST task                                                                                     |
| `--combined_models`                     | Use combined BERT models to train                                                                                                                          |
| `--train_mode`                          | Specifies if train the last hidden state, the pooler output or certain layers of the model. Options are `all_pooled`, `last_hidden_state`, `single_layers` |
| `--max_length`                          | Maximum length of tekens to chunk                                                                                                                          |
| `--n_hidden_layers`                     | Number of hidden layers to use                                                                                                                             |
| `--task_scheduler`                      | Choose how to schedule the task during training. Options are `random`, `round_robin`, `pal`, `para`, `sts`, `sst`                                          |
| `--projection`                          | How to handle competing gradients from different tasks. Options are `pcgrad`, `vaccine`, `none`                                                            |
| `--combine_strategy`                    | Needed in case of using projection. Options are `encourage`, `force`, `none`                                                                               |
| `--use_pal`                             | Add projected attention layers on top of BERT                                                                                                              |
| `--patience`                            | Number of epochs to wait without improvement before the training stops                                                                                     |
| `--use_smart_regularization`            | Implemented pytorch package to regularize the weights and reduce overfitting                                                                               |
| `--write_summary` or `--no_tensorboard` | Whether to save training logs for later view on tensorboard                                                                                                |
| `--model_name`                          | Choose the model to pretrain or fine-tune                                                                                                                  |

## Evaluation

The model is evaluated after each epoch on the validation set. The results are printed to the console and saved in the `--logdir`in case of setting `--no_tensorboard` to `store-false`. The model checkpoints are saved after each epoch in the `filepath` directory. After finishing the training, the model is loaded to be testedon the test set. The model's predictions are then saved in the `predictions_path`.

## Results

In the first phase of the project, we aimed at getting the baseline score for each task, where we trained the model for each task separately with the following parameters:

- option: `finetune`
- epochs: `10`
- learning rate: `1e-5`
- hidden dropout prob: `0.1`
- batch size: `64`

In the second phase, we used multitask training to let the model train on the data of all tasks, but with a different classifier for each task. This helped the model learn and the performance to imrpove since the tasks are dependent. In addition we performed single task training for the ones that didn't improve through the multitasking and tried to apply other approaches to improve their score.

For the multitask training, we used the following parameters:

- option: `finetune`
- epochs: `20`
- learning rate: `8e-5`
- hidden dropout prob: `0.3`
- batch size: `64`
- optimizer: `àdamw`
- clip: `0.25`
- weight decay: `0.05`
- samples per epoch: `20000`

This allowed us to set a standard training framework and try with other options for the sake of further improvement. For furher hyperparameter choices, see the default values in the [training script](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/main/train_multitask.py)or in the corresponding [slurm file](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/main/slurm_files/slurm-train-alldata-multitask-STS-QQP-improve.out).

---

We achieved the following results

### [Paraphrase Identification on Quora Question Pairs](https://paperswithcode.com/sota/paraphrase-identification-on-quora-question)

Paraphrase Detection is the task of finding paraphrases of texts in a large corpus of passages.
Paraphrases are “rewordings of something written or spoken by someone else”; paraphrase
detection thus essentially seeks to determine whether particular words or phrases convey
the same semantic meaning. This task measures how well systems can understand fine-grained notions of semantic meaning.

| Model name                    | Description                                           | Accuracy | link to slurm                                                                                                                                                      |
| ----------------------------- | ----------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Pcgrad projection             | Projected gradient descent with round robin scheduler | 86.4%    | [pcgrad](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train_pcgrad_classifier.out)                                    |
| Vaccine projection            | Surgery of the gradient with round robin scheduler    | 85.9%    | [vaccine](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train-vaccine_classifier.out)                                  |
| Combined BERT models          | 3 BERT models combined with a gating network          | 85.7%    | [combined BERT](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-combined-models-traini-multitask_classifier-1332732.out) |
| Augmented Attention Multitask | Attention layer on top of BERT                        | 84.4%    | [attention multitask](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train-alldata-multitask-STS-QQP-improve.out)       |
| BiLSTM-Multitask              | BiLSTM layer on top of BERT                           | 83.9%    | [bilstm multitask](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-bilstm-train-multitask-classifier-1358610.out)        |
| Bert-large                    | use bert-large model for multitasking                 | 82.4%    | [bert-large](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-bert-large-%20train-multitask_classifier.out)               |
| Max pooling                   | Max of the last hidden states' sequence               | 81.9%    | [max pooling](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train-QQP-MAX_Pooling-1296205.out)                         |
| Baseline (single task)        | Single task training                                  | 80.6%    | [baseline](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/main/slurm_files/slurm-train-multitask_classifier-711664.out)                                 |

### [Sentiment Classification on Stanford Sentiment Treebank (SST)](https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained)

A basic task in understanding a given text is classifying its polarity (i.e., whether the expressed
opinion in a text is positive, negative, or neutral). Sentiment analysis can be utilized to
determine individual feelings towards particular products, politicians, or within news reports.
Each phrase has a label of negative, somewhat negative,
neutral, somewhat positive, or positive.

| Model name                    | Description                                           | Accuracy | link to slurm                                                                                                                                                      |
| ----------------------------- | ----------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| BiLSTM-Multitask              | BiLSTM layer on top of BERT                           | 53.0%    | [bilstm multitask](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-bilstm-train-sst-classifier-1360936.out)              |
| Max pooling                   | Max of the last hidden states' sequence               | 52.8%    | [max pooling](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-max-pool-sst-single.out)                                   |
| Baseline (single task)        | Single task training                                  | 52.2%    | [baseline](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/main/slurm_files/slurm-train-multitask_classifier-711664.out)                                 |
| Bert-large                    | use bert-large model for multitasking                 | 51.2%    | [bert-large](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-bert-large-%20train-multitask_classifier.out)               |
| Vaccine projection            | Surgery of the gradient with round robin scheduler    | 50.4%    | [vaccine](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train-vaccine_classifier.out)                                  |
| Augmented Attention Multitask | Attention layer on top of BERT                        | 50.2%    | [attention multitask](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train-alldata-multitask-STS-QQP-improve.out)       |
| Combined BERT models          | 3 BERT models combined with a gating network          | 49.1%    | [combined BERT](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-combined-models-traini-multitask_classifier-1332732.out) |
| Pcgrad projection             | Projected gradient descent with round robin scheduler | 47.9%    | [pcgrad](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train_pcgrad_classifier.out)                                    |

### [Semantic Textual Similarity on STS](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts-benchmark)

The semantic textual similarity (STS) task seeks to capture the notion that some texts are
more similar than others; STS seeks to measure the degree of semantic equivalence [Agirre
et al., 2013]. STS differs from paraphrasing in it is not a yes or no decision; rather it
allows degrees of similarity from 5 (same meaning) to 0 (not at all related).

| Model name                    | Description                                           | Accuracy | link to slurm                                                                                                                                                      |
| ----------------------------- | ----------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Pcgrad projection             | Projected gradient descent with round robin scheduler | 86.4%    | [pcgrad](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train_pcgrad_classifier.out)                                    |
| Vaccine projection            | Surgery of the gradient with round robin scheduler    | 85.9%    | [vaccine](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train-vaccine_classifier.out)                                  |
| Combined BERT models          | 3 BERT models combined with a gating network          | 85.7%    | [combined BERT](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-combined-models-traini-multitask_classifier-1332732.out) |
| Augmented Attention Multitask | Attention layer on top of BERT                        | 84.4%    | [attention multitask](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train-alldata-multitask-STS-QQP-improve.out)       |
| BiLSTM-Multitask              | BiLSTM layer on top of BERT                           | 83.9%    | [bilstm multitask](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-bilstm-train-multitask-classifier-1358610.out)        |
| Bert-large                    | use bert-large model for multitasking                 | 82.4%    | [bert-large](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-bert-large-%20train-multitask_classifier.out)               |
| Max pooling                   | Max of the last hidden states' sequence               | 81.9%    | [max pooling](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train-QQP-MAX_Pooling-1296205.out)                         |
| Baseline (single task)        | Single task training                                  | 41.4%    | [baseline](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/main/slurm_files/slurm-train-multitask_classifier-711664.out)                                 |

Explain how we can run your code in this section. We should be able to reproduce the results you've obtained.

In addition, if you used libraries that were not included in the conda environment 'dnlp' explain the exact installation instructions or provide a `.sh` file for the installation.

Which files do we have to execute to train/evaluate your models? Write down the command which you used to execute the experiments. We should be able to reproduce the experiments/results.

_Hint_: At the end of the project you can set up a new environment and follow your setup instructions making sure they are sufficient and if you can reproduce your results.

Run `bash setup_gwdg.sh` and `sbatch gwdg_run.sh`

# Methodology

In this section explain what and how you did your project.

If you are unsure how this is done, check any research paper. They all describe their methods/processes. Describe briefly the ideas that you implemented to improve the model. Make sure to indicate how are you using existing ideas and extending them. We should be able to understand your project's contribution.

**BART paraphrase detection:**
Arne implemented a new lossfunction based on the new metric (mcc) and the old loss BCEWithLogitLoss. The penalizing weights were determined by a gridsearch.

# Experiments

Keep track of your experiments here. What are the experiments? Which tasks and models are you considering?

Write down all the main experiments and results you did, even if they didn't yield an improved performance. Bad results are also results. The main findings/trends should be discussed properly. Why a specific model was better/worse than the other?

You are **required** to implement one baseline and improvement per task. Of course, you can include more experiments/improvements and discuss them.

You are free to include other metrics in your evaluation to have a more complete discussion.

Be creative and ambitious.

For each experiment answer briefly the questions:

- What experiments are you executing? Don't forget to tell how you are evaluating things.
- What were your expectations for this experiment?
- What have you changed compared to the base model (or to previous experiments, if you run experiments on top of each other)?
- What were the results?
- Add relevant metrics and plots that describe the outcome of the experiment well.
- Discuss the results. Why did improvement _A_ perform better/worse compared to other improvements? Did the outcome match your expectations? Can you recognize any trends or patterns?

**BART paraphrase detection:**
SophiaG
learning rate scheduler
costumloss

- What experiments are you executing? Don't forget to tell how you are evaluating things.
- What were your expectations for this experiment?
- What have you changed compared to the base model (or to previous experiments, if you run experiments on top of each other)?
- What were the results?
- Add relevant metrics and plots that describe the outcome of the experiment well.
- Discuss the results. Why did improvement _A_ perform better/worse compared to other improvements? Did the outcome match your expectations? Can you recognize any trends or patterns?

## Results

Summarize all the results of your experiments in tables:

| **Stanford Sentiment Treebank (SST)** | **Metric 1** | **Metric n** |
| ------------------------------------- | ------------ | ------------ |
| Baseline                              | 45.23%       | ...          |
| Improvement 1                         | 58.56%       | ...          |
| Improvement 2                         | 52.11%       | ...          |
| ...                                   | ...          | ...          |

| **Quora Question Pairs (QQP)** | **Metric 1** | **Metric n** |
| ------------------------------ | ------------ | ------------ |
| Baseline                       | 45.23%       | ...          |
| Improvement 1                  | 58.56%       | ...          |
| Improvement 2                  | 52.11%       | ...          |
| ...                            | ...          | ...          |

| **Semantic Textual Similarity (STS)** | **Metric 1** | **Metric n** |
| ------------------------------------- | ------------ | ------------ |
| Baseline                              | 45.23%       | ...          |
| Improvement 1                         | 58.56%       | ...          |
| Improvement 2                         | 52.11%       | ...          |
| ...                                   | ...          | ...          |

| **Paraphrase Type Detection (PTD)** | **Metric 1** | **Metric n** |
| ----------------------------------- | ------------ | ------------ |
| Baseline                            | 45.23%       | ...          |
| Improvement 1                       | 58.56%       | ...          |
| Improvement 2                       | 52.11%       | ...          |
| ...                                 | ...          | ...          |

| **Paraphrase Type Generation (PTG)** | **Metric 1** | **Metric n** |
| ------------------------------------ | ------------ | ------------ |
| Baseline                             | 45.23%       | ...          |
| Improvement 1                        | 58.56%       | ...          |
| Improvement 2                        | 52.11%       | ...          |
| ...                                  | ...          | ...          |

Discuss your results, observations, correlations, etc.

Results should have three-digit precision.

### Hyperparameter Optimization

Describe briefly how you found your optimal hyperparameter. If you focussed strongly on Hyperparameter Optimization, you can also include it in the Experiment section.

_Note: Random parameter optimization with no motivation/discussion is not interesting and will be graded accordingly_

## Visualizations

Add relevant graphs of your experiments here. Those graphs should show relevant metrics (accuracy, validation loss, etc.) during the training. Compare the different training processes of your improvements in those graphs.

For example, you could analyze different questions with those plots like:

- Does improvement A converge faster during training than improvement B?
- Does Improvement B converge slower but perform better in the end?
- etc...

## Members Contribution

Explain what member did what in the project:

**Member 1:** _implemented the training objective using X, Y, and Z. Supported member 2 in refactoring the code. Data cleaning, etc._

- **Group member 1:** Hennecke, Frederik

- **Group member 2:** Aly, Mohamed

- **Group member 3:** Tillmann, Arne implemented the costum loss function for the paraphrase type detection task. He also implemented the grid search to optimize the hyperparameters (weights).

- **Group member 4:** Chang, Yue

- **Group member 5:** Yaghoubi, Forough
  ...

We should be able to understand each member's contribution within 5 minutes.

# AI-Usage Card

Artificial Intelligence (AI) aided the development of this project. Please add a link to your AI-Usage card [here](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/main/ai-usage-card.pdf) [here the website to create a new one](https://ai-cards.org/).

# References

Write down all your references (other repositories, papers, etc.) that you used for your project.

@article{liu2023sophia,
title={Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training},
author={Liu, Hong and Li, Zhiyuan and Hall, David and Liang, Percy and Ma, Tengyu},
journal={arXiv preprint arXiv:2305.14342},
year={2023}
}

# DNLP SS24 Final Project

This is the starting code for the default final project for the Deep Learning for Natural Language Processing course at the University of Göttingen. You can find the handout [here](https://docs.google.com/document/d/1pZiPDbcUVhU9ODeMUI_lXZKQWSsxr7GO/edit?usp=sharing&ouid=112211987267179322743&rtpof=true&sd=true)

In this project, you will implement some important components of the BERT model to better understanding its architecture.
You will then use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection and semantic similarity.

After finishing the BERT implementation, you will have a simple model that simultaneously performs the three tasks.
You will then implement extensions to improve on top of this baseline.

## Setup instructions

- Follow `setup.sh` to properly setup a conda environment and install dependencies.
- There is a detailed description of the code structure in [STRUCTURE.md](./STRUCTURE.md), including a description of which parts you will need to implement.
- You are only allowed to use libraries that are installed by `setup.sh` (Use `setup_gwdg.sh` if you are using the GWDG clusters).
- Libraries that give you other pre-trained models or embeddings are not allowed (e.g., `transformers`).
- Use this template to create your README file of your repository: <https://github.com/gipplab/dnlp_readme_template>

## Project Description

Please refer to the project description for a through explanation of the project and its parts.

### Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

For the 2024 edition of the DNLP course at the University of Göttingen, the project was modified by [Niklas Bauer](https://github.com/ItsNiklas/), [Jonas Lührs](https://github.com/JonasLuehrs), ...
