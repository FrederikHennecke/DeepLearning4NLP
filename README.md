# DNLP SS24 Final Project – BERT for Multitask Learning and BART for Paraphrasing

**Table of Contents**

- [DNLP SS24 Final Project – BERT for Multitask Learning and BART for Paraphrasing](#dnlp-ss24-final-project-bert-for-multitask-learning-and-bart-for-paraphrasing)

  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Data](#data)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Results](#results)
    - [Paraphrase Identification on Quora Question Pairs (QQP)](#paraphrase-identification-on-quora-question-pairs-qqp)
    - [Sentiment Classification on Stanford Sentiment Treebank (SST)](#sentiment-classification-on-stanford-sentiment-treebank-sst)
    - [Semantic Textual Similarity on STS](#semantic-textual-similarity-on-sts)
  - [Methodology](#methodology)
    - [POS and NER Tag Embeddings](#pos-and-ner-tag-embeddings)
      - [Experimental Results](#experimental-results)
      - [Explanation of Results](#explanation-of-results)
    - [Optimizers](#optimizers)
      - [Sophia](#sophia)
        - [Implementation](#implementation)
        - [Experimental Results](#experimental-results)
        - [Explanation of Results](#explanation-of-results)
    - [Gradient Optimization (Projectors)](#gradient-optimization-projectors)
      - [Pcgrad](#pcgrad)
      - [Experimental Results](#experimental-results)
      - [Explanation of Results](#explanation-of-results)
      - [GradVac](#gradvac)
      - [Experimental Results](#experimental-results)
      - [Explanation of Results](#explanation-of-results)
    - [Schedulers](#schedulers)
      - [Linear Warmup](#linear-warmup)
        - [Experimental Results](#experimental-results)
      - [Plateau](#plateau)
      - [Cosine](#cosine)
      - [Round Robin](#round-robin)
        - [Experimental Results](#experimental-results)
      - [PAL](#pal)
        - [Experimental Results](#experimental-results)
      - [Random](#random)
    - [Regularization](#regularization)
      - [SMART](#smart)
        - [Experimental Results](#experimental-results)
  - [Details](#details)
    - [Data Imbalance](#data-imbalance)
    - [Classifier Architecture](#classifier-architecture)
    - [Augmented Attention](#augmented-attention)
    - [BiLSTM](#bilstm)
    - [Feed more Sequences](#feed-more-sequences)
    - [Hierarchical BERT](#hierarchical-bert)
    - [CNN BERT](#cnn-bert)
    - [Combined Models](#combined-models)
    - [BERT-Large](#bert-large)
    - [PALs](#pals)
  - [Hyperparameter Search](#hyperparameter-search)
  - [Computation Resources](#computation-resources)
  - [Contributors](#contributors)
  - [Licence](#licence)
  - [Usage Guidelines](#usage-guidelines)
    - [Pre-commit Hooks](#pre-commit-hooks)
    - [GWDG Cluster](#gwdg-cluster)
  - [AI-Usage Card](#ai-usage-card)
  - [Acknowledgement](#acknowledgement)
  - [Disclaimer](#disclaimer)

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

A pre-trained
BERT ([BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805))
model was used as the basis for our experiments. The model was fine-tuned on the three tasks using a multitask learning
approach. The model was trained on the three tasks simultaneously, with a single shared BERT encoder and three separate
task-specific classifiers.

## Requirements

To install requirements and all dependencies using conda, run:

```sh
bash setup_gwdg.sh
```

This will download and install Miniconda on your machine, create the project's conda environment and activate it. It will also download the models used through the project and the POS and NER tags from spaCy.

## Data

We describe the datasets we used in the following table

| **Dataset**                       | **Task**                            | **Description**                                                                                                                               | **Size**                                                 |
| --------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Quora Dataset (QQP)               | Paraphrase detection                | Two sentences are given as input and a binary label (0, 1) is output indicating if sentences are paraphrases of one another                   | Train: 121, 620 <br /> Dev: 40, 540 <br /> Test: 40, 540 |
| Stanford Sentiment Treebank (SST) | Sentiment analysis (classification) | One sentence is given as input to be classified on a scale from 0 (most negative) to 5 (most positive)                                        | Train: 7, 111 <br /> Dev: 2, 365 <br /> Test: 2, 371     |
| SemEval STS Benchmark Dataset     | Textual Similarity (regression)     | Two sentences are given as input and their mutual relation is to be evaluated on continuous labels from 0 (least similar) to 5 (most similar) | Train: 5, 149 <br /> Dev: 1, 709 <br /> Test: 1, 721     |

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
| `--max_length`                          | Maximum length of tokens to chunk                                                                                                                          |
| `--n_hidden_layers`                     | Number of hidden layers to use                                                                                                                             |
| `--task_scheduler`                      | Choose how to schedule the task during training. Options are `random`, `round_robin`, `pal`, `para`, `sts`, `sst`                                          |
| `--projection`                          | How to handle competing gradients from different tasks. Options are `pcgrad`, `vaccine`, `none`                                                            |
| `--combine_strategy`                    | Needed in case of using projection. Options are `encourage`, `force`, `none`                                                                               |
| `--use_pal`                             | Add projected attention layers on top of BERT                                                                                                              |
| `--patience`                            | Number of epochs to wait without improvement before the training stops                                                                                     |
| `--use_smart_regularization`            | Implemented pytorch package to regularize the weights and reduce overfiting                                                                                |
| `--write_summary` or `--no_tensorboard` | Whether to save training logs for later view on tensorboard                                                                                                |
| `--model_name`                          | Choose the model to pre-train or fine-tune                                                                                                                 |

## Evaluation

The model is evaluated after each epoch on the validation set. The results are printed to the console and saved in the `--logdir`in case of setting `--no_tensorboard` to `store-false`. The model checkpoints are saved after each epoch in the `filepath` directory. After finishing the training, the model is loaded to be evaluated on the test set. The model's predictions are then saved in the `predictions_path`.

## Results

In the first phase of the project, we aimed at getting the baseline score for each task, where we trained the model for each task separately with the following parameters:

- option: `finetune`
- epochs: `10`
- learning rate: `1e-5`
- hidden dropout prob: `0.1`
- batch size: `64`

In the second phase, we used multitask training to let the model train on the data of all tasks, but with a different classifier for each task. This helped the model learn and the performance to improve since the tasks are dependent. In addition we performed single task training for the ones that didn't improve through the multitasking and tried to apply other approaches to improve their score.

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

This allowed us to set a standard training framework and try with other options for the sake of further improvement. The training parameters used as well as the branch where the related scripts exist can be found in the corresponding slurm file.

---

We achieved the following results

### [Paraphrase Identification on Quora Question Pairs (QQP)](https://paperswithcode.com/sota/paraphrase-identification-on-quora-question)

Paraphrase Detection is the task of finding paraphrases of texts in a large corpus of passages.
Paraphrases are “rewordings of something written or spoken by someone else”; paraphrase
detection thus essentially seeks to determine whether particular words or phrases convey
the same semantic meaning. This task measures how well systems can understand fine-grained notions of semantic meaning.

| **Model name**                | **Description**                                            | **Accuracy** | **Link to Slurm**                                                                                                                                                  |
| ----------------------------- | ---------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Pcgrad projection             | Projected gradient descent with round robin scheduler      | 86.4%        | [pcgrad](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train_pcgrad_classifier.out)                                    |
| Vaccine projection            | Surgery of the gradient with round robin scheduler         | 85.9%        | [vaccine](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train-vaccine_classifier.out)                                  |
| Combined BERT models          | 4 BERT models combined (3 sub-models and a gating network) | 85.7%        | [combined BERT](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-combined-models-traini-multitask_classifier-1332732.out) |
| Augmented Attention Multitask | Attention layer on top of BERT                             | 84.4%        | [attention multitask](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train-alldata-multitask-STS-QQP-improve.out)       |
| BiLSTM-Multitask              | BiLSTM layer on top of BERT                                | 83.9%        | [bilstm multitask](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-bilstm-train-multitask-classifier-1358610.out)        |
| Pal scheduler without vaccine | apply pal scheduler only                                   | 83.1%        | [pal no vaccine](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-pal-scheduler-no-vaccine.out)                           |
| Sophia with additional inputs | Sophia optimizer with POS and NER inputs                   | 82.5%        | [sophia with additional inputs](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-sophia-add-input-multitask.out)          |
| Bert-large                    | use bert-large model for multitasking                      | 82.4%        | [bert-large](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-bert-large-%20train-multitask_classifier.out)               |
| Max pooling                   | Max of the last hidden states' sequence                    | 81.9%        | [max pooling](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train-QQP-MAX_Pooling-1296205.out)                         |
| Baseline (single task)        | Single task training                                       | 80.6%        | [baseline](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/main/slurm_files/slurm-train-multitask_classifier-711664.out)                                 |
| Hierarchical-BERT             | Chunking the input up to a segment length                  | 75.6%        | [hierarchical bert](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-hirarchical-train-multitask-classifier.out)          |

### [Sentiment Classification on Stanford Sentiment Treebank (SST)](https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained)

A basic task in understanding a given text is classifying its polarity (i.e., whether the expressed
opinion in a text is positive, negative, or neutral). Sentiment analysis can be utilized to
determine individual feelings towards particular products, politicians, or within news reports.
Each phrase has a label of negative, somewhat negative,
neutral, somewhat positive, or positive.

| **Model name**                | **Description**                                            | **Accuracy** | **Link to Slurm**                                                                                                                                                  |
| ----------------------------- | ---------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| BiLSTM-Multitask              | BiLSTM layer on top of BERT                                | 53.0%        | [bilstm multitask](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-bilstm-train-sst-classifier-1360936.out)              |
| Max pooling                   | Max of the last hidden states' sequence                    | 52.8%        | [max pooling](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-max-pool-sst-single.out)                                   |
| Baseline (single task)        | Single task training                                       | 52.2%        | [baseline](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/main/slurm_files/slurm-train-multitask_classifier-711664.out)                                 |
| Bert-large                    | use bert-large model for multitasking                      | 51.2%        | [bert-large](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-bert-large-%20train-multitask_classifier.out)               |
| Vaccine projection            | Surgery of the gradient with round robin scheduler         | 50.4%        | [vaccine](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train-vaccine_classifier.out)                                  |
| Augmented Attention Multitask | Attention layer on top of BERT                             | 50.2%        | [attention multitask](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train-alldata-multitask-STS-QQP-improve.out)       |
| Combined BERT models          | 4 BERT models combined (3 sub-models and a gating network) | 49.1%        | [combined BERT](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-combined-models-traini-multitask_classifier-1332732.out) |
| Sophia with additional inputs | Sophia optimizer with POS and NER inputs                   | 48.8%        | [sophia with additional inputs](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-sophia-add-input-multitask.out)          |
| Hirarchical-BERT              | Chunking the input up to a segment length                  | 48.7%        | [hirarchical bert](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-hirarchical-train-multitask-classifier.out)           |
| Pcgrad projection             | Projected gradient descent with round robin scheduler      | 47.9%        | [pcgrad](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train_pcgrad_classifier.out)                                    |
| Pal scheduler without vaccine | apply pal scheduler only                                   | 45.0%        | [pal no vaccine](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-pal-scheduler-no-vaccine.out)                           |

### [Semantic Textual Similarity on STS](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts-benchmark)

The semantic textual similarity (STS) task seeks to capture the notion that some texts are
more similar than others; STS seeks to measure the degree of semantic equivalence [Agirre
et al., 2013]. STS differs from paraphrasing in it is not a yes or no decision; rather it
allows degrees of similarity from 5 (same meaning) to 0 (not at all related).

| **Model name**                | **Description**                                            | **Accuracy** | **Link to Slurm**                                                                                                                                                  |
| ----------------------------- | ---------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Vaccine projection            | Surgery of the gradient with round robin scheduler         | 89.3%        | [vaccine](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train-vaccine_classifier.out)                                  |
| Pcgrad projection             | Projected gradient descent with round robin scheduler      | 89.0%        | [pcgrad](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train_pcgrad_classifier.out)                                    |
| Pal scheduler without vaccine | apply pal scheduler only                                   | 86.5%        | [pal no vaccine](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-pal-scheduler-no-vaccine.out)                           |
| Augmented Attention Multitask | Attention layer on top of BERT                             | 85.8%        | [attention multitask](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-train-alldata-multitask-STS-QQP-improve.out)       |
| Bert-large                    | use bert-large model for multitasking                      | 85.3%        | [bert-large](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-bert-large-%20train-multitask_classifier.out)               |
| Combined BERT models          | 4 BERT models combined (3 sub-models and a gating network) | 85.1%        | [combined BERT](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-combined-models-traini-multitask_classifier-1332732.out) |
| Sophia with additional inputs | Sophia optimizer with POS and NER inputs                   | 84.3%        | [sophia with additional inputs](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-sophia-add-input-multitask.out)          |
| BiLSTM-Multitask              | BiLSTM layer on top of BERT                                | 80.7%        | [bilstm multitask](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-bilstm-train-multitask-classifier-1358610.out)        |
| Hierarchical-BERT             | Chunking the input up to a segment length                  | 71.4%        | [hierarchical bert](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-hirarchical-train-multitask-classifier.out)          |
| Max pooling                   | Max of the last hidden states' sequence                    | 53.4%        | [max pooling](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/BERT_PALs-Aly/slurm_clean/slurm-max-pool-sts-single.out)                                   |
| Baseline (single task)        | Single task training                                       | 41.4%        | [baseline](https://github.com/FrederikHennecke/DeepLearning4NLP/blob/main/slurm_files/slurm-train-multitask_classifier-711664.out)                                 |

## Methodology

In this section we will describe the methods we used to obtain our results for the given tasks.

### POS and NER Tag Embeddings

Enriching the corpus with subword embeddings can enhance the model's understanding of language as reported by T. Mikolov, et al.[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546) and P. Bojanowski et al. [Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606). We used [spaCy](https://spacy.io/) package for that purpose to extract the POS and NER tags from the input text and feed their embeddings to the model together with the input sequence.

- **POS:** Identifying the grammatical category of words (noun, verb, adjective, etc.) helps the model understand the structure of a sentence and disambiguate meanings which then results in better capturing the relationships between words.

- **NER:** Recognizing named entities (names, organizations, etc.) makes the model focus on relevant words and therefore better sentiment analysis of the text.

Adding such semantic information can make the model robust and therefore better generalization to unseen data.

#### Experimental Results

In contrast to our expectations, adding POS and NER tags to the model combined with the baseline AdamW optimizer did not improve the performance too much, rather it made the training too slow and consumed lots of computational resources without an actual benefit. The reason for the high training costs is that we train all the available data and indeed extracting the tags from each word, computing the embeddings and feeding them to the model is quite expensive.

#### Explanation of Results

The reason why including additional inputs did not enhance the performance is that Large Language Models (LLMs) like Bert are pre-trained on massive corpora whose embeddings are contextual, meaning that the representation of a word depends on the context around it in the specified window which allows the model to capture information relevant to POS and NER from the text. Researchers have conducted studies where they probe the hidden layers to asses the amount of syntactic or semantic information encoded in the layers. ([A Structural Probe for Finding Syntax in Word Representations](https://aclanthology.org/N19-1419.pdf), [BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/pdf/1905.05950)).

---

### Optimizers

#### Sophia

Sophia is a **S**econd-**O**rder Cli**p**ped Stoc**h**astic Optimiz**a**tion method that uses. It uses the Hessian matrix as an approximation of second-order information to represent the curvature of the loss function which can lead to faster convergence and better generalization. Sophia incorporates an adaptive learning rate mechanism, adjusting the step size based on the curvature of the loss landscape and uses clipping to control the worst-case update size. in all directions, safeguarding against the negative impact of inaccurate Hessian estimates. Thanks to the light weight diagonal Hessian estimate, the speed-up in the number of steps translates to a speed-up in total compute wall-clock time.

##### Implementation

We used the Hutchinson's unbiased estimator of the Hessian diagonal in our implementation, which is to sample from the spherical Gaussian distribution. It can be activated in the training script by setting the `--optimizer` option to `sophiah`. The estimator can be used for all tasks including classification and regression, and requires only a Hessian vector product instead of the full Hessian matrix. It also has efficient implementation in PyTorch and JAX as mentioned in the original paper by H. Liu, et al. [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/pdf/2305.14342).

##### Experimental Results

The implementation of Sophiah with additional inputs from spaCy did actually improve the performance for the QQP and STS tasks compared to the baseline AdamW, however it did not improve the SST task. Running Sophiah without POS and NER tags, we observed that the convergence is slower than AdamW.

##### Explanation of Results

Sophia optimizer is basically designed to be computationally feasible for pre-training large scale language models, which is different from the frame in which we used it (fine tuning of a relatively small corpus than usual). J.Kaddour, et al. ([No Train No Gain: Revisiting Efficient Training Algorithms For Transformer-based Language Models](https://arxiv.org/pdf/2307.06440)) have applied the Sophia optimizer using the Gauss-Newton-Barlett method for downstream tasks and observed that it achieves comparable performance as AdamW.

---

### Gradient Optimization (Projectors)

It is an optimization technique to handle the conflicting gradients during multitask-training. Since we train the model on all the data simultaneously, the gradients from these tasks might diverge when the gradients from tasks under training point in different directions. That indeed could lead to suboptimal performance and slowing down the training as well.

#### Pcgrad

T. Yu et al. ([Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782)) proposed the Projected Gradient Descent or Pcgrad method which is a form of gradient surgery. They do the surgery by projecting the gradients of each task onto the normal plane of the gradient of any other task. if the projection is large, then the gradients are conflicting. At the end the projected gradients are summed up and used to update the model. This method proved to effectively reduce the conflict between gradients and enhance the converging speed of the model.

##### Experimental results

PCgrad achieved a high performance on the QQP (first rank) and the STS (second rank) tasks, however it worsened the SST (lower than the baseline).

##### Explanation of Results

Our hypothesis is that the pair sequence tasks (STS, QQP) are complicated and they have conflicting gradients, which is why they benefited from Pcgrad technique. Unlike the SST task which includes only one sequence and its gradients are less likely to conflict which is why the benefits of surgery are less pronounced for that task. Following that hypothesis, D. Saykin and K. Dolev ([Gradient Descent in Multi-Task Learning](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/DavidSaykinKfirShmuelDolev.pdf)) reported that training the SST task separately with using the Pcgrad surgery achieves better results than in multitask-training framework.

#### GradVac

GradVac stands for Gradient Vaccine which was originally proposed by Z. Wang, et al. ([GRADIENT VACCINE: INVESTIGATING AND IMPROVING MULTI-TASK OPTIMIZATION IN MASSIVELY MULTILINGUAL MODELS](https://arxiv.org/pdf/2010.05874)). This technique was developed to mitigate the negative effects of conflict gradients in multitask-training. It works the same as Pcgrad, but involves a process of first clipping the gradients before aligning them, which adds computational overhead, however it is still efficient and scalable to large models and corpora.

##### Experimental results

It achieved relatively comparable results to Pcgrad, additionally it performed better on the SST task.

##### Explanation of Results

The additional clipping introduced in GradVac can prevent the model from making overly aggressive updates that could lead to instability. Therefore it helps stabilize the training, especially in case of noisy gradients due to class imbalance, as for the SST data, which could lead to an improvement as we obtained.

---

### Schedulers

Schedulers in general are mechanisms that help adjusting the learning rate dynamically during training, which can lead to better performance and faster convergence. Schedulers are useful for escaping the local minima by momentarily increasing the learning rate, preventing overfiting, and efficient training. There are lots of schedule algorithms and we will consider some of them.

#### Linear Warmup

The idea behind it is to start with a small learning rate so that the basic features of the data are learned, then gradually increase it in a linear way over a predefined number of steps or epochs. After the warmup period, the learning rate transits to another schedule such as constant rate or a decay rate. Linear warmup has the benefits of stabilizing the training and avoiding exploding gradients in case of using large batch sizes, large models or large dataset. Therefore, it was used by the state-of-the-art LLM research ([BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding:](https://arxiv.org/pdf/1810.04805), [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)). We implemented the hyper-parameters parameters of the scheduler from the first link (Delvin, et al.).

##### Experimental results

We applied the linear warmup scheduler on the single SST task combined with a Bidirectional LSTM (BiLSTM) layer on top of BERT. We set the scheduler hyper-parameters `num_train_steps = len(sst_train_dataloader) * num_train_epochs` and `num_warmup_steps = 0` as the default. This achieved a relatively higher performance than the baseline.

#### Plateau

The plateau scheduler reduces the learning rate when a specific metric (validation loss or accuracy) plateaus, namely renmins unchanged for specific number of training steps or epochs. This helps with fine-tuning the model by keeping the learning rate high when the model still learning effectively. and then reducing it when it comes to complex structures in the data (when improvement stops). The amount of reducing the learning rate is pre-determined by a factor that defaults to 0.1. The number of epochs to wait on the plateau is pre-determined by the patience parameter which we set to 2 for faster training, and we monitor the validation accuracy.

#### Cosine

The basic idea behind the cosine annealing scheduler is to reduce the learning rate following a cosine function rather than a step-wise or exponential decay. This can lead to more effective training because it smoothly decreases the learning rate in a way that can help the model escape local minima and explore the loss landscape more effectively. I. Loshchilov and F. Hutter ([SGDR: STOCHASTIC GRADIENT DESCENT WITH
WARM RESTARTS](https://arxiv.org/pdf/1608.03983)) proposed a warm restart technique, meaning that the learning rate is set periodically to the maximum value and then follows the cosine annealing schedule. They applied their methodology on CIFAR-10 and CIFAR-100 in the computer vision scheme.

#### Round Robin

This algorithm is basically implemented by process and network schedulers in computing in which time slices (from 10 to 100 milliseconds) are assigned to each process in equal portions and in circular order, namely cycling trough the different tasks one after the other. The problem with round robin scheduler technique is that if the dataset contains more instances of one task than another (e.g QQP and SST), it will repeat several times all the examples of the task with few data points before it repeats all the examples of the other task. This will lead to overfitting for the task that repeats a lot and underfitting for the other one. To this end Stickland and Murray ([BERT and PALs: Projected Attention Layers for
Efficient Adaptation in Multi-Task Learning](https://arxiv.org/pdf/1902.02671)) proposed a novel method for scheduling training. At first the tasks are sampled proportionally to their training set size but then to avoid interferences, the weighting is reduced to have tasks sampled more uniformly.

##### Experimental results

We implemented the round robin schedulers with the gradient surgery algorithm and showed a high performance. However, we need to turn off the other optimization methods to have a rough estimate of the effect of the scheduler on the training in terms of score and convergence speed.

#### PAL

The PAL (Performance and Locality) scheduler was developed by R. Jain, et al ([PAL: A Variability-Aware Policy for Scheduling ML Workloads in GPU Clusters](https://arxiv.org/pdf/2408.11919v1)) to address the issue of performance variability in large-scale computing environments and to handle machine Learning workloads in GPU clusters. Thus, this method aims at harnessing the performance variability by redesigning scheduling policies for GPU clusters to consider performance variability. First, the authors identified which GPUs exhibit similar performance variability and the used K-Means clustering to group them. Then they extend their novel algorithm by including also locality of GPUs that are widely distributed on the cluster.

##### Experimental results

We used the PAL scheduler without vaccine and showed relatively good performance on the STS and QQP tasks. However, we could not run it with the Gradient surgery algorithms due to memory issues on the GWDG cluster.

#### Random

Random scheduler is included for the case of single task training, where at each step of the training the data of the selected task will be sampled randomly to be more robust against overfitting.

---

### Regularization

Due to the limited data from the target tasks/domain and the extremely high complexity of the pre-trained open domain models, aggressive fine-tuning often makes the adapted model overfit the training data of the target task/domain and therefore does not generalize well to unseen data. Regularization comes into play to mitigate this issue by introducing additional constraints or penalities on the model's parameters during training, which helps simplify the model and improves its generalization.

#### SMART

Sharpness-Aware Minimization with Robustness Techniques (SMART) is one of the efficient regularization techniques developed by H. Jiang, et al. ([SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization](https://aclanthology.org/2020.acl-main.197.pdf)). SMART is built on top of SAM introduced by P.Foret et al. ([SHARPNESS-AWARE MINIMIZATION FOR EFFICIENTLY IMPROVING GENERALIZATION](https://arxiv.org/pdf/2010.01412v3)) which was originally proposed to simultaneously minimize loss values and loss sharpness. The sharpness refers to how much the loss value changes when there are small perturbation in the model parameters and the goal of SAM functions is to optimize the parameters that both minimize the loss as well as produce a flat loss landscape around the minimum. SMART icludes more robustness techniques for better generalization. The first one is smoothness-inducing adversarial regularization which manages the complexity of the model. Its role is to encourage the output of the model not to change much, when injecting a small perturbation to the input and therefore enforces the smothness of the model and controls its capacity.
The second is Bregman proximal point optimization which can prevent aggressive updating by only updating the model within a small neighborhood region of the previous iterate and hence controls aggressive updating and stabilizes the training.

##### Exprimental results

We implemented the SMART regularization in our gradient surgery training (Pcgrad and vaccine) as well as with using PAL scheduler, where the performance is high especially for the STS and QQP tasks. In order to spot the focus only on SMART, we would then need to switch off the optimization methods to estimate its impact on the baseline.

---

## Details

### Data Imbalance

The dataset we worked is imbalanced, either inside the individual datasets, such as the imbalanced classes of SST or in terms of the size of the datasets of different tasks when it comes to multi-tasking. Such effects of data imbalance can misguide the model to the majorioty classes at the expense of the minor ones and therefore incorrect predictions on test data. There are two solutions to overcome this:

- **Fixed samples per epoch:** Is to select a fixed number of samples randomly from the dataloader of each task in each epoch. We ran our experiment with 10000 and 20000, but did not find a significant differtence on performance. In that case 10000 samples will be better for computational reaseons. This strategy is somehow simple and does not actually solve the problem since we still cannot balance the classes to mitigate overfitting, especially for the SST task.

- **Data augmentation:** J. Wei and K. Zou ([EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/pdf/1901.11196)) came up with the idea of doing Easy Data Augmentation (EDA) for boosting performance on text classification tasks. EDA consists of four operations:

  - Synonym replacement: Randomly choose _n_ words from the sentence that are not stop words. Replace each of these words with one of its synonymschosen at random.
  - Random insertion: Find a random synonym of a random word in the sentence that is not a stop word. Insert that synonym into a random position in the sentence. Do this _n_ times.
  - Random swap: Randomly choose two words in the sentence and swap their positions. Do this _n_ times.
  - Random deletion: Randomly remove each word in the sentence with probability $p$

  It is adequate for small datasets. It also saves computation costs since the authors reported that they achieved the same accuracy with only 50% of the training set as with the full available data. There is also another **spaCy** integrated package of data augmentation ([augmenty](https://github.com/KennethEnevoldsen/augmenty)) that was developed by K. Enevoldsen ([AUGMENTY: A PYTHON LIBRARY FOR STRUCTURED TEXT AUGMENTATION](https://arxiv.org/pdf/2312.05520)) that serves the same purpose.

Although, the data augmentation approach seems more promising, we unfortunately came about its idea a bit late in the project's schedule, which is why we only implement the first approach and leave the latter for further investigations.

### Classifier Architecture

We share the BERT layers among all the datasets and build a classifier on top that is characteristic for each task. We tried many different architectures to improve the performance of both single and multitask training. We use the first token of the sequence **[CLS]** of the last hidden state that encodes all the learned parameters of the model during the open domain pre-training phase and feed it to the classifiers, as already described in these research studies ([Devlin et al. ](https://arxiv.org/pdf/1810.04805), [Sun et al.](https://arxiv.org/pdf/1905.05583#page=9&zoom=100,402,290), [Karimi et al.](https://arxiv.org/pdf/2001.11316)). The classifiers we build are:

- **Sentiment Analysis Classifier:** At first we build three fully connected layers of size `768 * 768` each, with `relu` activation applied after each layer. Then the last layer which is of size `hidden size * num_classes`. These layers refine the BERT embeddings into logits corresponding to each class. We take the softamx and compute the loss betwee the predicted labels and the ground truth using the cross enropy function from PyTorch.

- **Semantic Textual Similarity Regressor:** We compute the embeddings for the sequence pair and the cosine similarity between the embeddings vectors. The cosine similarity distance is an estimate of how semantically similar the two sentences are. The output is then multiplied by 5 to scale it against the true labels. Since it is a regression task, we use the Mean Square Error (MSE) to compute the loss.

- **Paraphrase Detection Classifier:** We first generate the emnbeddings for each pair, then apply a fully connected layer of size `768 * 768` to both embedding vectors. We take the absolute sum and difference between the two vectors and concatenate them. The concatenated vector of dimension `2 * 768` is fed to two linear layers and then the to the binary classifier. We again applied the `relu`activation after each of the connected layers in the last processing phase. At the end, we take the sigmoid function to get the predicted probabilities (logits) and compare them with the true labels through the cross entropy loss function.

### Augmented Attention

We add an attention layer on top of the BERT model to refine the model's focus on certain parts of the input sequence, iproving its ability to capture relevant contextual information for the downstream tasks. It is important for the model to give more attention to certain tokens in the input sequence (great, amazing, terrible, bad, etc.) to better classify it or the relations between sequence pairs to decide if they are actually paraphrased. For that purpose we apply an attention mechanism inspired by A. Vaswani, et al. ([Attention Is All You Need](https://arxiv.org/pdf/1706.03762)) that takes in the last hidden state of the model and apply a weighted sum mechanism to enhance the importance of specific tokens and ignores the others. Also, G. Limaye et al. ([BertNet : Combining BERT language representation with Attention and CNN for Reading Comprehension](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15783457.pdf)) implemented a Query-Context attention layer for the question answering task (English-Arabic-English) and proved a significant performance on "difficult" questions.

### BiLSTM

We added 3 Bidirectional Long-Short Term Memory (BiLSTM) layers on top of BERT to further capture long sequential dependencies and richer contextual information. A BiLSTM processes the sequence in both forward and backward directions to capture sequence relations across time steps. We looked through a hidden size of 128 sequence length of the last hidden state of BERT. We also added a dropout and an average pooling layer to avoid overfitting. The model however was still overfitting which therefore we then reduced the BiLSTM layers to one.

### Feed more Sequences

We set a `train_mode`option to include contexual information from some of the most top BERT layers instead of the last hidden state only. We first tried with all the 12 layers sequences, but that worsened the performance. Inspired from ([Adversarial Training for Aspect-Based Sentiment Analysis with BERT](https://arxiv.org/pdf/2001.11316)) Then we took the **[CLS]** output from the last four layers and took the average pooling over them. That approach has reduced overfitting and enhanced the performance, espcially for the SST task.

### Hierarchical BERT

We tried to investigate the hirarchical structure in the input sequences by chunking the input with a segment length of 34 and investigating the relation between the chunks and the sentence as a whole. This method is designed to process text at multiple levels granularity to capture both local and global contexts. Although this approach has improved the STS and QQP scores, it is particularly desgined to handle long text sequences as reported by J. Lu, et al. ([A Sentence-level Hierarchical BERT Model for Document Classification with Limited Labelled Data](https://arxiv.org/pdf/2106.06738)), which does not apply perfectly to our small sequence data. The results for the QQP and SST tasks went under the baseline, as we suspected. Another reason for the result is that we don't actually know where we should split the sequence. It could be that the the sentence loses its meaning when it gets splitted and the resultant chunks become meaningless on their own. That's why as we explained earlier, that approach could be helpful for long sequences when we want to investigate the relation between sentences on a local level and the global document.

### CNN BERT

Seeking improvements, we tried to add Convolution Network (CNN) layers on top of BERT. CNN is known for its ability to scan features to a high depth and capture complex relations in the input, which we thought it could be useful for our tasks. We added threee 1-dimensional CNN layers with input channel of size 768 (embeddings size or number of features per input token) and an output dimension of 512 (size of the feature map for each token). We implemented convolutional filters of size $5 \times 5$ meaning that the filter will consider five adjacent tokens at a timeas it slides along the sequence. We used a paddine of one to ensure that the output sequence has the same length as the input. Unexpectedly, this approach showed a degradation in performance, which we decided to drop it.

### Combined Models

Here we use a way of combined BERT models called Mixture of Experts (MoE) which is composed of several sub-models or experts and a gating mechanism to dynamically select which experts to use for a given input. This approach is particularly useful when different subsets of data may benefit from specialized models. State-of-the-art models use the MoE such as Switch Transformer and Gshard. We basically implemented three BERT sub-models and one BERT model that functions as a gating network. We applied the softmax to consider the contribution of each expert. Our experiment showed a boost in performance compared to other approaches.

### BERT-Large

We tried to run BERT-Large on our multitask training, which has a larger embeddings dimension than minBERT and therefore able to encode more contextual information. However, we did not perform pre-training, which needs a much large corpus than that we have. That' why model's performance was comparable and did not show any improvement.

### PALs

Projected Attention Layers (PALs) are a novel technique introduced by ([BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning](https://proceedings.mlr.press/v97/stickland19a/stickland19a.pdf)) to efficiently adapt LLMs to new tasks without the need to pre-train the model. This can be done by introducing multi-head attention layers parallel to the BERT layers. The additional layers can be fine-tuned separately from the base model. The attention layers project the input into task-specific subspace and only the parameters of the PALs get updated, while keeping the parameters of the base model unchanged or frozen which saves computational costs. Also, PAls allow for sharing parameters among layers in multitask-learning which could have a high positive impact on the performance. Although, We integrated the PAls functions into our framework in branch [BERT_PALs-Aly](https://github.com/FrederikHennecke/DeepLearning4NLP/tree/BERT_PALs-Aly), we haven't measured its performance yet, since we still have to complete processing and debuging its implementation. We leave that part to further investigations of the project.

---

## Hyperparameter Search

We used [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) to perform hyperparameter tuning for the multitasking and select the best parameters that reduce overfitting. We used [Optuna](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.optuna.OptunaSearch.html) to search the hyperparameter space and [AsyncHyperBandScheduler](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.AsyncHyperBandScheduler.html) as the scheduler. The hyperparameters were searched for the whole model on all training data, not for each task individually to avoid overfitting to a single task. The trained models were evaluated on the validation data and the best one was selected based on the validation results. The metrics applied were accuracy for the sentiment analysis and paraphrase detection, and the Pearson correlation for the semantic textual similarity.

---

## Computation Resources

We trained our models on the [GWDG](https://docs.hpc.gwdg.de/usage_guide/slurm/gpu_usage/index.html) HPC clusters of the George-August university of Göttingen. We used differents set ups depending on the implemented approach and how much computation costs it needs, We mainly used Grete and Emmy cluster nodes and trained the basic multitasking approach (with augmented attention) with the following configurations:

- **GPU cores:** 4
- **Model:** NVIDIA A100-SXM4-40GB
- **NVIDIA driver version:** 535.104.12
- **XNNPACK available:** True
- **OS:** Rocky Linux 8.8 (Green Obsidian) (x86_64)
- **CPU cores per socket:** 32

More details about the computation utilities can be found in the slurm file corresponding to each training.

---

## Contributors

| **Mohamed Aly**                                             | **Contrib1** | **contrib2** |
| ----------------------------------------------------------- | ------------ | ------------ |
| BERT Tasks (SST, STS, QQP) and their implemented approaches |              |              |
| Writing the README file                                     |              |              |

---

## Licence

The project involves the creation of software and documentation to be released under an open source licence.
This license is the Apache License 2.0, which is a permissive licence that allows the use of the software for
commercial purposes. The licence is also compatible with the licences of the libraries used in the project. Parts of the project were inspired by other open source codes (see acknowledgment). Using or modifying these parts in this project should adhere to the licence (Copyrights) of the original authors.

---

## Usage Guidelines

To run or contribute to this project, please follow these steps:

Clone the repository to your local machine

```sh
git clone git@github.com:FrederikHennecke/DeepLearning4NLP.git
```

Add the upstream repository as a remote and disable pushing to it (only pull but not push)

```sh
git remote add upstream https://github.com/FrederikHennecke/DeepLearning4NLP.git
git remote set-url --push upstream DISABLE
```

You can pull the latest updates of the project with

```sh
git fetch upstream
git merge upstream/main
```

or you can specify a branch to pull from after fetching the changes

```sh
git pull <branch-name>
```

### Pre-commit Hooks

The code quality and integrity is checked with pre-commit hooks. This is used to ensure that the code quality is consistent and that the code is formatted uniformly. To install the pre-commit hooks in your local repository, run the following

```sh
pip install pre-commit
pre-commit install
```

The pre-commit hooks will run automatically before each commit. If the hooks fail, the commit will be aborted. You can skip the pre-commit hooks by add `--no-verify`flag to your commit command.

The installed pre-commit hooks are:

- [`black`](https://github.com/psf/black) - Code formatter (Line length 100)
- [`flake8`](https://github.com/PyCQA/flake8) - Code linter (Selected rules)
- [`isort`](https://github.com/PyCQA/isort) - Import sorter

### GWDG Cluster

You can run the project on the GWDG cluster, in case you are a memeber of the GWDG community and specify the [GPU usage](https://docs.hpc.gwdg.de/how_to_use/slurm/gpu_usage/index.html) you need. To start an interactive session for testing the project, execute the following

```sh
srun -p grete:shared --pty -n 1 -C inet -c 32 -G A100:1 --interactive /bin/bash
```

This command will allocate resources of one node with 32 CPU cores and A100:1 GPU NVIDIA core on the grete shared cluster. For the actual run of the script and usage of higher computational power, you need to submit a job. The job configurations are found in the `run_train.sh` script where you can select the file to run with the options you like to activate and your prefered parameters. You can also adjust the resources you want to use (nodes per task, GPU cores, memory, etc.) and add your email to receive a notification of the job's status. You can submit a job with the following command

```sh
sbatch run_train.sh
```

You can check the status of your submitted jobs with

```sh
squeue --me
```

This will show the job-id, if the job is running or still pending allocation, time elapsed since submission and the gpu node assigned to it.

The job details will be saved to the `slurm_files` directory where you can see a summary of the resources used, the installed packages, the executed command and the git related information (current branch, latest commit, etc.). If `checkpoint`and `write_summary` options are activated, then the training logs will be saved to the `logdir` directory. The saved results can be viewed on the tensorboard. To create a tunnel to your local machine and start the tensorboard on the cluster, run the following

```sh
ssh -L localhost:16006:localhost:6006 <username>@glogin.hlrn.de
module load anaconda3
conda activate dnlp2
tensorboard --logdir logdir
```

---

## AI-Usage Card

Artificial Intelligence (AI) aided the development of this project. For transparency, we provide our [AI-Usage Card](./ai-usage-card-Aly.pdf/) at the top. The card is based on [https://ai-cards.org/](https://ai-cards.org/).

---

## Acknowledgement

- The project description, partial implementation, and scripts were adapted from the default final project for the
  Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun,
  John Cho, and their (large) team (Thank you!)

- The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon
  University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
  created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig (Thank you!)

- Multitask-training with augmented attention is inspired from [Lars Kaesberg and Niklas Bauer](https://github.com/token-tricksters/deep-learning-nlp) under [Apache License 2.0](https://github.com/token-tricksters/deep-learning-nlp/blob/main/LICENSE) (Thank you!).

- PALs and schedulers implementations are inspired from [Josselin Roberts, Marie Huynh and Tom Pritsky](https://github.com/JosselinSomervilleRoberts/BERT-Multitask-learning) under [Apache License 2.0](https://github.com/JosselinSomervilleRoberts/BERT-Multitask-learning/blob/main/LICENSE) (Thank you!).

- The structure of README file is adapted from [good_example](https://github.com/maly-phy/dnlp_readme_template/tree/main/good_example) (Thank you!)

---

## Disclaimer

We didn't get enough time to clean the repository and arrange the scripts in a perfectly engineered way. We also couldn't merge all the implemented approaches of the BERT part to the main. Therefore, we advise the user to refer to the original branches, which can be found in the slurm file in the tables of the [results section](#results)
