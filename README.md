# SC4002-NLP

SC4002 Natural Language Processing group assignment for Sem 1 AY 26/27

## Group Members

- Li Shuhui
- Lee Yun-Tung
- Hou Junning
- Chin Li-Loong
- Sera Tan Zhe Xuan
- Low Zhen Jie Alden

## Project Overview

This project's task is to build a system that identifies a user’s intent and extracts the slots in an English assistant request. This is done in 2 stages:

_stage 1_: develop a baseline model
_stage 2_: collaborate with AI to diagnose weaknesses and implement improvements

The models will then be evaluated on their performance for the above stages.

## Installation and Setup

fill in later

### GRU with attention

[`Source/main_GRU_Attention.ipynb`](Source/main_GRU_Attention.ipynb) contains the
complete joint intent classification and BIOES slot filling implementation,
following the shared `Source/main.ipynb` template.

See [setup and training details](docs/gru_attention_baseline.md) and
[the trained checkpoint, logs and evaluation results](Results/gru_attention/seed_42/README.md).

## File structure

Models are jupyter notebooks in located in the Source folder

## Results

| Model | Split | Intent accuracy (%) | Slot micro F1 (%) |
| --- | --- | ---: | ---: |
| GRU with attention | Validation | 85.49 | 73.18 |
| GRU with attention | Test | 83.96 | 70.44 |

The run trained for 40 epochs with seed 42. Epoch 39 was selected using the mean
of validation intent accuracy and slot micro F1; the test split was evaluated
after checkpoint selection. Total trainable parameters: 1,019,673.

## Report

The report used for submission is found in the Docs folder. This report contains details and contributions for our implementations.
