import numpy as np
import nltk
import os

import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import seaborn as sns

def tokenize_text(text, tokenizer):
    """Tokenizes a given text using the provided tokenizer."""
    return tokenizer.tokenize(text)