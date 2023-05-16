import torch
import torch.nn as nn
import numpy as np
import json
import sys
import torch
import pickle

def get_vocab(vocab_pkl_path):
    with open(vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    return vocab,vocab_size