"""
BacchaLM

Usage:
    python BacchaLM.py train
    python BacchaLM.py generate "the cat"
"""

import os
import sys
import json
import numpy as np




T = 4          # context length (words the model sees when training or generating)
D = 16         # embedding dimension - idk what this is. i got this from deepseek
LR = 0.1       # learning rate - i guess it's okay for now
STEPS = 3000   # training steps - might need to increase
BATCH = 8      # batch size
SEED = 0
DATA_PATH = "data/dataset.txt"      #i don't have a dataset but will make one
CKPT_PATH = "BacchaLM.npz"          #the trained weights will be saved in this
META_PATH = "BacchaLM.json"         #vocab and config will be saved in this

np.random.seed(SEED)    #meh



#Loading Dataset
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

#vocabulary building
def build_vocab(text):
    words = text.split()
    vocab = sorted(set(words))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for w, i in word_to_id.items()}
    return words, vocab, word_to_id, id_to_word

#dataset making
def make_dataset(words, word_to_id):
    ids = [word_to_id[w] for w in words]
    X, Y = [], []
    for i in range(len(ids) - T):
        X.append(ids[i : i + T])
        Y.append(ids[i + 1 : i + T + 1])
    return np.array(X, dtype=np.int64), np.array(Y, dtype=np.int64)

