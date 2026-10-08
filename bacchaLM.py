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

