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



#-------------------------------------------------------------------------------------------------------------------------------------------------

#softmax - Converts a vector of raw scores into a probability distribution.
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


#max(0, x). Nonlinearity. Without it, stacking linear layers would just be one big linear layer — the model couldn't learn anything complex.
# need to learn more about this
def relu(x):
    return np.maximum(0, x)


#Creates a random weight matrix of shape (in_dim, out_dim). scale=0.1 keeps values small so early gradients don't explode.
def init_linear(in_dim, out_dim, scale=0.1):
    return (np.random.randn(in_dim, out_dim) * scale).astype(np.float32)

#-------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------

#entire model as a dictionary of arrays.
def init_params(V):
    return {
        "E":    (np.random.randn(V, D) * 0.1).astype(np.float32),   # token embeddings
        "P":    (np.random.randn(T, D) * 0.1).astype(np.float32),   # position embeddings
        "Wq":   init_linear(D, D),
        "Wk":   init_linear(D, D),
        "Wv":   init_linear(D, D),
        "Wo":   init_linear(D, D),
        "W1":   init_linear(D, 4 * D),
        "b1":   np.zeros(4 * D, dtype=np.float32),
        "W2":   init_linear(4 * D, D),
        "b2":   np.zeros(D, dtype=np.float32),
        "Wout": init_linear(D, V),
        "bout": np.zeros(V, dtype=np.float32),
    }

'''
Param	            Shape	            Purpose
E	                (V, D)	            Token embedding table. Row i = vector for word id i.
P	                (T, D)	            Position embedding. Row t = vector for position t.
Wq, Wk, Wv	        (D, D)	            Attention projections: query, key, value.
Wo	                (D, D)	            Attention output projection (back to D).
W1, b1	            (D, 4D), (4D,)	    Feedforward layer 1. Expands 4×.
W2, b2	            (4D, D), (D,)	    Feedforward layer 2. Compresses back.
Wout, bout	        (D, V), (V,)	    Output projection to vocab size.
'''





#THE OG ATTENTION
# need to understand it more clearfully, currently using deepseek

def attention(Q, K, V):

    N, T_, D_ = Q.shape
    
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(D_)           # (N, T, T)
    
    mask = np.triu(np.ones((T_, T_), dtype=bool), k=1)
    
    scores = np.where(mask, -1e9, scores)
    
    A = softmax(scores, axis=-1)
    
    out = A @ V
    
    return out, A




#plugging attention to the model.
def forward(X, p):
    h0 = p["E"][X] + p["P"][None, :, :]                        # (N, T, D)

    Q = h0 @ p["Wq"]
    K = h0 @ p["Wk"]
    V = h0 @ p["Wv"]
    attn_out, A = attention(Q, K, V)
    h_attn = h0 + attn_out @ p["Wo"]                           # residual

    h_ff_in = h_attn
    h_ff_hidden = relu(h_ff_in @ p["W1"] + p["b1"])
    h_final = h_ff_in + h_ff_hidden @ p["W2"] + p["b2"]        # residual

    logits = h_final @ p["Wout"] + p["bout"]                   # (N, T, V)
    probs = softmax(logits, axis=-1)

    cache = (h0, Q, K, V, A, attn_out, h_attn,
             h_ff_in, h_ff_hidden, h_final, logits, probs)
    return logits, probs, cache



#loss function - cross entropy
def cross_entropy(logits, Y):
    N, T_, V_ = logits.shape
    probs = softmax(logits, axis=-1)
    correct = probs[np.arange(N)[:, None], np.arange(T_)[None, :], Y]
    loss = -np.log(correct + 1e-9).mean()
    return loss, probs




#toooooo big
# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------
def backward(X, Y, p, cache, lr):
    (h0, Q, K, V, A, attn_out, h_attn,
     h_ff_in, h_ff_hidden, h_final, logits, probs) = cache
    N, T_, D_ = h0.shape
    V_ = p["E"].shape[0]

    # dL/dlogits for softmax + cross-entropy
    dlogits = probs.copy()
    dlogits[np.arange(N)[:, None], np.arange(T_)[None, :], Y] -= 1
    dlogits /= (N * T_)

    # ---- output projection ----
    dWout = h_final.reshape(-1, D_).T @ dlogits.reshape(-1, V_)
    dbout = dlogits.reshape(-1, V_).sum(axis=0)
    dh_final = dlogits @ p["Wout"].T                           # (N, T, D)

    # ---- feedforward block ----
    dh_ff_in_res = dh_final                                    # residual path
    d_hidden = dh_final @ p["W2"].T
    d_hidden[h_ff_hidden <= 0] = 0                             # relu backward
    dW2 = h_ff_hidden.reshape(-1, 4 * D_).T @ dh_final.reshape(-1, D_)
    db2 = dh_final.reshape(-1, D_).sum(axis=0)
    dW1 = h_ff_in.reshape(-1, D_).T @ d_hidden.reshape(-1, 4 * D_)
    db1 = d_hidden.reshape(-1, 4 * D_).sum(axis=0)
    dh_ff_in = d_hidden @ p["W1"].T
    dh_attn = dh_ff_in_res + dh_ff_in                          # (N, T, D)

    # ---- attention output projection ----
    dWo = attn_out.reshape(-1, D_).T @ dh_attn.reshape(-1, D_)
    d_attn_out = (dh_attn @ p["Wo"].T).reshape(N, T_, D_)

    # ---- attention block ----
    dA = d_attn_out @ V.transpose(0, 2, 1)                     # (N, T, T)
    dV = A.transpose(0, 2, 1) @ d_attn_out                     # (N, T, D)

    # softmax backward: dS = A * (dA - sum(dA*A))
    dS = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
    dS /= np.sqrt(D_)

    dQ = dS @ K
    dK = dS.transpose(0, 2, 1) @ Q

    dWq = h0.reshape(-1, D_).T @ dQ.reshape(-1, D_)
    dWk = h0.reshape(-1, D_).T @ dK.reshape(-1, D_)
    dWv = h0.reshape(-1, D_).T @ dV.reshape(-1, D_)
    dh0_attn = dQ @ p["Wq"].T + dK @ p["Wk"].T + dV @ p["Wv"].T
    dh0 = dh_attn + dh0_attn

    # ---- embedding + positional gradients ----
    dE = np.zeros_like(p["E"])
    np.add.at(dE, X, dh0)
    dP = dh0.sum(axis=0)

    # ---- SGD update ----
    grads = {
        "E": dE, "P": dP,
        "Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo,
        "W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
        "Wout": dWout, "bout": dbout,
    }
    for k in p:
        p[k] = (p[k] - lr * grads[k]).astype(np.float32)
    return p





#training
def train():
    text = load_text(DATA_PATH)
    words, vocab, word_to_id, id_to_word = build_vocab(text)
    V = len(vocab)
    print(f"Loaded {len(words)} words, vocab size {V}")

    X, Y = make_dataset(words, word_to_id)
    N = X.shape[0]
    print(f"Training samples: {N}")

    p = init_params(V)

    for step in range(STEPS):
        # random minibatch
        idx = np.random.randint(0, N, size=min(BATCH, N))
        Xb, Yb = X[idx], Y[idx]

        logits, probs, cache = forward(Xb, p)
        loss, _ = cross_entropy(logits, Yb)
        p = backward(Xb, Yb, p, cache, LR)

        if step % 200 == 0 or step == STEPS - 1:
            print(f"step {step:5d}  loss {loss:.4f}")

    # save
    np.savez(CKPT_PATH, **p)
    with open(META_PATH, "w") as f:
        json.dump({
            "word_to_id": word_to_id,
            "id_to_word": {str(k): v for k, v in id_to_word.items()},
            "T": T, "D": D, "V": V,
        }, f)
    print(f"Saved {CKPT_PATH} and {META_PATH}")