import numpy as np


def split_data(x, y, fraction_tr=0.5):
    n_samples = x.shape[0]
    idx = list(range(0, n_samples))  # [0 1 ... 999]  np.linspace
    np.random.shuffle(idx)
    n_tr = int(fraction_tr * n_samples)

    idx_tr = idx[:n_tr]
    idx_ts = idx[n_tr:]

    xtr = x[idx_tr, :]
    ytr = y[idx_tr]
    xts = x[idx_ts, :]
    yts = y[idx_ts]

    return xtr, ytr, xts, yts
