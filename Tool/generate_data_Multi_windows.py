import numpy as np
import pandas as pd
import random
import Tool.Build_selfmatrix as build
from scipy import stats
from sklearn.cluster import SpectralClustering
from more_itertools import locate


def split(n, i):
    # method 1 :uniform
    rnd_array = np.random.multinomial(n, np.ones(i) / i, size=1)[0]

    # method 2 : ununiform
    while True:
        pick = random.sample(range(1, n), i)
        if np.sum(pick) == n:
            break
    rnd_array = pick

    return rnd_array


def get_timeseries_variation(split_series, indices, window_size):
    [Z, regime_num, D] = build.KSRM(split_series, indices, window_size)
    variation = max(regime_num) / window_size
    return [Z, variation, D]


def stock_get_timeseries_variation(split_series, indices, window_size):
    [Z, regime_num] = build.stockKSRM(split_series, indices, window_size)
    variation = max(regime_num) / window_size
    return [Z, variation, regime_num]


def distinct_regime(Z, R_num, split_series_global):
    # R = []
    R = pd.DataFrame()
    Number_S = pd.DataFrame()
    Label = []
    for i in range(len(Z)):
        y_pred = SpectralClustering(n_clusters=R_num[i], affinity='precomputed').fit_predict(Z[i])
        cluster, num = np.unique(y_pred, return_counts=True)

        if Number_S.empty:
            Number_S = pd.DataFrame(num)
        else:
            Number_S = pd.concat([Number_S, pd.DataFrame(num)])
        Label.append(y_pred + sum(R_num[0:i]))
        split_series_global[i] = split_series_global[i].reset_index(drop=True)
        split_series_global[i] = ((split_series_global[i].T).reset_index(drop=True)).T
        for j in range(len(cluster)):
            locals()['index_pos_list%s' % (j + 1)] = list(locate(y_pred.tolist(), lambda a: a == cluster[j]))

            if R.empty:
                R = pd.DataFrame(
                    pd.DataFrame(split_series_global[i])[locals()['index_pos_list%s' % (j + 1)]].mean(axis=1)).T
            else:
                # R = pd.DataFrame(list(zip(pd.Series(R.values.tolist()), pd.DataFrame(split_series_global[i])[locals()['index_pos_list%s' % (j + 1)]].mean(axis=1))))
                R = pd.concat([R, pd.DataFrame(
                    pd.DataFrame(split_series_global[i])[locals()['index_pos_list%s' % (j + 1)]].mean(axis=1)).T])

    R = (R.T).reset_index(drop=True).T
    R = R.reset_index(drop=True)

    return [R.T, Number_S, Label]


def get_timeseries_variation2(split_series, indices, window_size):
    [Z, regime_num] = build.KSRM2(split_series, indices, window_size)

    return [Z]


def entropy_cut_off(scores):
    E = {}
    F = {}
    entropy_E = {}
    entropy_F = {}
    code = {}
    result = None
    for i in range(2, len(scores)):
        E_temp = scores[0:i]
        F_temp = scores[i:len(scores)]
        entropy_E[i - 1] = stats.entropy(E_temp) / len(E_temp)
        entropy_F[i] = stats.entropy(F_temp) / len(F_temp)
        E[i] = E_temp
        F[i] = F_temp
        code[i] = np.abs(entropy_E[i - 1] - entropy_F[i])
    minim = min(list(code.values()))
    for i in sorted(list(code.keys())):
        if code[i] == minim:
            result = (i, E[i], F[i], code[i])
            break
    return result[0] - 2

