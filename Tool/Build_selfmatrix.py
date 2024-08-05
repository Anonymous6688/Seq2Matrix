from SRModel.Block_SSC import Block_SSC
from SRModel.LRR_model import demo
import pandas as pd
import numpy as np
from sympy import *
import math
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from scipy import linalg
import statsmodels.api as sm

np.random.random(6)
lam = [0.5, 1, 5, 10, 20, 60, 100, 200];
gam = [0.8, 1, 5, 10, 20, 60, 100, 200];


def find_lrr(data):
    [B, E] = demo(np.array(data.values.tolist()))

    W = np.maximum(0, (B + B.T) / 2)
    W = W - np.diag(np.diag(B))
    D = np.diag(np.sum(W, axis=1))
    I = np.eye(W.shape[1])
    # L = D-B
    D_half = D ** (-0.5)
    whereinf = np.isinf(D_half)
    D_half[whereinf] = 0
    L_norm = I - D_half @ W @ D_half


    eigenvals, eigenvcts = linalg.eig(L_norm)
    eigenvals = np.real(eigenvals)
    eigenvals_sorted_indices = np.argsort(eigenvals)
    eigenvals_sorted = eigenvals[eigenvals_sorted_indices]
    i = 0
    while i < len(eigenvals_sorted) - 2:
        diff1 = np.abs(math.exp(eigenvals_sorted[i + 1] - eigenvals_sorted[i]))
        diff2 = np.abs(math.exp(eigenvals_sorted[i + 2] - eigenvals_sorted[i + 1]))
        diff = np.abs(diff1 - diff2)
        i += 1
        if diff > 0.3:
            break

    return [B, i + 1]


def EuclideanDistances(A, B):
    A = np.array(A)
    B = np.array(B)
    BT = B.transpose()
    vecProd = A * BT
    SqA = A.getA() ** 2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    SqB = B.getA() ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    ED = (SqED.getA()) ** 0.5
    return np.matrix(ED)


def find_edu(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    D2 = np.sum(X * X, axis=1, keepdims=True) + np.sum(Y * Y, axis=1, keepdims=True).T - 2 * np.dot(X, Y.T)
    D2 = np.sqrt(D2)
    wherenan = np.isnan(D2)
    D2[wherenan] = 0
    return D2


def find_kernel(X, Y, sigma=0.1):
    X = np.array(X)
    Y = np.array(Y)
    D2 = np.sum(X * X, axis=1, keepdims=True) + np.sum(Y * Y, axis=1, keepdims=True).T - 2 * np.dot(X, Y.T)
    sigma = np.max(D2)
    Kernel = np.exp(-D2 / (70 * sigma ** 2))

    clustering = AgglomerativeClustering(n_clusters=None, linkage='ward', compute_full_tree=True,
                                         distance_threshold=0.3).fit(X)
    cluster = np.unique(clustering.labels_)
    return [Kernel, len(cluster)]


def l1_nslove(p_test):
    x = symbols('x')
    f1 = 2 * x - x ** (np.shape(p_test)[0] + 1) - 1
    l1 = nsolve(f1, x, 0.5)
    return l1


def find_KBDR(data, win_size, lam, gam):
    [B, regime_number] = find_lrr(data)
    #print('regime_number=' + str(regime_number))
    B = np.maximum(0, (B + B.T) / 2)
    B = B - np.diag(np.diag(B))
    degree = np.diag(np.sum(B, axis=1))
    D_half = degree ** (-0.5)
    whereinf = np.isinf(D_half)
    D_half[whereinf] = 0

    B = D_half @ B @ D_half

    B = B + np.diag(np.sum(B, axis=1)) + 0.00001 * np.eye(B.shape[0])
    [Z, Z1] = Block_SSC(B, B.shape[1],
                        win_size, lam[5],
                        gam[0]).KBDR_solver(K=4)
    return [Z, Z1, regime_number, D_half]


def stock_find_KBDR(data, win_size, lam, gam):
    [B, regime_number] = find_kernel(data.T, data.T)
    print('regime_number=' + str(regime_number))


    [Z, Z1] = Block_SSC(B, B.shape[1],
                        win_size, lam[5],
                        gam[0]).KBDR_solver(K=regime_number)
    return [Z, Z1, regime_number]


def find_BDR(data, win_size, lam, gam):
    [B, regime_number] = find_lrr(data)
    print('regime_number=' + str(regime_number))
    [Z, Z1] = Block_SSC(np.array(data.values.tolist()), data.shape[1],
                        win_size, lam[5],
                        gam[0]).BDR_solver(K=regime_number)
    return [Z, Z1, regime_number]


def Multi_matrix(data, win_size, win_number, lam, gam):
    Z = []
    for i in range(win_number):
        [globals()['Z%s' % (i + 1)], Z_notuse] = find_KBDR(data.iloc[i * win_size:(i + 1) * win_size, :], win_size, lam,
                                                           gam)
        Z.append(globals()['Z%s' % (i + 1)])
    return Z


def KSRM(data, indices, win_size):
    Z = []
    regime_num = []
    D = []
    # if parallel:
    #     executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
    #     tasks = (delayed(find_KBDR)(pd.DataFrame(data[i]), win_size, lam, gam) for i in tqdm(range(len(indices))))
    #     [Z, Z_notuse, number] = executor(tasks)
    for i in tqdm(range(len(indices))):
        [globals()['Z%s' % (i + 1)], Z_notuse, number, Degree] = find_KBDR(pd.DataFrame(data[i]), win_size, lam,
                                                                           gam)
        if np.all(globals()['Z%s' % (i + 1)] == 0):
            Z.append(Z_notuse)
        else:
            Z.append(globals()['Z%s' % (i + 1)])

        regime_num.append(number)
        D.append(Degree)

    return [Z, regime_num, D]


def stockKSRM(data, indices, win_size):
    Z = []
    regime_num = []

    for i in tqdm(range(len(indices)-1)):
        [globals()['Z%s' % (i + 1)], Z_notuse, number] = stock_find_KBDR(pd.DataFrame(data[i]), win_size, lam,
                                                                         gam)
        if np.all(globals()['Z%s' % (i + 1)] == 0):
            Z.append(Z_notuse)
        else:
            Z.append(globals()['Z%s' % (i + 1)])

        regime_num.append(number)
        # Label.append(label)
    return [Z, regime_num]


def KSRM2(data, indices, win_size):
    Z = []
    regime_num = []
    D = []
    # if parallel:
    #     executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
    #     tasks = (delayed(find_KBDR)(pd.DataFrame(data[i]), win_size, lam, gam) for i in tqdm(range(len(indices))))
    #     [Z, Z_notuse, number] = executor(tasks)
    for i in tqdm(range(len(indices))):
        [globals()['Z%s' % (i + 1)], Z1, number] = find_BDR(pd.DataFrame(data[i]), win_size, lam, gam)

        Z.append(globals()['Z%s' % (i + 1)])

        regime_num.append(number)

    return [Z, regime_num]


def mar(X, pred_step, maxiter=100):
    T, m, n = X.shape
    B = np.random.randn(n, n)
    for it in range(maxiter):
        temp0 = B.T @ B
        temp1 = np.zeros((m, m))
        temp2 = np.zeros((m, m))
        for t in range(1, T):
            temp1 += X[t, :, :] @ B @ X[t - 1, :, :].T
            temp2 += X[t - 1, :, :] @ temp0 @ X[t - 1, :, :].T
        try:
            A = temp1 @ np.linalg.inv(temp2)
        except:
            A = temp1 @ np.linalg.pinv(temp2)
        else:
            A = A

        temp0 = A.T @ A
        temp1 = np.zeros((n, n))
        temp2 = np.zeros((n, n))
        for t in range(1, T):
            temp1 += X[t, :, :].T @ A @ X[t - 1, :, :]
            temp2 += X[t - 1, :, :].T @ temp0 @ X[t - 1, :, :]
        # B = temp1 @ np.linalg.inv(temp2)
        try:
            B = temp1 @ np.linalg.inv(temp2)
        except:
            B = temp1 @ np.linalg.pinv(temp2)
        else:
            B = B
    tensor = np.append(X, np.zeros((pred_step, m, n)), axis=0)
    for s in range(pred_step):
        tensor[T + s, :, :] = A @ tensor[T + s - 1, :, :] @ B.T
    return tensor[- pred_step:, :, :]


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def regression_predict(Z, split_series, window_size):
    x_train = [0 for x in range(0, np.shape(Z)[1] * np.shape(Z)[1])]
    x_predict = [0 for x in range(0, np.shape(Z)[1] * np.shape(Z)[1])]
    data_series = [0 for x in range(0, window_size * np.shape(Z)[1])]
    # D_pre = [0 for x in range(0, np.shape(D)[1] * np.shape(D)[1])]
    for j in range(np.shape(Z)[0] - 5, np.shape(Z)[0] - 2):
        x_train = np.column_stack((x_train, np.array(Z[j]).reshape(-1, 1)))
        x_predict = np.column_stack((x_predict, np.array(Z[j + 1]).reshape(-1, 1)))
        data_series = np.column_stack((data_series, np.array(split_series[j + 1]).reshape(-1, 1)))
        # D_pre = np.column_stack((D_pre, np.array(D[j + 1]).reshape(-1, 1)))
    x_train = pd.DataFrame(x_train)
    x_predict = pd.DataFrame(x_predict)
    data_series = pd.DataFrame(data_series)
    # D_pre = pd.DataFrame(D_pre)
    x_train = x_train.drop(x_train.iloc[:, 0], axis=1).T.reset_index(drop=True).T
    x_predict = x_predict.drop(x_predict.iloc[:, 0], axis=1).T.reset_index(drop=True).T
    data_series = data_series.drop(data_series.iloc[:, 0], axis=1).T.reset_index(drop=True).T


    y = pd.DataFrame(np.array(Z[np.shape(Z)[0] - 2]).reshape(-1, 1))

    model = sm.OLS(y, x_train)
    result = model.fit()


    Z_predict = result.predict(np.array(x_predict))
    X_predict = result.predict(np.array(data_series))
    # D_predict = result.predict(D_pre)
    Z_predict = Z_predict.reshape(np.shape(Z)[1], np.shape(Z)[1])
    X_predict = X_predict.reshape(window_size, np.shape(Z)[1])
    # D_predict = D_predict.values.reshape(np.shape(D)[1], np.shape(D)[1])
    # D_predict = np.linalg.inv(D_predict)
    Forecast_matrix = X_predict @ Z_predict
    # Forecast_matrix = X_predict @ (D_predict @ Z_predict @ D_predict)

    return [Forecast_matrix, Z_predict]


def stock_regression_predict(Z, split_series, window_size):
    x_train = [0 for x in range(0, np.shape(Z)[1] * np.shape(Z)[1])]
    x_predict = [0 for x in range(0, np.shape(Z)[1] * np.shape(Z)[1])]
    # D_pre = [0 for x in range(0, np.shape(D)[1] * np.shape(D)[1])]
    for j in range(np.shape(Z)[0] - 5, np.shape(Z)[0] - 2):
        x_train = np.column_stack((x_train, np.array(Z[j]).reshape(-1, 1)))
        x_predict = np.column_stack((x_predict, np.array(Z[j + 1]).reshape(-1, 1)))
        # D_pre = np.column_stack((D_pre, np.array(D[j + 1]).reshape(-1, 1)))
    x_train = pd.DataFrame(x_train)
    x_predict = pd.DataFrame(x_predict)
    # D_pre = pd.DataFrame(D_pre)
    x_train = x_train.drop(x_train.iloc[:, 0], axis=1).T.reset_index(drop=True).T
    x_predict = x_predict.drop(x_predict.iloc[:, 0], axis=1).T.reset_index(drop=True).T



    y = pd.DataFrame(np.array(Z[np.shape(Z)[0] - 2]).reshape(-1, 1))

    model = sm.OLS(y, x_train)
    result = model.fit()

    Z_predict = result.predict(np.array(x_predict))

    # D_predict = result.predict(D_pre)
    Z_predict = Z_predict.reshape(np.shape(Z)[1], np.shape(Z)[1])


    return [Z_predict]
