

import numpy as np
import json
from Tool.load_data import data
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import zscore
import time
import Tool.plotting as xplt
import pandas as pd
import Tool.generate_data_Multi_windows as gen
import Tool.Build_selfmatrix as build
import os
from Tool.Multiscale_division import Multi_division
from Tool.window_search import Winsearch
import warnings
warnings.filterwarnings('ignore')
np.random.random(6)

lam = [0.5, 1, 5, 10, 20, 60, 100, 200]
gam = [0.8, 1, 5, 10, 20, 60, 100, 200]

p = 1; #p=1, plot all the result figures
file_name = 'Synthetic.csv'
root = '../Seq2Matrix/'
path_to_csv = root + 'data/'
subdir = file_name.split('.')[0]
path_to_json = root + 'results/' + subdir + '/'


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

if __name__ == '__main__':
    start = time.time()
    print('Loading data ...')
    series = data(path_to_csv, file_name).download()
    multi_division = Multi_division(data=series,name=subdir,root=root)
    data = multi_division.division()
    Forecast = []
    small_scale = Winsearch(data[len(data) - 1], len(data[len(data) - 1]), root=root, file_name=file_name)
    [window_size, small_scale_KSRM] = small_scale.win_number(str(0))  # str(len(data)-2) and str(0)
    for i in range(len(data)):
        print('\n processing scale' + str(len(data) - i - 1))
        data_KSRM = {}
        data_Degree = {}
        if 'data_KSRM_scale_' + str(len(data) - i - 1) + '.json' not in os.listdir(path_to_json):
            indices = range(1 * window_size, (len(data[i]) // window_size + 1) * window_size, window_size)
            split_series = np.split(data[i], indices)
            [Z, var, D] = gen.get_timeseries_variation(split_series, indices, window_size)
            data_KSRM[str(window_size)] = Z
            with open(path_to_json + 'data_KSRM_scale_' + str(len(data) - i - 1) + '.json', 'w') as outputfile:
                json.dump(data_KSRM, outputfile, cls=NpEncoder, indent=3)
        else:
            with open(path_to_json + 'data_KSRM_scale_' + str(len(data) - i - 1) + '.json', 'r') as inputfile:
                data_KSRM = json.load(inputfile)
        Z = data_KSRM.get(str(window_size))
        indices = range(1 * window_size, (len(data[i]) // window_size + 1) * window_size, window_size)
        split_series = np.split(data[i], indices)
        [Forecast_matrix, Z_predict] = build.regression_predict(Z, split_series, window_size)
        Forecast.append(Forecast_matrix)
    Forecast_global_matrix = np.zeros((window_size, np.shape(Z)[1]))
    for i in range(np.shape(Forecast)[0]):
        Forecast_global_matrix = Forecast_global_matrix + Forecast[i]
    RMSE = []
    a = pd.DataFrame(Forecast_global_matrix).apply(zscore)
    indices_global = range(1 * window_size, (len(series) // window_size + 1) * window_size, window_size)
    split_series_global = np.split(np.array(series), indices_global)
    b = pd.DataFrame(split_series_global[len(indices_global) - 1]).apply(zscore)
    wherenan = np.isnan(b)
    b[wherenan]=0
    if p == 1:
        xplt.concept(split_series_global, Z, root, subdir)
        xplt.Z_matrix(Z, root, subdir)
        xplt.ablation(series, window_size, lam, gam, root, subdir)
        xplt.predict(a,b,root,subdir)
    for i in range(np.shape(a)[1]):
        rmse = sqrt(mean_squared_error(a.iloc[:, [i]], b.iloc[:, [i]]))
        RMSE.append(rmse)
    print(str(subdir) + '_RMSE=' + str(np.array(RMSE).sum() / np.shape(a)[1]))

    print('time=', time.time() - start)


