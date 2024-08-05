import numpy as np
import json
from tqdm import tqdm
from PyPDF2 import PdfFileMerger
from Tool.load_data import data
import Tool.generate_data_Multi_windows as gen
from more_itertools import locate
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
import pandas as pd
import Tool.Build_selfmatrix as build
import os
import Tool.plotting_exper as plte
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import warnings

warnings.filterwarnings('ignore')
np.random.random(6)

file_name = 'Stock2.csv'
root = '../Seq2Matrix/'

path_to_csv = root + 'data/'
subdir = file_name.split('.')[0].upper()
subdir1 = file_name.split('.')[0]
if subdir + '_parameters' not in os.listdir(root + 'results/'):
    os.makedirs(root + 'results/' + subdir + '_parameters/')
if 'stock2' not in os.listdir(root + 'results/'):
    os.makedirs(root + 'results/' + 'stock2' + '/')
if 'SP500_predict_images' not in os.listdir(root + 'results/stock2/'):
    os.makedirs(root + 'results/stock2/' + 'SP500_predict_images' + '/')
if 'All_window' not in os.listdir(root + 'results/stock2/SP500_predict_images/'):
    os.makedirs(root + 'results/stock2/SP500_predict_images/' + 'All_window' + '/')
if 'SP500_RS_images' not in os.listdir(root + 'results/stock2/'):
    os.makedirs(root + 'results/stock2/' + 'SP500_RS_images' + '/')
if 'Regime_images' not in os.listdir(root + 'results/stock2/'):
    os.makedirs(root + 'results/stock2/' + 'Regime_images' + '/')
if 'Merge_predict' not in os.listdir(root + 'results/stock2/'):
    os.makedirs(root + 'results/stock2/' + 'Merge_predict' + '/')
if 'Merge_RS' not in os.listdir(root + 'results/stock2/'):
    os.makedirs(root + 'results/stock2/' + 'Merge_RS' + '/')
path_to_json = root + 'results/' + subdir + '_parameters/'
if 'Slide_info' not in os.listdir(path_to_json):
    os.makedirs(path_to_json + 'Slide_info' + '/')
slideinfo_path_to_json = path_to_json + 'Slide_info' + '/'


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
    series = data(path_to_csv, file_name).stock_download()
    window_size = 11
    History_size = window_size * 4
    start_slide = History_size
    end_slide = len(series)

    Pre_all_win = pd.DataFrame()
    RMSE_all = []
    for slide_point in tqdm(range(start_slide, end_slide, window_size)):
        if 'Slide_point_' + str(slide_point - start_slide + 1) not in os.listdir(
                root + 'results/stock2/SP500_predict_images/'):
            os.makedirs(
                root + 'results/stock2/SP500_predict_images/' + 'Slide_point_' + str(
                    slide_point - start_slide + 1) + '/')
        if 'Slide_point_' + str(slide_point - start_slide + 1) not in os.listdir(
                root + 'results/stock2/SP500_RS_images/'):
            os.makedirs(
                root + 'results/stock2/SP500_RS_images/' + 'Slide_point_' + str(
                    slide_point - start_slide + 1) + '/')

        data = series.iloc[slide_point - History_size:slide_point + window_size, :]
        data = data.reset_index(drop=True)
        data = data.T.reset_index(drop=True).T

        print('\n processing_slide' + str(slide_point - start_slide + 1))
        data_KSRM = {}

        data_Degree = {}
        indices = range(1 * window_size, (len(data) // window_size + 1) * window_size, window_size)
        split_series = np.split(data, indices)
        slide_info = {}

        if 'slide_' + str(slide_point - start_slide + 1) + 'info' + '.json' not in os.listdir(slideinfo_path_to_json):

            if 'data_KSRM_slide_' + str(slide_point - start_slide + 1) + '.json' not in os.listdir(path_to_json):
                indices = range(1 * window_size, (len(data) // window_size + 1) * window_size, window_size)
                split_series = np.split(data, indices)
                [Z, var, regime_number] = gen.stock_get_timeseries_variation(split_series, indices, window_size)

                [Regime_center, Number_S, Label] = gen.distinct_regime(Z, regime_number, split_series)


                data_KSRM[str(window_size)] = Z
                data_KSRM['regime_number'] = regime_number
                data_KSRM['Regime_center_window'] = np.array(Regime_center)
                data_KSRM['Number_S'] = np.array(Number_S)
                data_KSRM['Label'] = Label


                with open(path_to_json + 'data_KSRM_slide_' + str(slide_point - start_slide + 1) + '.json',
                          'w') as outputfile:
                    json.dump(data_KSRM, outputfile, cls=NpEncoder, indent=3)

            else:
                with open(path_to_json + 'data_KSRM_slide_' + str(slide_point - start_slide + 1) + '.json',
                          'r') as inputfile:
                    data_KSRM = json.load(inputfile)

            Z = data_KSRM.get(str(window_size))
            regime_number = data_KSRM.get('regime_number')
            Regime_center = pd.DataFrame(data_KSRM.get('Regime_center_window'))
            Number_S = pd.DataFrame(data_KSRM.get('Number_S'))
            Label = pd.DataFrame(data_KSRM.get('Label'))

            Number_S = Number_S.reset_index(drop=True)
            [Ker_center_matrix, _] = build.find_kernel(Regime_center.T, Regime_center.T)
            Edu_matrix = build.find_edu(Regime_center.T, Regime_center.T)
            clustering = AgglomerativeClustering(n_clusters=None, linkage='ward', compute_full_tree=True,
                                                 distance_threshold=0.03).fit(Regime_center.T)
            cluster = np.unique(clustering.labels_)
            label = clustering.labels_
            R = pd.DataFrame()
            for j in range(len(cluster)):
                locals()['index_pos_list%s' % (j + 1)] = list(
                    locate(clustering.labels_.tolist(), lambda a: a == cluster[j]))
                if R.empty:
                    R = pd.DataFrame(pd.DataFrame(Regime_center)[locals()['index_pos_list%s' % (j + 1)]].mean(axis=1)).T
                else:

                    R = pd.concat(
                        [R,
                         pd.DataFrame(
                             pd.DataFrame(Regime_center)[locals()['index_pos_list%s' % (j + 1)]].mean(axis=1)).T])

            plte.save_r(R, cluster, slide_point, root, start_slide)

            Label_trans = np.array(Label)
            for i in range(np.shape(Label)[0]):
                for j in range(np.shape(Label)[1]):
                    Label_trans[i, j] = label[Label.iloc[i, j]]

            Lambda = []
            for i in range(len(Z) - 2 - 1):
                Lambda_inside = np.zeros((len(R), len(R)))
                for j in range(len(R)):
                    for k in range(len(R)):
                        if np.maximum(
                                Counter(Label_trans[i, :])[j], Counter(Label_trans[i + 1, :])[k]) == 0:
                            Lambda_inside[j, k] = 0
                        else:
                            Lambda_inside[j, k] = np.minimum(Counter(Label_trans[i, :])[j],
                                                             Counter(Label_trans[i + 1, :])[k]) / np.maximum(
                                Counter(Label_trans[i, :])[j], Counter(Label_trans[i + 1, :])[k])
                Lambda.append(Lambda_inside)
            Lambda_b1 = np.zeros((len(R), len(R)))
            for i in range(len(Lambda)):
                Lambda_b1 = Lambda_b1 + Lambda[i]
            # solve Theta for each series
            Theta = {}
            P = {}


            def CountOccurrences(string, substring):

                count = 0
                start = 0

                while start < len(string):

                    flag = string.find(substring, start)

                    if flag != -1:

                        start = flag + 1

                        count += 1
                    else:

                        return count


            P_predict_all = []
            P_value_all = pd.DataFrame()
            r_next_all = []
            for i in range(np.shape(data)[1]):
                Theta['data' + str(i + 1)] = []
                P['data' + str(i + 1)] = []

                P_inside = np.zeros((len(R), len(R)))
                theta_inside = np.zeros((len(R), len(R)))
                for m in range(len(R)):
                    for n in range(len(R)):
                        str_ = ''.join('%s' % id for id in Label_trans[0:len(Z) - 1 - 1, i].tolist())
                        num_mn = CountOccurrences(str_, str(m) + str(n))
                        theta_inside[m, n] = num_mn / (len(Z) - 1 - 1)

                for m in range(len(R)):
                    for n in range(len(R)):

                        sum_mn = np.sum(np.multiply(Lambda_b1[:, n], theta_inside[m, :]))

                        if np.sum(np.multiply(Lambda_b1, theta_inside)) == 0:
                            P_inside[m, n] = 0
                        else:

                            P_inside[m, n] = sum_mn

                P_inside = P_inside / np.sum(P_inside)

                Theta['data' + str(i + 1)].append(theta_inside)
                P['data' + str(i + 1)].append(P_inside)

                # predict_value
                myarray = np.random.randint(0, 2, np.shape(data)[1])

                r = Label_trans[-2, i]
                if np.all(P_inside[r, :] == 0):
                    r_next = np.argmax(np.bincount(Label_trans[:, i]))
                else:
                    r_next = (P_inside[r, :].tolist()).index(max(P_inside[r, :].tolist()))
                r_next_all.append(r_next)

                index = [i2 for i2, x in enumerate(Label_trans[0:len(Z) - 1 - 1, i]) if x == r_next]
                p_test = pd.DataFrame()
                for im in range(len(index)):
                    if p_test.empty:
                        p_test = pd.DataFrame(split_series[index[im]].iloc[:, i]).reset_index(drop=True).T
                    else:
                        p_split = pd.DataFrame(split_series[index[im]].iloc[:, i]).reset_index(drop=True).T
                        p_test = pd.concat([p_test, p_split])

                l1 = build.l1_nslove(p_test)

                lam_test = []

                for num_ptest in range(np.shape(p_test)[0]):
                    lam_test.append(l1 ** (np.shape(p_test)[0] - num_ptest))

                lam_test = np.array(lam_test).reshape(-1, 1)
                p_pre = (p_test.T @ lam_test)

                rmse = sqrt(mean_squared_error(split_series[len(indices) - 1].iloc[:, i], p_pre))
                Stock_name_list = series.columns.values.tolist()
                stock_name = Stock_name_list[i]
                index = series.columns.get_loc(stock_name)
                target_path = root + 'results/stock2/SP500_predict_images/' + 'Slide_point_' + str(
                    slide_point - start_slide + 1)
                pdf_lst = [f for f in os.listdir(target_path) if f.endswith('.pdf')]
                pdf_lst = [os.path.join(target_path, filename) for filename in pdf_lst]
                if len(pdf_lst) != np.shape(data)[1]:
                    plte.Save_SP500(split_series, indices, stock_name, index, p_pre, rmse, r_next, Label_trans, root,
                                    slide_point, start_slide)

                if P_value_all.empty:
                    P_value_all = pd.DataFrame(p_pre).T
                else:
                    P_value_all = pd.concat([P_value_all, pd.DataFrame(p_pre).T])
            P_value_all = P_value_all.reset_index(drop=True)
            slide_info['Predict_value_last_window'] = np.float32(P_value_all)
            slide_info['True_value_last_window'] = np.array(split_series[len(indices) - 1].reset_index(drop=True))
            with open(slideinfo_path_to_json + 'slide_' + str(slide_point - start_slide + 1) + 'info' + '.json',
                      'w') as outputfile:
                json.dump(slide_info, outputfile, cls=NpEncoder, indent=3)
        else:
            with open(slideinfo_path_to_json + 'slide_' + str(slide_point - start_slide + 1) + 'info' + '.json',
                      'r') as inputfile:
                slide_info = json.load(inputfile)

        P_value_all = pd.DataFrame(slide_info.get('Predict_value_last_window'))
        true_value = pd.DataFrame(slide_info.get('True_value_last_window'))
        if Pre_all_win.empty:
            Pre_all_win = P_value_all.T
        else:
            Pre_all_win = pd.concat([Pre_all_win, P_value_all.T])
        RMSE = []
        for i in range(np.shape(data)[1]):
            # rmse = sqrt(mean_squared_error(split_series[len(indices) - 1].iloc[:, i], P_value_all.iloc[i, :]))
            rmse = sqrt(mean_squared_error(true_value.iloc[:, i], P_value_all.iloc[i, :]))
            RMSE.append(rmse)
        print('\n' + str(subdir) + '_slide_' + str(slide_point - start_slide + 1) + '_RMSE=' + str(
            np.array(RMSE).sum() / np.shape(data)[1]))

        RMSE_all.append(np.array(RMSE).sum() / np.shape(data)[1])

        # Merge predict PDF
        target_path = root + 'results/stock2/SP500_predict_images/' + 'Slide_point_' + str(
            slide_point - start_slide + 1)
        pdf_lst = [f for f in os.listdir(target_path) if f.endswith('.pdf')]
        pdf_lst = [os.path.join(target_path, filename) for filename in pdf_lst]

        file_merger = PdfFileMerger()
        for pdf in pdf_lst:
            file_merger.append(pdf)

        file_merger.write(
            root + 'results/stock2/' + 'Merge_predict/Merge_SP500_predict_Slide' + str(
                slide_point - start_slide + 1) + '.pdf')
        file_merger.close()

        # Merge PDF
        target_path = root + 'results/stock2/SP500_RS_images/' + 'Slide_point_' + str(slide_point - start_slide + 1)
        pdf_lst = [f for f in os.listdir(target_path) if f.endswith('.pdf')]
        pdf_lst = [os.path.join(target_path, filename) for filename in pdf_lst]

        file_merger = PdfFileMerger()
        for pdf in pdf_lst:
            file_merger.append(pdf)

        file_merger.write(
            root + 'results/stock2/' + 'Merge_RS/Merge_SP500_RS_Slide' + str(slide_point - start_slide + 1) + '.pdf')
        file_merger.close()

    Pre_all_win = Pre_all_win.reset_index(drop=True)
    True_all_win = series.iloc[44:143, :].reset_index(drop=True)
    True_all_win = True_all_win.T.reset_index(drop=True).T
    plte.merge_allpre(Pre_all_win, True_all_win, series, root)
    print('RMSE_mean = ' + str(
        np.array(RMSE_all).sum() / len(RMSE_all)))
    print('time=', time.time() - start)
