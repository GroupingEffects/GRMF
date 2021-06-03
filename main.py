import numpy as np
import pandas as pd
import time
import multiprocessing

from utils import evaluate, group_evaluation, sparsity_evaluation, add_outlier
from data_loader import load_main_datasets
from GRMF import multiprocess_GRMF, GRMF, GRMF_L2
from benchmark_algorithm import RPMF, GoDec_plus, truncated_svd, RPCA, robust_nmf
from settings import default_param_A

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
num_cores = int(multiprocessing.cpu_count())
method_names = ['GRMF', 'RPMF', 'GRMF_L2', 'GoDec_plus', 'truncated_svd', 'RPCA', 'rNMF']


# call all the algorithm
def experiment(data, rank, origin_data):
    # type:(np.array, int, np.array)->(dict[float], dict[float], dict[float], dict[float])
    """
    Factorization using 7 algorithms
    :param data: the input data
    :param rank: the rank chosen for every algorithm
    :param origin_data: the original data (compare to the corrupted data)
    :return: the error, time, groups and sparsity for every algorithm
    """
    default_tau = default_param_A['tau']  # the tau used for count groups
    # dictionaries to collect the information
    algo_loss = dict()
    algo_time = dict()
    algo_group = dict()
    algo_sparsity = dict()

    # algorithm 1 GRMF L1 approximate version fastest
    start = time.time()
    # use a multiprocessing version if the rank is bigger than 4
    if rank < 5:
        U, V, _ = GRMF(data, rank, 0, max_iter=10, param=default_param_A)
    else:
        U, V, _ = multiprocess_GRMF(data, rank, 0, max_iter=10, param=default_param_A)
    end = time.time()
    algo_loss['GRMF'] = evaluate(origin_data, U@V.T)
    algo_time['GRMF'] = end - start
    algo_group['GRMF'] = group_evaluation(U, V, default_tau)
    algo_sparsity['GRMF'] = sparsity_evaluation(U, V)

    # algorithm 2 L2 version
    start = time.time()
    if rank < 10:
        U, V, _ = GRMF_L2(data, rank, 0, max_iter=10, param=default_param_A)
    else:
        U, V, _ = multiprocess_GRMF(data, rank, 0, max_iter=10, param=default_param_A, version='L2')
    end = time.time()
    algo_loss['GRMF_L2'] = evaluate(origin_data, U@V.T)
    algo_time['GRMF_L2'] = end - start
    algo_group['GRMF_L2'] = group_evaluation(U, V, default_tau)
    algo_sparsity['GRMF_L2'] = sparsity_evaluation(U, V)

    # algorithm 3 RPMF
    start = time.time()
    U, V = RPMF(data, rank, 0.01, 0.01, 1e-4)
    end = time.time()
    algo_loss['RPMF'] = evaluate(origin_data, U@V.T)
    algo_time['RPMF'] = end - start
    algo_group['RPMF'] = group_evaluation(U, V, default_tau)
    algo_sparsity['RPMF'] = sparsity_evaluation(U, V)

    # algorithm 4 GoDec+
    start = time.time()
    U, V = GoDec_plus(data, rank, 5, 1e-7, 5)
    end = time.time()
    algo_loss['GoDec_plus'] = evaluate(origin_data, U@V.T)
    algo_time['GoDec_plus'] = end - start
    algo_group['GoDec_plus'] = group_evaluation(U, V, default_tau)
    algo_sparsity['GoDec_plus'] = sparsity_evaluation(U, V)

    # algorithm 5 Truncated SVD
    start = time.time()
    U, V = truncated_svd(data, rank)
    end = time.time()
    algo_loss['truncated_svd'] = evaluate(origin_data, U@V.T)
    algo_time['truncated_svd'] = end - start
    algo_group['truncated_svd'] = group_evaluation(U, V, default_tau)
    algo_sparsity['truncated_svd'] = sparsity_evaluation(U, V)

    # algorithm 6 RPCA
    start = time.time()
    U, V = RPCA(data, rank)
    end = time.time()
    algo_loss['RPCA'] = evaluate(origin_data, U)
    algo_time['RPCA'] = end - start
    algo_group['RPCA'] = group_evaluation(U, U, default_tau)
    algo_sparsity['RPCA'] = sparsity_evaluation(U, U)

    # algorithm 7 robust NMF
    start = time.time()
    U, V, _, _ = robust_nmf(data, rank)
    end = time.time()
    algo_loss['rNMF'] = evaluate(origin_data, U@V.T)
    algo_time['rNMF'] = end - start
    algo_group['rNMF'] = group_evaluation(U, V, default_tau)
    algo_sparsity['rNMF'] = sparsity_evaluation(U, V)
    return algo_loss, algo_time, algo_group, algo_sparsity


# generate a dataframe to write
def generate_table(method_names, size):  # type:(list[str], int) -> pd.DataFrame
    """
    Generate a table to collect the MF information for each method
    Each row represents the information of factorizing certain picture using certain algorithm
    :param method_names:
    :param size: the size of the datasets
    :return:
    """
    columns_names = ['RMAE', 'RMSE', 'RE', 'Time', 'Groups_mean', 'Groups_std', 'Sparsity', 'MAE', 'MSE']
    algo_names = method_names * size
    algo_names.sort()
    pic_list = [i for i in range(size)] * len(algo_names)
    tuples = list(zip(algo_names, pic_list))
    idx = pd.MultiIndex.from_tuples(tuples, names=['Algorithm', 'Picture id'])
    df = pd.DataFrame(np.zeros((len(method_names)*size, len(columns_names))),
                      columns=[columns_names])
    df = df.set_index(idx)
    return df


# write the result
def results_writer(df, num_pic, algo_loss, algo_time, algo_groups, algo_sparsity):
    """
    Write the information to the table
    :param df: the given table
    :param num_pic: the number of the given picture
    :param algo_loss:
    :param algo_time:
    :param algo_groups:
    :param algo_sparsity:
    :return:
    """
    for k, v in algo_loss.items():
        df.loc[(k, num_pic), 'RMAE'] = v[0]
        df.loc[(k, num_pic), 'RMSE'] = v[1]
        df.loc[(k, num_pic), 'RE'] = v[2]
        df.loc[(k, num_pic), 'MAE'] = v[3]
        df.loc[(k, num_pic), 'MSE'] = v[4]
    for k, v in algo_time.items():
        df.loc[(k, num_pic), 'Time'] = v
    for k, v in algo_groups.items():
        df.loc[(k, num_pic), 'Groups_mean'] = v[0]
        df.loc[(k, num_pic), 'Groups_std'] = v[1]
    for k, v in algo_sparsity.items():
        df.loc[(k, num_pic), 'Sparsity'] = v
    return df


# main
def main(datasets='ORL'):
    """

    :param datasets:
    :return:
    """
    # YaleB 192*168 with 210 pictures
    print('==========================Load the datasets:', datasets)
    if datasets == 'YaleB':
        origin_data = load_main_datasets()
        rank, size = 5, 210
        row, col = 192, 168
    # ORL origin, 200 pictures with 112*90
    elif datasets == 'ORL':
        origin_data = load_main_datasets(datasets='ORL')
        rank, size = 3, 200
        row, col = 112, 90
    # COIL-20 choose in every 5 picture, 288 pictures with 128*128
    elif datasets == 'COIL':
        origin_data = load_main_datasets(datasets='COIL')
        rank, size = 4, 206
        row, col = 128, 128
    elif datasets == 'toy':
        origin_data = np.random.randint(0, 255, (4, 8))
        rank, size = 2, 2
        row, col = 4, 4
    # JAFFE, 213 pictures with size of 180*150
    else:
        origin_data = load_main_datasets(datasets='JAFFE')
        rank, size = 4, 213
        row, col = 180, 150
    for outlier in [0.5, 0]:
        print("================Now for outlier [{}]================".format(outlier))
        data = add_outlier(origin_data, outlier_size=outlier)
        df = generate_table(method_names, size)
        for i in range(size):
            origin_pic = origin_data[:, i*col:(i+1)*col]
            pic = data[:, i*col:(i+1)*col]
            algo_loss, algo_time, algo_groups, algo_sparsity = \
                experiment(pic, rank, origin_pic)
            print('===Decompose picture [{}]==='.format(i))
            df = results_writer(df, i, algo_loss, algo_time, algo_groups, algo_sparsity)
        rank += 1
        name = datasets + 'results_' + str(outlier) + '.csv'
        df.to_csv(name)
    print('==================Fantastic! All the experiment done!=======================')


def print_table_info(value='RMAE', datasets='YaleB', outlier='0.5', decimal=3):
    """
    Load the experiment table and print the information
    :param value:
    :param datasets:
    :param outlier:
    :param decimal:
    :return:
    """
    # load
    name = datasets + 'results_' + outlier + '.csv'
    df = pd.read_csv(name, index_col=[0, 1], header=0).iloc[1:, :]
    value_mean, value_std = dict(), dict()
    for method_name in method_names:
        item = df.loc[(method_name,), value]
        value_mean[method_name] = np.mean(item)
        value_std[method_name] = np.std(item)
    print('-------------The mean of {} of the datasets ---[{}]--- under outlier [{}] is ----'
          '-----'.format(value, datasets, outlier))
    for k, v in value_mean.items():
        print('*', k, ':', round(v, decimal))
    print('-------------The std of {} of the datasets ---[{}]--- under outlier [{}] is ----'
          '-----'.format(value, datasets, outlier))
    for k, v in value_std.items():
        print('*', k, ':', round(v, decimal))
    print('-----------------------------------------------------------------------------\n')


def print_main_table_info():
    """
    Print the MF information for the for datasets under the two corruption ratios
    :return:
    """
    for dataset in ['YaleB', 'COIL', 'ORL', 'JAFFE']:
        for outlier in ['0', '0.5']:
            print_table_info(datasets=dataset, outlier=outlier)


if __name__ == "__main__":
    main('YaleB')
    main('COIL')
    main('ORL')
    main('JAFFE')
    print_main_table_info()