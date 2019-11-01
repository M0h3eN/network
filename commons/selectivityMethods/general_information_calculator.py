import numpy as np
import pandas as pd
import zlib
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from commons.tools.basicFunctions import normalize, generate_lagged_epochs
from scipy.stats import pearsonr, wilcoxon
from pyvlmc.internals import vlmc


def ncs(x: np.ndarray, y: np.ndarray) -> float:
    """
    Evaluate The Normalized Compression Similarity NCS(x, y) = 1 - NCD(x, y)
    :param x: numpy one dimension array of one neuron spike count
    :param y: numpy one dimension array of one neuron spike count
    :return: normalized compression similarity
    """
    x_compressed = zlib.compress(x)
    y_compressed = zlib.compress(y)
    x_y_compress = zlib.compress(np.concatenate([x, y]))

    ncd = (len(x_y_compress) - min(len(x_compressed), len(y_compressed)))/max(len(x_compressed), len(y_compressed))
    ncs = 1 - ncd
    return ncs


# methods are:
# 1- pearsonr
# 2- mutual information (continuous) ---- mutual
# 3- mutual information score (discreet) --- mutualScore


def info(datax: pd.core.frame.DataFrame, datay: pd.core.frame.DataFrame, method: str = 'pearson') -> np.ndarray:
    """
    Evalute the neuronal information matrix
    :param datax: neurons data frame
    :param datay: possibly lagged neurons data frame
    :type method: Information method including: 1- pearson, 2- mutual information (continuous) ---- mutual,
    3- mutual information score (discreet) --- mutualScore and 4- normalized compression similarity ---- ncs.
    :return: information matrix
    """
    numeric_df_x = datax._get_numeric_data()
    numeric_df_y = datay._get_numeric_data()
    cols = numeric_df_x.columns
    idx = cols.copy()
    matx = numeric_df_x.values.T
    maty = numeric_df_y.values.T
    K = len(cols)
    correl = np.empty((K, K), dtype=float)

    if method == 'pearson':
        corrf = np.corrcoef
        for i, ac in enumerate(matx):
            for j, bc in enumerate(maty):
                if i > j:
                    continue
                elif i == j:
                    c = 0.
                else:
                    c = corrf(ac, bc)[0, 1]
                correl[i, j] = c
                correl[j, i] = c
    elif method == 'mutual':
        corrf = mutual_info_regression
        for i in range(K):
            for j in range(K):
                if i > j:
                    continue
                elif i == j:
                    c = 0.
                else:
                    c = corrf(matx[i, :].reshape(-1, 1), maty[j, :])[0]
                correl[i, j] = c
                correl[j, i] = c

    elif method == 'mutualScore':
        corrf = mutual_info_score
        for i in range(K):
            for j in range(K):
                if i > j:
                    continue
                elif i == j:
                    c = 0.
                else:
                    c = corrf(matx[i, :], maty[j, :])
                correl[i, j] = c
                correl[j, i] = c
    elif method == 'ncs':
        corrf = ncs
        for i in range(K):
            for j in range(K):
                if i > j:
                    continue
                elif i == j:
                    c = 0.
                else:
                    c = corrf(matx[i, :], maty[j, :])
                correl[i, j] = c
                correl[j, i] = c
    else:
        raise ValueError("method must be either 'pearson', "
                         "'mutual', 'ncs' or 'mutualScore', '{method}' "
                         "was supplied".format(method=method))

    # Search for nan values and replace it with zero
    nanIndexes = np.isnan(correl)
    correl[nanIndexes] = 0
    return np.array(correl, dtype=float)


# set threshold based on p_value matrix

def set_threshold(data, reference):

    K = data.shape[0]
    thresholded_mat = np.empty(data.shape, dtype=float)

    for i in range(K):
        for j in range(K):
            if reference[i, j] > 0:
                thresholded_mat[i, j] = data[i, j]
            else:
                thresholded_mat[i, j] = 0
    return thresholded_mat


def get_mean_over_trials(arr: np.ndarray, arr_laged: np.ndarray, method: str) -> np.ndarray:
    """
    This function evaluate information on each trial and take average over them
    :param arr: lag zero response matrix
    :param arr_laged: lagged response matrix
    :param method: information method
    :return: flatten array of information
    """
    shape_of_array = np.shape(arr)[1]
    if len(np.shape(arr)) > 2:
        info_array = np.array([np.array(
            info(pd.DataFrame(arr[:, i, :]).transpose(), pd.DataFrame(arr_laged[:, i, :]).transpose(), method)) for i in
                               range(shape_of_array)])
    else:
        info_array = info(pd.DataFrame(arr.transpose()), pd.DataFrame(arr_laged.transpose()), method)

    return info_array.mean(axis=0).flatten()


def complete_info_df(neurons_df: pd.core.frame.DataFrame, saccad_df: pd.core.frame.DataFrame, thresh: float,
                     epoch: str, method: str, lag: int, typ: str) -> pd.core.frame.DataFrame:
    """
    This function construct information detail data frame
    :param neurons_df: data set of all neurons
    :param saccad_df: data set of all saccade aligned neurons
    :param thresh: alpha for p-value computation
    :param epoch: 'Enc-In-NoStim', 'Mem-In-NoStim', 'Sac-In-NoStim',
                  'Enc-Out-NoStim', 'Mem-Out-NoStim', 'Sac-Out-NoStim',
                  'Enc-In-Stim', 'Mem-In-Stim', 'Sac-In-Stim',
                  'Enc-Out-Stim', 'Mem-Out-Stim', 'Sac-Out-Stim',
                  'Enc-In-Diff', 'Mem-In-Diff', 'Sac-In-Diff',
                  'Enc-Out-Diff', 'Mem-Out-Diff', 'Sac-Out-Diff'.
    :param method: information method -> correlation and mutual information
    :param lag: time lag negative and positive for epochs
    :param typ: response type -> evoked, count, all_trials
    :return: Data frame of assembly information
    """
    df = pd.DataFrame()

    df_zero = generate_lagged_epochs(neurons_df, saccad_df, epoch, 0, typ)
    df_lagged = generate_lagged_epochs(neurons_df, saccad_df, epoch, lag, typ)

    if typ == 'evoked':
        info_flat = info(pd.DataFrame(df_zero), pd.DataFrame(df_lagged), method).flatten()
    else:
        info_flat = get_mean_over_trials(df_zero, df_lagged, method)

    info_sem = np.std(info_flat) / np.sqrt(len(info_flat))
    info_mean = info_flat.mean()
    info_pvalue = wilcoxon(info_flat, correction=True)[1]

    if info_pvalue < thresh:
        info_color = 'rgba(222,45,38,0.8)'
    else:
        info_color = 'rgba(204,204,204,1)'

    df['info'] = pd.Series(info_mean)
    df['SEM'] = pd.Series(info_sem)
    df['95%ConfidenceInterval'] = pd.Series(1.96 * info_sem)
    df['p-value'] = pd.Series(info_pvalue)
    df['colorList'] = pd.Series(info_color)

    epoch_name_full = str(epoch).split('.')[0]
    epoch_name_sp = epoch_name_full.split('-')
    if epoch_name_sp[0] == 'Enc':
        epoch_name_sp[0] = 'Vis'
    if len(epoch_name_sp) > 2:
        epoch_name = epoch_name_sp[0] + '-' + epoch_name_sp[2]
        epoch_in_out = epoch_name_sp[1]
    else:
        epoch_name = epoch_name_sp[0] + '-' + epoch_name_sp[1]
    df['lag'] = lag
    df['epoch'] = epoch_name
    df['in/out'] = epoch_in_out
    df['method'] = method
    return df


def compute_info_partial(neurons_df, saccad_df, method_list, lag_list, typ, epoch):
    info_list = [complete_info_df(neurons_df, saccad_df, 0.05, epoch, y, z, typ) for y in method_list for z in
                 lag_list]
    return info_list


def compute_vlmc_for_each_trial(neuron: pd.core.frame.DataFrame, maxTime: int, number_of_column_added: int, trial: int):
    contexClassVlmc = vlmc.fit_vlmc(neuron.iloc[trial, 0:(maxTime - number_of_column_added)])
    return contexClassVlmc


