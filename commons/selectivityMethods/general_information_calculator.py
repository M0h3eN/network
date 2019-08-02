import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from commons.tools.basicFunctions import normalize
from scipy.stats import pearsonr

# methods are:
# 1- pearsonr
# 2- mutual information (continuous) ---- mutual
# 3- mutual information score (discreet) --- mutualScore


def info(data, method='pearson'):

    numeric_df = data._get_numeric_data()
    cols = numeric_df.columns
    idx = cols.copy()
    mat = numeric_df.values.T
    K = len(cols)
    correl = np.empty((K, K), dtype=float)

    if method == 'pearson':
        corrf = np.corrcoef
        for i, ac in enumerate(mat):
            for j, bc in enumerate(mat):
                if i > j:
                    continue
                elif i == j:
                    c = 1.
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
                    c = 1.
                else:
                    c = corrf(mat[i, :].reshape(-1, 1), mat[j, :])[0]
                correl[i, j] = c
                correl[j, i] = c

    elif method == 'mutualScore':
        corrf = mutual_info_score
        for i in range(K):
            for j in range(K):
                if i > j:
                    continue
                elif i == j:
                    c = 1.
                else:
                    c = corrf(mat[i, :], mat[j, :])
                correl[i, j] = c
                correl[j, i] = c

    else:
        raise ValueError("method must be either 'pearson', "
                         "'mutual', or 'mutualScore', '{method}' "
                         "was supplied".format(method=method))

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

