import numpy as np
from sklearn.feature_selection import mutual_info_regression
from commons.tools.basicFunctions import normalize
from scipy.stats import pearsonr

# methods are pearsonr and mutual informatin


def info(data, method='pearson'):

    numeric_df = data._get_numeric_data()
    cols = numeric_df.columns
    idx = cols.copy()
    mat = numeric_df.values.T
    K = len(cols)
    correl = np.empty((K, K), dtype=float)
    p_values = np.empty((K, K), dtype=float)

    # Compute p_values based on correlation

    for i, ac in enumerate(mat):
        for j, bc in enumerate(mat):
            if i > j:
                continue
            elif i == j:
                p = 0.
            else:
                p = pearsonr(ac, bc)[1]
            p_values[i, j] = p
            p_values[j, i] = p

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

    else:
        raise ValueError("method must be either 'pearson', "
                         "'spearman', or 'kendall', '{method}' "
                         "was supplied".format(method=method))

    return [np.array(correl, dtype=float), np.array(p_values, dtype=float)]


# set threshold based on p_value matrix

def set_threshold(data, p_values):

    K = data.shape[0]
    thresh_mat = np.empty(data.shape, dtype=float)
    thresholded_mat = np.empty(data.shape, dtype=float)

    for i in range(K):
        for j in range(K):
            if p_values[i, j] > 0.05:
                thresh_mat[i, j] = data[i, j]
            else:
                thresh_mat[i, j] = 0
    thresh = thresh_mat.flatten().mean(axis=0)
    for i in range(K):
        for j in range(K):
            if data[i, j] > thresh:
                thresholded_mat[i, j] = data[i, j]
            else:
                thresholded_mat[i, j] = 0
    return np.array(normalize(thresholded_mat.flatten(), 0, 1).reshape(K, K), dtype=float)

