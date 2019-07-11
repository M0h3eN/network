import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# methods are pearsonr and mutual informatin


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
                    c = corrf(ac, bc)[0,1]
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

    return pd.DataFrame(correl, index=idx, columns=cols)

