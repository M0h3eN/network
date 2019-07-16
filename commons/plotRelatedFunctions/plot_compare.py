import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sig

from commons.tools.basicFunctions import normalize

# =============================================================================
# A function for selecting rate at different epochs
# @path: path to estimated data from model.
# @pos: In or Out.
# @st: Stim or WithoutStim.
# @epoch: Which epoch to select Enc,Mem or Sac.
# @al: concat all epochs or not.
# returns a dictionary of two rates
# =============================================================================
def fr_epoch(path, pos, st, epoch='Enc', al=False):
    if (al == True):
        estimated_rate_stim = pd.concat(
            [pd.read_csv(path + "EstimatedRate" + '/' + 'Enc' + '-' + pos + '-' + st + '.csv'),
             pd.read_csv(path + "EstimatedRate" + '/' + 'Mem' + '-' + pos + '-' + st + '.csv'),
             pd.read_csv(path + "EstimatedRate" + '/' + 'Sac' + '-' + pos + '-' + st + '.csv')], axis=0)
        rate_stim = pd.concat([pd.read_csv(path + "Firing Rate" + '/' + 'Enc' + '-' + pos + '-' + st + '.csv'),
                               pd.read_csv(path + "Firing Rate" + '/' + 'Mem' + '-' + pos + '-' + st + '.csv'),
                               pd.read_csv(path + "Firing Rate" + '/' + 'Sac' + '-' + pos + '-' + st + '.csv')], axis=0)
    else:
        estimated_rate_stim = pd.read_csv(path + "EstimatedRate" + '/' + epoch + '-' + pos + '-' + st + '.csv')
        rate_stim = pd.read_csv(path + "Firing Rate" + '/' + epoch + '-' + pos + '-' + st + '.csv')

    labels = list(map(lambda x: 'N' + str(x + 1), range(estimated_rate_stim.shape[1])))
    estimated_rate_stim.columns = labels
    rate_stim.columns = labels

    return {'rate_stim': rate_stim, 'estimated_rate_stim': estimated_rate_stim}


# A function for comparing firing rate and effective firing rate plot
# for desired epochs
def fr_plot(path, pos, st, write_path, dataSet, epoch='Enc', al=False):
    data = fr_epoch(path, pos, st, al=al)
    rate = data[dataSet]
    if dataSet == 'rate_stim':
        window = 11
        order = 3
    else:
        window = 35
        order = 3
    if al:
        col = plt.cm.get_cmap('jet', rate.shape[1])
        plt.xlabel('Time (ms)')
        plt.ylabel('Normalized Firing Rate')
        for pl in range(rate.shape[1]):
            plt.plot(normalize(sig.savgol_filter(np.array(rate)[:, pl], window, order)),
                     label=rate.columns[pl], c=col(pl))
        plt.savefig(write_path + 'Firing Rates' + '-' + dataSet + '-' + pos + '-' + st + '-' + 'All' + '.svg', dpi=1000)
        plt.close()
    else:
        col = plt.cm.get_cmap('jet', rate.shape[1])
        for pl in range(rate.shape[1]):
            plt.plot(normalize(sig.savgol_filter(np.array(rate)[:, pl], window, order)),
                     label=rate.columns[pl], c=col(pl))
        plt.savefig(write_path + 'Firing Rates' + '-' + dataSet + '-' + pos + '-' + st + '-' + epoch + '.svg', dpi=1000)
        plt.close()

## Correlation matrixes
def cor_plot(path, pos, st, write_path, dataSet, epoch='Enc', al=False):
    data = fr_epoch(path, pos, st, epoch, al=al)
    rate = data[dataSet]
    correlation_rate = rate.corr()

    plt.matshow(correlation_rate, vmin=-1, vmax=1, cmap=plt.cm.jet)
    ticks = np.arange(0, rate.shape[1], 1)
    plt.xticks(ticks, rate.columns, rotation='vertical')
    plt.yticks(ticks, rate.columns)
    plt.colorbar()
    if al:
        plt.savefig(write_path + 'Correlation Matrix' + '-' + dataSet + '-' + pos + '-' + st + '-' + 'All' + '.svg', dpi=1000)
        plt.close()
    else:
        plt.savefig(write_path + 'Correlation Matrix' + '-' + dataSet + '-' + pos + '-' + st + '-' + epoch + '.svg', dpi=1000)
        plt.close()

# Scatter plot matrix
def scatter_plot(path, pos, st, write_path, dataSet, epoch='Enc', al=False):
    data = fr_epoch(path, pos, st, epoch, al=al)
    if dataSet == 'rate_stim':
        typ = 'Raw'
    else:
        typ = 'Effective'
    rate = data[dataSet]
    pd.plotting.scatter_matrix(rate, alpha=0.2, figsize=(rate.shape[1], rate.shape[1]),
                               diagonal='kde')
    if al:
        plt.savefig(write_path + 'Scatter Plot' + '-' + dataSet + '-' + pos + '-' + st + '-' + typ + '-' + 'All' + '.svg',dpi=1000)
        plt.close()
    else:
        plt.savefig(write_path + 'Scatter Plot' + '-' + dataSet + '-' + pos + '-' + st + '-' + typ + '-' + epoch + '.svg',dpi=1000)
        plt.close()
