import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sig

from dataImport.commons.basicFunctions import normalize

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


# rate_stim, estimated_rate_stim = fr_epoch('/home/partnerpc9_ib/SpyderDir/Out/Chain1/', 'In', 'Stim', al=True)


# A function for comparing firing rate and effective firing rate plot
# for desired epochs

def fr_plot(path, pos, st, epoch='Enc', al=False):
    mapper = {'Enc': 'Visual', 'Mem': 'Memory', 'Sac': 'Saccade'}

    if (al == True):
        data = fr_epoch(path, pos, st, al=True)
        rate_stim = data['rate_stim']
        estimated_rate_stim = data['estimated_rate_stim']
        col = plt.cm.get_cmap('jet', estimated_rate_stim.shape[1])
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        for pl in range(rate_stim.shape[1]):
            plt.plot(normalize(sig.savgol_filter(np.array(estimated_rate_stim)[:, pl], 11, 7)),
                     label=rate_stim.columns[pl], c=col(pl))
        plt.title('Normalized Effective Firing Rate')
        fig.legend(loc='center right')
        plt.subplot(2, 1, 2)
        for pl in range(estimated_rate_stim.shape[1]):
            plt.plot(normalize(sig.savgol_filter(np.array(rate_stim)[:, pl], 21, 3)),
                     label=estimated_rate_stim.columns[pl], c=col(pl))
        plt.title('Normalized Raw Firing Rate')
        fig.subplots_adjust(right=0.95)
        fig.suptitle('Visual,Memory,Saccade')
        plt.show()
    else:
        data = fr_epoch(path, pos, st)
        rate_stim = data['rate_stim']
        estimated_rate_stim = data['estimated_rate_stim']
        col = plt.cm.get_cmap('jet', estimated_rate_stim.shape[1])
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        for pl in range(rate_stim.shape[1]):
            plt.plot(normalize(sig.savgol_filter(np.array(estimated_rate_stim)[:, pl], 11, 7)),
                     label=rate_stim.columns[pl], c=col(pl))
        plt.title('Normalized Effective Firing Rate')
        fig.legend(loc='center right')
        plt.subplot(2, 1, 2)
        for pl in range(estimated_rate_stim.shape[1]):
            plt.plot(normalize(sig.savgol_filter(np.array(rate_stim)[:, pl], 21, 3)),
                     label=estimated_rate_stim.columns[pl], c=col(pl))
        plt.title('Normalized Raw Firing Rate')
        fig.subplots_adjust(right=0.95)
        fig.suptitle(mapper[epoch])
        plt.show()


# fr_plot('/home/partnerpc9_ib/SpyderDir/Out/Chain1/', 'In', 'Stim', 'Enc', al=False)
# fr_plot('/home/partnerpc9_ib/SpyderDir/Out/Chain1/', 'In', 'Stim', 'Mem', al=False)
# fr_plot('/home/partnerpc9_ib/SpyderDir/Out/Chain1/', 'In', 'Stim', 'Sac', al=False)
#
# fr_plot('/home/partnerpc9_ib/SpyderDir/Out/Chain1/', 'In', 'Stim', al=True)


## Correlation matrixes

def cor_plot(data):
    rate_stim = data['rate_stim']
    estimated_rate_stim = data['estimated_rate_stim']
    correlation_rate = rate_stim.corr()
    correlation_rate_estim = estimated_rate_stim.corr()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    cax = ax1.matshow(correlation_rate, vmin=-1, vmax=1, cmap=plt.cm.jet)
    cax2 = ax2.matshow(correlation_rate_estim, vmin=-1, vmax=1, cmap=plt.cm.jet)
    ticks = np.arange(0, rate_stim.shape[1], 1)
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.set_xticklabels(rate_stim.columns)
    ax1.set_yticklabels(rate_stim.columns)
    ax1.set_title('Raw Firing Rate')
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    ax2.set_xticklabels(estimated_rate_stim.columns)
    ax2.set_yticklabels(estimated_rate_stim.columns)
    ax2.set_title('Effective Firing Rate')
    fig.subplots_adjust(right=0.81)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cax, cax=cbar_ax)
    fig.suptitle('Correlation Matrix')
    plt.show()


# cor_plot(fr_epoch('/home/partnerpc9_ib/SpyderDir/Out/Chain1/', 'In', 'Stim', al=True))
#
# cor_plot(fr_epoch('/home/partnerpc9_ib/SpyderDir/Out/Chain1/', 'In', 'Stim', 'Mem', al=False))
# cor_plot(fr_epoch('/home/partnerpc9_ib/SpyderDir/Out/Chain1/', 'In', 'Stim', 'Sac', al=False))

# Scatter plot matrix

# pd.plotting.scatter_matrix(rate_stim, alpha=0.2, figsize=(rate_stim.shape[1], rate_stim.shape[1]), diagonal='kde')
# pd.plotting.scatter_matrix(estimated_rate_stim, alpha=0.2, figsize=(rate_stim.shape[1], rate_stim.shape[1]),
#                            diagonal='kde')