import os
import time
import numpy as np
import pandas as pd
import tqdm

from argparse import ArgumentParser

from commons.selectivityMethods.general_information_calculator import complete_info_df, compute_info_partial
from commons.tools.basicFunctions import (assembleData2, saccade_df)
from multiprocessing import Pool, cpu_count
from functools import partial

parser = ArgumentParser(description='This is a Python program for analysis on network of neurons to '
                                    'detect functional connectivity between neurons')

parser.add_argument('-d', '--data', action='store',
                    dest='data', help='Raw data directory')

parser.add_argument('-l', '--lag', action='store',
                    dest='lag', help='epoch lag', type=int)

parser.add_argument('-w', '--write', action='store',
                    dest='write', help='Output directory')

parser.add_argument('-v', '--version', action='version',
                    dest='', version='%(prog)s 0.1')

args = parser.parse_args()

# parallel computing config
pool = Pool(cpu_count())

# prepare data
start_time = time.time()
# read all neurons
dirr = os.fsencode(args.data)
allNeurons = assembleData2(dirr)
saccad_df = saccade_df(allNeurons, 3000)
# align end date
minTime = np.min([allNeurons[x].shape[1] for x in range(len(allNeurons))])
# Find total spike count across all neurons
number_of_column_added = len(list(filter(lambda x: not x.startswith("T"), allNeurons.get(0).columns)))
sorted_total_spike_count = sorted(tuple([(x, sum(allNeurons[x].iloc[:, 0:(minTime - number_of_column_added)].sum()))
                                         for x in range(len(allNeurons))]), key=lambda tup: tup[1])

print('**** Total spike count across neurons sorted ascending ****')
for x in range(len(sorted_total_spike_count)):
    print('Neuron: ' + str(sorted_total_spike_count[x][0]) +
          ' Total Spike Count: ' + str(sorted_total_spike_count[x][1]))

writePath = args.write

if not os.path.exists(writePath):
    os.makedirs(writePath)

total_lag_ms = args.lag
method_list = ['pearson', 'mutualScore']
epoch_list = ['Enc-In-NoStim', 'Mem-In-NoStim', 'Sac-In-NoStim',
              'Enc-Out-NoStim', 'Mem-Out-NoStim', 'Sac-Out-NoStim',
              'Enc-In-Stim', 'Mem-In-Stim', 'Sac-In-Stim',
              'Enc-Out-Stim', 'Mem-Out-Stim', 'Sac-Out-Stim',
              'Enc-In-Diff', 'Mem-In-Diff', 'Sac-In-Diff',
              'Enc-Out-Diff', 'Mem-Out-Diff', 'Sac-Out-Diff']

lag_list = np.arange(-total_lag_ms, total_lag_ms + 1)

partial_info_function_evoked = partial(compute_info_partial, *[allNeurons, saccad_df, method_list, lag_list, 'evoked'])
partial_info_function_all = partial(compute_info_partial, *[allNeurons, saccad_df, method_list, lag_list, 'all_trials'])

all_df_list_all = list(tqdm.tqdm(pool.imap(partial_info_function_all, epoch_list), total=len(epoch_list)))
all_df_list_evoked = list(tqdm.tqdm(pool.imap(partial_info_function_evoked, epoch_list), total=len(epoch_list)))

# Flatten the lists
all_df_list_all = [item for sublist in all_df_list_all for item in sublist]
all_df_list_evoked = [item for sublist in all_df_list_evoked for item in sublist]


all_data_all = pd.concat(all_df_list_all)
all_data_evoked = pd.concat(all_df_list_evoked)

all_data_all.to_csv(writePath + 'All' + '.csv', index=False)
all_data_evoked.to_csv(writePath + 'Evoked' + '.csv', index=False)

print('************ Network assembly information evaluator completed in' +
      " %s seconds " % round((time.time() - start_time), 2) + '************')

