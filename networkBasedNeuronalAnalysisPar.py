import os
import numpy as np
import plotly.io as pio
import time
import tqdm

from argparse import ArgumentParser
from commons.tools.basicFunctions import assembleData, conditionSelect, saccade_df,\
    computeSpikeCount, evoked_response_count
from commons.selectivityMethods.mi import computeMI
from fitModel.fit_model_par import fit_model_discrete_time_network_hawkes_spike_and_slab
from multiprocessing import Pool, cpu_count
from functools import partial

# plotly save configuration
pio.orca.reset_status()
pio.orca.ensure_server()
time.sleep(10)

parser = ArgumentParser(description='This is a Python program for analysis on network of neurons to '
                                    'detect functional connectivity between neurons')

parser.add_argument('-d', '--data', action='store',
                    dest='data', help='Raw data directory')

parser.add_argument('-H', '--host', action='store',
                    dest='host', help='MongoDB host name')

parser.add_argument('-p', '--port', action='store',
                    dest='port', help='MongoDB port number')

parser.add_argument('-w', '--write', action='store',
                    dest='write', help='Output directory')

parser.add_argument('-s', '--sparsity', action='store',
                    dest='sparsity', help='Initial sparsity of the network', type=float)

parser.add_argument('-S', '--self', action='store_true',
                    default=False,
                    dest='self', help='Allow self connection')

parser.add_argument('-l', '--lag', action='store',
                    dest='lag', help='Impulse response lag', type=int)

parser.add_argument('-i', '--iter', action='store',
                    dest='iter', help='Number of MCMC iteration', type=int)

parser.add_argument('-c', '--chain', action='store',
                    dest='chain', help='Number of MCMC chain', type=int)

parser.add_argument('-v', '--version', action='version',
                    dest='', version='%(prog)s 0.1')

args = parser.parse_args()

# parallel computing config
pool = Pool(cpu_count())

# prepare data

# read all neurons
dirr = os.fsencode(args.data)
allNeurons = assembleData(dirr)
saccad_df = saccade_df(allNeurons, 3000)
# align end date
minTime = np.min([allNeurons[x].shape[1] for x in range(len(allNeurons))])
# Find total spike count across all neurons
number_of_column_added = 10
sorted_total_spike_count = sorted(tuple([(x, sum(allNeurons[x].iloc[:, 0:(minTime - number_of_column_added)].sum()))
                                         for x in range(len(allNeurons))]), key=lambda tup: tup[1])

print('**** Total spike count across neurons sorted ascending ****')
for x in range(len(sorted_total_spike_count)):
    print('Neuron: ' + str(sorted_total_spike_count[x][0]) +
          ' Total Spike Count: ' + str(sorted_total_spike_count[x][1]))

# slicing time to decompose Enc, Memory and saccade times
# Base line start time
base_line = lambda x: np.arange((x[0] - 1) - 50, (x[0] - 1))
# Visual start time
visual = np.arange(1051, 1251)
# Memory start time
memory = np.arange(2501, 2701)
# Saccade start time
saccade = np.arange(2751, 2951)

# slicing time to decompose Enc, Memory, saccade and stimulation difference epochs

neuronalData = {'Enc-In-NoStim': np.array([evoked_response_count(
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, visual]),
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line(visual)]))
                      for b in range(len(allNeurons))]).transpose(),
                'Mem-In-NoStim': np.array([evoked_response_count(
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, memory]),
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line(memory)]))
                      for b in range(len(allNeurons))]).transpose(),
                'Sac-In-NoStim': np.array([evoked_response_count(
                    computeSpikeCount(conditionSelect(saccad_df[b], 'inNoStim').iloc[:, saccade]),
                    computeSpikeCount(conditionSelect(saccad_df[b], 'inNoStim').iloc[:, base_line(saccade)]))
                      for b in range(len(allNeurons))]).transpose(),
                'Enc-In-Stim': np.array([evoked_response_count(
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inStim').iloc[:, visual]),
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inStim').iloc[:, base_line(visual)]))
                      for b in range(len(allNeurons))]).transpose(),
                'Mem-In-Stim': np.array([evoked_response_count(
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inStim').iloc[:, memory]),
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inStim').iloc[:, base_line(memory)]))
                      for b in range(len(allNeurons))]).transpose(),
                'Sac-In-Stim': np.array([evoked_response_count(
                    computeSpikeCount(conditionSelect(saccad_df[b], 'inStim').iloc[:, saccade]),
                    computeSpikeCount(conditionSelect(saccad_df[b], 'inStim').iloc[:, base_line(saccade)]))
                      for b in range(len(allNeurons))]).transpose(),
                'Enc-diff': np.array([evoked_response_count(
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inStim').iloc[:, visual]) -
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, visual]),
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inStim').iloc[:, base_line(visual)]) -
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line(visual)]))
                      for b in range(len(allNeurons))]).transpose(),
                'Mem-diff': np.array([evoked_response_count(
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inStim').iloc[:, memory]) -
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, memory]),
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inStim').iloc[:, base_line(memory)]) -
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line(memory)]))
                      for b in range(len(allNeurons))]).transpose(),
                'Sac-diff': np.array([evoked_response_count(
                    computeSpikeCount(conditionSelect(saccad_df[b], 'inStim').iloc[:, saccade]) -
                    computeSpikeCount(conditionSelect(saccad_df[b], 'inNoStim').iloc[:, saccade]),
                    computeSpikeCount(conditionSelect(saccad_df[b], 'inStim').iloc[:, base_line(saccade)]) -
                    computeSpikeCount(conditionSelect(saccad_df[b], 'inNoStim').iloc[:, base_line(saccade)]))
                      for b in range(len(allNeurons))]).transpose()
                }

# network hyper parameter definition

network_hypers = {"p": args.sparsity, "allow_self_connections": args.self}

# fit model

period, data = zip(*neuronalData.items())

# Compute the mutual information

saccade_data_set = saccade_df(allNeurons)
mivalues = dict(Stim=computeMI(allNeurons, saccade_data_set, 'Stim'),
                NoStim=computeMI(allNeurons, saccade_data_set, 'NoStim'))

# Chain loop

writePath = args.write + 'Chain' + str(args.chain)
if not os.path.exists(writePath):
    os.makedirs(writePath)
tempPath = writePath

# create fit_par partial function
fit_par = partial(fit_model_discrete_time_network_hawkes_spike_and_slab,
                  *[args.lag, network_hypers, args.iter, period, data, allNeurons,
                    tempPath, args, mivalues, args.chain])

list(tqdm.tqdm(pool.imap(fit_par, list(range(len(period)))), total=len(period)))

pio.orca.shutdown_server()
