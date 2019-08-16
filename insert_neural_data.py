import os
import numpy as np
import pandas as pd
import time

from functools import partial
from multiprocessing import Pool, cpu_count
from argparse import ArgumentParser
from commons.tools.basicFunctions import assembleData2, conditionSelect, computeFr, evoked_response, \
    computeSpikeCount, evoked_response_count, saccade_df, base_line
from fitModel.pre_processing import split_epoch_condition, network_info_writer, raw_neuronal_data_info_compute

parser = ArgumentParser(description='This is a Python program for inserting neurons information in mongoDB')

parser.add_argument('-d', '--data', action='store',
                    dest='data', help='Raw data directory')

parser.add_argument('-w', '--write', action='store',
                    dest='write', help='Output directory')

parser.add_argument('-H', '--host', action='store',
                    dest='host', help='MongoDB host name')

parser.add_argument('-p', '--port', action='store',
                    dest='port', help='MongoDB port number')

args = parser.parse_args()

# parallel computing config
pool = Pool(cpu_count())

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

# slicing time to decompose Enc, Memory and saccade times
# Visual start time
visual = np.arange(1051, 1251)
# Memory start time
memory = np.arange(2500, 2700)
# Saccade start time
saccade = np.arange(2750, 2950)

firingRate = {'Enc-In-NoStim': pd.DataFrame([evoked_response(
                  computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, visual]),
                  computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line(visual)]))
                  for b in range(len(allNeurons))]).transpose(),
              'Mem-In-NoStim': pd.DataFrame([evoked_response(
                  computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, memory]),
                  computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line(visual)]))
                  for b in range(len(allNeurons))]).transpose(),
              'Sac-In-NoStim': pd.DataFrame([evoked_response(
                  computeFr(conditionSelect(saccad_df[b], 'inNoStim').iloc[:, saccade]),
                  computeFr(conditionSelect(saccad_df[b], 'inNoStim').iloc[:, base_line(visual)]))
                  for b in range(len(allNeurons))]).transpose(),
              'Enc-In-Stim': pd.DataFrame([evoked_response(
                  computeFr(conditionSelect(allNeurons[b], 'inStimVis').iloc[:, visual]),
                  computeFr(conditionSelect(allNeurons[b], 'inStimVis').iloc[:, base_line(visual)]))
                  for b in range(len(allNeurons))]).transpose(),
              'Mem-In-Stim': pd.DataFrame([evoked_response(
                  computeFr(conditionSelect(allNeurons[b], 'inStimMem').iloc[:, memory]),
                  computeFr(conditionSelect(allNeurons[b], 'inStimMem').iloc[:, base_line(visual)]))
                  for b in range(len(allNeurons))]).transpose(),
              'Sac-In-Stim': pd.DataFrame([evoked_response(
                  computeFr(conditionSelect(saccad_df[b], 'inStimSac').iloc[:, saccade]),
                  computeFr(conditionSelect(saccad_df[b], 'inStimSac').iloc[:, base_line(visual)]))
                  for b in range(len(allNeurons))]).transpose(),
              'Enc-diff': pd.DataFrame([evoked_response(
                  computeFr(conditionSelect(allNeurons[b], 'inStimVis').iloc[:, visual]) -
                  computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, visual]),
                  computeFr(conditionSelect(allNeurons[b], 'inStimVis').iloc[:, base_line(visual)]) -
                  computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line(visual)]))
                  for b in range(len(allNeurons))]).transpose(),
              'Mem-diff': pd.DataFrame([evoked_response(
                  computeFr(conditionSelect(allNeurons[b], 'inStimMem').iloc[:, memory]) -
                  computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, memory]),
                  computeFr(conditionSelect(allNeurons[b], 'inStimMem').iloc[:, base_line(visual)]) -
                  computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line(visual)]))
                  for b in range(len(allNeurons))]).transpose(),
              'Sac-diff': pd.DataFrame([evoked_response(
                  computeFr(conditionSelect(saccad_df[b], 'inStimSac').iloc[:, saccade]) -
                  computeFr(conditionSelect(saccad_df[b], 'inNoStim').iloc[:, saccade]),
                  computeFr(conditionSelect(saccad_df[b], 'inStimSac').iloc[:, base_line(visual)]) -
                  computeFr(conditionSelect(saccad_df[b], 'inNoStim').iloc[:, base_line(visual)]))
                  for b in range(len(allNeurons))]).transpose()
              }

spikeCounts = {'Enc-In-NoStim': pd.DataFrame([evoked_response_count(
                   computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, visual]),
                   computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line(visual)]))
                   for b in range(len(allNeurons))]).transpose(),
               'Mem-In-NoStim': pd.DataFrame([evoked_response_count(
                   computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, memory]),
                   computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line(visual)]))
                   for b in range(len(allNeurons))]).transpose(),
               'Sac-In-NoStim': pd.DataFrame([evoked_response_count(
                   computeSpikeCount(conditionSelect(saccad_df[b], 'inNoStim').iloc[:, saccade]),
                   computeSpikeCount(conditionSelect(saccad_df[b], 'inNoStim').iloc[:, base_line(visual)]))
                   for b in range(len(allNeurons))]).transpose(),
               'Enc-In-Stim': pd.DataFrame([evoked_response_count(
                   computeSpikeCount(conditionSelect(allNeurons[b], 'inStimVis').iloc[:, visual]),
                   computeSpikeCount(conditionSelect(allNeurons[b], 'inStimVis').iloc[:, base_line(visual)]))
                   for b in range(len(allNeurons))]).transpose(),
               'Mem-In-Stim': pd.DataFrame([evoked_response_count(
                   computeSpikeCount(conditionSelect(allNeurons[b], 'inStimMem').iloc[:, memory]),
                   computeSpikeCount(conditionSelect(allNeurons[b], 'inStimMem').iloc[:, base_line(visual)]))
                   for b in range(len(allNeurons))]).transpose(),
               'Sac-In-Stim': pd.DataFrame([evoked_response_count(
                   computeSpikeCount(conditionSelect(saccad_df[b], 'inStimSac').iloc[:, saccade]),
                   computeSpikeCount(conditionSelect(saccad_df[b], 'inStimSac').iloc[:, base_line(visual)]))
                   for b in range(len(allNeurons))]).transpose(),
               'Enc-diff': pd.DataFrame([evoked_response_count(
                   computeSpikeCount(conditionSelect(allNeurons[b], 'inStimVis').iloc[:, visual]) -
                   computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, visual]),
                   computeSpikeCount(conditionSelect(allNeurons[b], 'inStimVis').iloc[:, base_line(visual)]) -
                   computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line(visual)]))
                   for b in range(len(allNeurons))]).transpose(),
               'Mem-diff': pd.DataFrame([evoked_response_count(
                   computeSpikeCount(conditionSelect(allNeurons[b], 'inStimMem').iloc[:, memory]) -
                   computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, memory]),
                   computeSpikeCount(conditionSelect(allNeurons[b], 'inStimMem').iloc[:, base_line(visual)]) -
                   computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line(visual)]))
                   for b in range(len(allNeurons))]).transpose(),
               'Sac-diff': pd.DataFrame([evoked_response_count(
                   computeSpikeCount(conditionSelect(saccad_df[b], 'inStimSac').iloc[:, saccade]) -
                   computeSpikeCount(conditionSelect(saccad_df[b], 'inNoStim').iloc[:, saccade]),
                   computeSpikeCount(conditionSelect(saccad_df[b], 'inStimSac').iloc[:, base_line(visual)]) -
                   computeSpikeCount(conditionSelect(saccad_df[b], 'inNoStim').iloc[:, base_line(visual)]))
                   for b in range(len(allNeurons))]).transpose()
               }

# get neural data information
start_time = time.time()
raw_neuronal_data_info_compute(allNeurons, args)
split_epoch_condition(firingRate, spikeCounts, args)
print('************ Raw data ingestion completed in' +
      " %s seconds " % round((time.time() - start_time), 2) + ' ************')

# create a list of all networks
file_names = os.listdir(args.write + 'Firing Rate/')
referencePath = args.write + '/Chain1/MCMCValues/'

# create fit_par partial function
pearson_par = partial(network_info_writer, *[args, referencePath, 0.5, 'pearson'])
mutual_par = partial(network_info_writer, *[args, referencePath, 0.5, 'mutualScore'])
hawkes_par = partial(network_info_writer, *[args, referencePath, 0.5, 'hawkes'])

# pearson
start_time = time.time()
list(map(pearson_par, file_names))
print('**** Network information(pearson correlation) ingestion completed in'
      + " %s seconds " % round((time.time() - start_time), 2) + ' ****')

# mutual information
start_time = time.time()
list(map(mutual_par, file_names))
print('**** Network information(mutual information) ingestion completed in' +
      " %s seconds " % round((time.time() - start_time), 2) + ' ****')

# hawkes
start_time = time.time()
list(map(hawkes_par, file_names))
print('**** Network information(hawkes information) ingestion completed in' +
      " %s seconds " % round((time.time() - start_time), 2) + ' ****')
