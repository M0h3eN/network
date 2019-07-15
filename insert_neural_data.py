import os
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from dataImport.commons.basicFunctions import assembleData1, conditionSelect, computeFr, evoked_response,\
    computeSpikeCount, evoked_response_count
from fitModel.pre_processing import raw_neuronal_data_info_compute, split_epoch_condition


parser = ArgumentParser(description='This is a Python program for inserting neurons information in mongoDB')

parser.add_argument('-d', '--data',  action='store',
                    dest='data', help='Raw data directory')

parser.add_argument('-w', '--write', action='store',
                    dest='write', help='Output directory')

parser.add_argument('-H', '--host', action='store',
                    dest='host', help='MongoDB host name')

parser.add_argument('-p', '--port',action='store',
                    dest='port', help='MongoDB port number')

args = parser.parse_args()

# read all neurons
dirr = os.fsencode(args.data)
allNeurons = assembleData1(dirr)

# align end date
minTime = np.min([allNeurons[x].shape[1] for x in range(len(allNeurons))])
# Find total spike count across all neurons
number_of_column_added = 3
sorted_total_spike_count = sorted(tuple([(x, sum(allNeurons[x].iloc[:, 0:(minTime - number_of_column_added)].sum()))
                                         for x in range(len(allNeurons))]), key=lambda tup: tup[1])

print('**** Total spike count across neurons sorted ascending ****')
for x in range(len(sorted_total_spike_count)):
    print('Neuron: ' + str(sorted_total_spike_count[x][0]) +
          ' Total Spike Count: ' + str(sorted_total_spike_count[x][1]))

# slicing time to decompose Enc, Memory and saccade times
# Base line start time
base_line = np.arange(950, 1050)
# Visual start time
visual = np.arange(1051, 1251)
# Memory start time
memory = np.arange(2500, 2700)
# Saccade start time
saccade = np.arange(3150, 3350)
# All epochs
all_epochs = np.arange(1051, 3351)

firingRate = {'Enc-In-NoStim': pd.DataFrame([evoked_response(
                    computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, visual]),
                    computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line]))
                      for b in range(len(allNeurons))]).transpose(),
                'Mem-In-NoStim': pd.DataFrame([evoked_response(
                    computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, memory]),
                    computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line]))
                      for b in range(len(allNeurons))]).transpose(),
                'Sac-In-NoStim': pd.DataFrame([evoked_response(
                    computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, saccade]),
                    computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line]))
                      for b in range(len(allNeurons))]).transpose(),
                'All-NoStim': pd.DataFrame([evoked_response(
                    computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, all_epochs]),
                    computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line]))
                      for b in range(len(allNeurons))]).transpose(),
                'Enc-In-Stim': pd.DataFrame([evoked_response(
                    computeFr(conditionSelect(allNeurons[b], 'inStim').iloc[:, visual]),
                    computeFr(conditionSelect(allNeurons[b], 'inStim').iloc[:, base_line]))
                      for b in range(len(allNeurons))]).transpose(),
                'Mem-In-Stim': pd.DataFrame([evoked_response(
                    computeFr(conditionSelect(allNeurons[b], 'inStim').iloc[:, memory]),
                    computeFr(conditionSelect(allNeurons[b], 'inStim').iloc[:, base_line]))
                      for b in range(len(allNeurons))]).transpose(),
                'Sac-In-Stim': pd.DataFrame([evoked_response(
                    computeFr(conditionSelect(allNeurons[b], 'inStim').iloc[:, saccade]),
                    computeFr(conditionSelect(allNeurons[b], 'inStim').iloc[:, base_line]))
                      for b in range(len(allNeurons))]).transpose(),
                'All-Stim': pd.DataFrame([evoked_response(
                    computeFr(conditionSelect(allNeurons[b], 'inStim').iloc[:, all_epochs]),
                    computeFr(conditionSelect(allNeurons[b], 'inStim').iloc[:, base_line]))
                      for b in range(len(allNeurons))]).transpose()
             }

spikeCounts = {'Enc-In-NoStim': pd.DataFrame([evoked_response_count(
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, visual]),
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line]))
                      for b in range(len(allNeurons))]).transpose(),
                'Mem-In-NoStim': pd.DataFrame([evoked_response_count(
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, memory]),
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line]))
                      for b in range(len(allNeurons))]).transpose(),
                'Sac-In-NoStim': pd.DataFrame([evoked_response_count(
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, saccade]),
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line]))
                      for b in range(len(allNeurons))]).transpose(),
                'All-NoStim': pd.DataFrame([evoked_response_count(
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, all_epochs]),
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, base_line]))
                      for b in range(len(allNeurons))]).transpose(),
                'Enc-In-Stim': pd.DataFrame([evoked_response_count(
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inStim').iloc[:, visual]),
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inStim').iloc[:, base_line]))
                      for b in range(len(allNeurons))]).transpose(),
                'Mem-In-Stim': pd.DataFrame([evoked_response_count(
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inStim').iloc[:, memory]),
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inStim').iloc[:, base_line]))
                      for b in range(len(allNeurons))]).transpose(),
                'Sac-In-Stim': pd.DataFrame([evoked_response_count(
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inStim').iloc[:, saccade]),
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inStim').iloc[:, base_line]))
                      for b in range(len(allNeurons))]).transpose(),
                'All-Stim': pd.DataFrame([evoked_response_count(
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inStim').iloc[:, all_epochs]),
                    computeSpikeCount(conditionSelect(allNeurons[b], 'inStim').iloc[:, base_line]))
                      for b in range(len(allNeurons))]).transpose()
                }

split_epoch_condition(firingRate, spikeCounts, args)

print('**** data ingestion completed ****')
