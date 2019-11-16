import os
import time
import numpy as np

from argparse import ArgumentParser
from commons.tools.basicFunctions import (assembleData2, saccade_df)
from fitModel.pre_processing import raw_neuronal_data_info_compute

parser = ArgumentParser(description='This is a Python program for analysis on network of neurons to '
                                    'detect functional connectivity between neurons')

parser.add_argument('-d', '--data', action='store',
                    dest='data', help='Raw data directory')

parser.add_argument('-H', '--host', action='store',
                    default='localhost', dest='host', help='MongoDB host name')

parser.add_argument('-p', '--port', action='store',
                    default='27017', dest='port', help='MongoDB port number')

parser.add_argument('-w', '--write', action='store',
                    dest='write', help='Output directory')

parser.add_argument('-v', '--version', action='version',
                    dest='', version='%(prog)s 0.1')

args = parser.parse_args()

# Data preparation
start_time_initial = time.time()

start_time = time.time()
# read all neurons
dirr = os.fsencode(args.data)
allNeurons = assembleData2(dirr)
saccad_df = saccade_df(allNeurons, 3000)
# align end date
minTime = np.min([allNeurons[x].shape[1] for x in range(len(allNeurons))])
# Find total spike count across all neurons
number_of_column_added = len(list(filter(lambda x: not x.startswith("T"), allNeurons.get(0).columns)))

print('************ Neurons assemblies completed in' +
      " %s minutes " % round((time.time() - start_time) / 60, 4) + ' ************')

sorted_total_spike_count = sorted(tuple([(x, sum(allNeurons[x].iloc[:, 0:(minTime - number_of_column_added)].sum()))
                                         for x in range(len(allNeurons))]), key=lambda tup: tup[1])

print('**** Total spike count across neurons sorted ascending ****')
for x in range(len(sorted_total_spike_count)):
    print('Neuron: ' + str(sorted_total_spike_count[x][0]) +
          ' Total Spike Count: ' + str(sorted_total_spike_count[x][1]))


parrent_write_paths = 'Raw'
writeArg = args.write

raw_neuronal_data_info_compute(allNeurons, args)

