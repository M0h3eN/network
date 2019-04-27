import os
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from dataImport.commons.basicFunctions import assembleData, conditionSelect, computeFr
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
allNeurons = assembleData(dirr)

# align end date
minTime = np.min([allNeurons[x].shape[1] for x in range(len(allNeurons))])
np.sort(np.array([sum(allNeurons[x].iloc[:, 0:(minTime - 9)].sum()) for x in range(len(allNeurons))]))

# slicing time to decompose Enc, Memory and saccade times

neuronalData = {'Enc-In-NoStim': pd.DataFrame([computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, 1050:1250], 0, abs(1050-1250))
                      for b in range(len(allNeurons))]).transpose(),
                'Mem-In-NoStim': pd.DataFrame([computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, 2500:2700], 0, abs(2500-2700))
                      for b in range(len(allNeurons))]).transpose(),
                'Sac-In-NoStim': pd.DataFrame([computeFr(conditionSelect(allNeurons[b], 'inNoStim').iloc[:, 3150:3350], 0, abs(3150-3350))
                      for b in range(len(allNeurons))]).transpose(),
                'Enc-In-Stim': pd.DataFrame([computeFr(conditionSelect(allNeurons[b], 'inStim').iloc[:, 1050:1250], 0, abs(1050-1250))
                      for b in range(len(allNeurons))]).transpose(),
                'Mem-In-Stim': pd.DataFrame([computeFr(conditionSelect(allNeurons[b], 'inStim').iloc[:, 2500:2700], 0, abs(2500-2700))
                      for b in range(len(allNeurons))]).transpose(),
                'Sac-In-Stim': pd.DataFrame([computeFr(conditionSelect(allNeurons[b], 'inStim').iloc[:, 3150:3350], 0, abs(3150-3350))
                      for b in range(len(allNeurons))]).transpose()
                # 'Enc-Out-NoStim': np.array([conditionSelect(allNeurons[b], 'OutNoStim').iloc[:, 1050:1250].sum(axis=0)
                #       for b in range(len(allNeurons))]).transpose(),
                # 'Mem-Out-NoStim': np.array([conditionSelect(allNeurons[b], 'OutNoStim').iloc[:, 2500:2700].sum(axis=0)
                #       for b in range(len(allNeurons))]).transpose(),
                # 'Sac-Out-NoStim': np.array([conditionSelect(allNeurons[b], 'OutNoStim').iloc[:, 3150:3350].sum(axis=0)
                #       for b in range(len(allNeurons))]).transpose(),
                # 'Enc-Out-Stim': np.array([conditionSelect(allNeurons[b], 'outStim').iloc[:, 1050:1250].sum(axis=0)
                #       for b in range(len(allNeurons))]).transpose(),
                # 'Mem-Out-Stim': np.array([conditionSelect(allNeurons[b], 'outStim').iloc[:, 2500:2700].sum(axis=0)
                #       for b in range(len(allNeurons))]).transpose(),
                # 'Sac-Out-Stim': np.array([conditionSelect(allNeurons[b], 'outStim').iloc[:, 3150:3350].sum(axis=0)
                #       for b in range(len(allNeurons))]).transpose()
}

split_epoch_condition(neuronalData, args)
# get neural data information
raw_neuronal_data_info_compute(allNeurons, args)

print('**** data ingestion completed ****')
