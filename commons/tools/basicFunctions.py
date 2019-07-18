import os

import numpy as np
import pandas as pd
import scipy.io as sio
import networkx as nx

from networkx import random_reference, lattice_reference


# Columns name

def generatorTemp(size):
    for ii in range(size):
        yield 'T' + str(ii)


# generate eye data

def generateEyeDF(eye):
    tempList = {}

    for i in np.arange(4):
        tempList[i] = pd.DataFrame(data=eye[0][0][0][0][i])
    tdf = pd.DataFrame(data=pd.concat([tempList[0], tempList[1],
                                       tempList[2], tempList[3]], axis=1))
    colsName = [
        'SacStartTime',
        'SacStopTime',
        'SacStartPosX',
        'SacStartPosY',
        'SacStopPosX',
        'SacStopPosY',
    ]
    tdf.columns = colsName
    return tdf


# assemble data in all neurons

def assembleData(directory):
    dirr = directory
    os.chdir(dirr)
    iter = 0
    neurons = {}
    allNeurons = {}

    for file in os.listdir(dirr):
        filename = os.fsdecode(file)
        if filename.endswith('.mat'):
            print('File:' + filename)
            matData = sio.loadmat(filename)
            dictVal = matData.get('Res')
            Eye = dictVal['Eye']
            spk = dictVal['spk']
            Cond = dictVal['Cond']
            spkLenOuter = len(spk[0][0])
            for iter0 in range(spkLenOuter):
                spkLeninner = len(spk[0][0][iter0])
                for iter2 in range(spkLeninner):
                    arrSize = spk[0][0][iter0][iter2]
                    if arrSize.size > 0:
                        df = pd.DataFrame(spk[0][0][iter0][iter2])
                        if sum(df.sum()) > 2000:
                            tmp = df
                            colName = generatorTemp(tmp.shape[1])
                            tmp.columns = colName
                            neurons[iter] = pd.concat([tmp,
                                                       pd.DataFrame(data={'Cond': Cond[0][0][0]})],
                                                      axis=1)
                            neurons[iter]['stimStatus'] = np.where(neurons[iter]['Cond'] > 8, 0, 1)
                            neurons[iter]['inOutStatus'] = np.where(neurons[iter]['Cond'] % 2 == 1, 1, 0)
                            allNeurons[iter] = pd.concat([neurons[iter],
                                                          generateEyeDF(Eye)], axis=1)
                            print("Neuron" + str(iter))
                            iter = iter + 1
                        else:
                            iter = iter
                            print("Neuron" + str(iter) + " " + "got few action potentials, skipping...")
    return allNeurons


def assembleData1(directory):
    dirr = directory
    os.chdir(dirr)
    neurons = {}
    matData = sio.loadmat('Nu.mat')
    dictVal = matData['nu']
    Cond = sio.loadmat('conNU.mat').get('condNU')
    for iter in range(len(dictVal[0])):
        arrSize = dictVal[0][iter]
        if arrSize.size > 0:
            df = pd.DataFrame(dictVal[0][iter])
            if sum(df.sum()) > 2000:
                tmp = df
                colName = generatorTemp(tmp.shape[1])
                tmp.columns = colName
                neurons[iter] = pd.concat([tmp,
                                           pd.DataFrame(Cond, columns=['Cond'])],
                                          axis=1)
                neurons[iter]['stimStatus'] = np.where(neurons[iter]['Cond'] > 8, 0, 1)
                neurons[iter]['inOutStatus'] = np.where(neurons[iter]['Cond'] % 2 == 1, 1, 0)
                print("Neuron" + str(iter))
            else:
                print("Neuron" + str(iter) + " " + "got few action potentials, skipping...")

    return neurons


def saccade_df(neurons_df):
    saccade_df = {}
    tmp_list1 = []
    tmp_list2 = []

    neurons_df1 = {}

    for numerator in range(len(neurons_df)):
        saccade_time = neurons_df[numerator]['SacStartTime']
        neurons_df1[numerator] = neurons_df[numerator].iloc[:,
                                 0:(neurons_df[numerator].columns.get_loc("Cond") - 1)]
        for i in range(len(saccade_time)):
            tmp_list1.append(
                neurons_df1[numerator].iloc[i,
                (saccade_time[i] - 3000):saccade_time[i]].reset_index(drop=True))
            tmp_list2.append(
                neurons_df1[numerator].iloc[i,
                (saccade_time[i] + 1):(saccade_time[i] + 400)].reset_index(drop=True))
        saccade_df[numerator] = pd.concat([pd.DataFrame(tmp_list1),
                                           pd.DataFrame(tmp_list2),
                                           neurons_df[numerator].iloc[:,
                                           (neurons_df[numerator].columns.get_loc("Cond")):
                                           (neurons_df[numerator].columns.get_loc("SacStopPosY"))]], axis=1)
        tmp_list1 = []
        tmp_list2 = []
    return saccade_df


def sacTime(df):
    dfNew = int(df['SacStartTime'].mean())
    return dfNew


# Compute firing rate

def computeFr(df, min, max):
    dtemp = np.mean(df.iloc[:, min:max]) * 1000
    return dtemp


def computeFr(df):
    min = 0
    max = df.shape[1]
    dtemp = np.mean(df.iloc[:, min:max]) * 1000
    return dtemp


def evoked_response(df, base_line):
    evoked = (df - min(base_line)) / (max(df) - min(base_line))
    return evoked


def evoked_response_count(df, base_line):
    evoked = np.ceil((df - np.mean(base_line)))
    return evoked


def computeSpikeCount(df, min, max):
    dtemp = np.sum(df.iloc[:, min:max])
    return dtemp


def computeSpikeCount(df):
    min = 0
    max = df.shape[1]
    dtemp = np.sum(df.iloc[:, min:max])
    return dtemp


def computeFrDict(df, min, max):
    dtemp = np.mean(df.iloc[:, min:max]) * 1000
    dtemp = pd.DataFrame(dtemp).reset_index(drop=True)
    dtemp.columns = ['firing_rate']
    return dtemp


def computeSpikeCountDict(df, min, max):
    dtemp = np.sum(df.iloc[:, min:max])
    dtemp = pd.DataFrame(dtemp).reset_index(drop=True)
    dtemp.columns = ['spike_count']
    return dtemp


# A domain mapper(between a and b) function
def normalize(vec, a, b):
    return (((vec - np.min(vec)) * (b - a)) / (np.max(vec) - np.min(vec))) + a


# Extract key-value from dict
def extract_from_dict(diction):
    key, value = zip(*diction.items())
    return key, value


# select different conditions

def conditionSelect(df, subStatus):
    if subStatus == 'inStim':
        dfNew = df[(df['stimStatus'] == 1) & (df['inOutStatus'] == 1)]
    elif subStatus == 'outStim':
        dfNew = df[(df['stimStatus'] == 1) & (df['inOutStatus'] == 0)]
    elif subStatus == 'inNoStim':
        dfNew = df[(df['stimStatus'] == 0) & (df['inOutStatus'] == 1)]
    elif subStatus == 'allStim':
        dfNew = df[(df['stimStatus'] == 1)]
    elif subStatus == 'allNoStim':
        dfNew = df[(df['stimStatus'] == 0)]
    elif subStatus == 'allIn':
        dfNew = df[(df['inOutStatus'] == 1)]
    elif subStatus == 'allOut':
        dfNew = df[(df['inOutStatus'] == 0)]
    else:
        dfNew = df[(df['stimStatus'] == 0) & (df['inOutStatus'] == 0)]
    return dfNew


# general firing rate computations

def computerFrAll(neurons_df, period):
    lend = len(neurons_df)
    saccade_data_frame = saccade_df(neurons_df)
    sep_by_cond = {}
    if period == 'vis':
        for it in range(lend):
            sep_by_cond[it] = [computeFr(conditionSelect(neurons_df[it],
                                                         'inStim'), 0, 3000),
                               computeFr(conditionSelect(neurons_df[it],
                                                         'outStim'), 0, 3000),
                               computeFr(conditionSelect(neurons_df[it],
                                                         'inNoStim'), 0, 3000),
                               computeFr(conditionSelect(neurons_df[it],
                                                         'outNoStim'), 0, 3000)]
    else:
        for it in range(lend):
            inStimDF = conditionSelect(saccade_data_frame[it], 'inStim')
            outStimDF = conditionSelect(saccade_data_frame[it], 'outStim')
            inNoStimDF = conditionSelect(saccade_data_frame[it], 'inNoStim')
            outNoStimDF = conditionSelect(saccade_data_frame[it], 'outNoStim')

            sep_by_cond[it] = [computeFr(inStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond") - 1)),
                               computeFr(outStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond") - 1)),
                               computeFr(inNoStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond") - 1)),
                               computeFr(outNoStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond") - 1))]
    return sep_by_cond


def computerSpkCountAll(neurons_df, period):
    lend = len(neurons_df)
    saccade_data_frame = saccade_df(neurons_df)
    sep_by_cond = {}
    if period == 'vis':
        for it in range(lend):
            sep_by_cond[it] = [computeSpikeCount(conditionSelect(neurons_df[it],
                                                                 'inStim'), 0, 3000),
                               computeSpikeCount(conditionSelect(neurons_df[it],
                                                                 'outStim'), 0, 3000),
                               computeSpikeCount(conditionSelect(neurons_df[it],
                                                                 'inNoStim'), 0, 3000),
                               computeSpikeCount(conditionSelect(neurons_df[it],
                                                                 'outNoStim'), 0, 3000)]
    else:
        for it in range(lend):
            inStimDF = conditionSelect(saccade_data_frame[it], 'inStim')
            outStimDF = conditionSelect(saccade_data_frame[it], 'outStim')
            inNoStimDF = conditionSelect(saccade_data_frame[it], 'inNoStim')
            outNoStimDF = conditionSelect(saccade_data_frame[it], 'outNoStim')

            sep_by_cond[it] = [computeSpikeCount(inStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond") - 1)),
                               computeSpikeCount(outStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond") - 1)),
                               computeSpikeCount(inNoStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond") - 1)),
                               computeSpikeCount(outNoStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond") - 1))]
    return sep_by_cond


def computerFrAllDict(neurons_df):
    lend = len(neurons_df)
    saccade_data_frame = saccade_df(neurons_df)
    sep_by_cond = []

    for it in range(lend):
        inStimDF = conditionSelect(saccade_data_frame[it], 'inStim')
        outStimDF = conditionSelect(saccade_data_frame[it], 'outStim')
        inNoStimDF = conditionSelect(saccade_data_frame[it], 'inNoStim')
        outNoStimDF = conditionSelect(saccade_data_frame[it],
                                      'outNoStim')

        sep_by_cond.append({'visual': {
            'inStim': computeFrDict(conditionSelect(neurons_df[it], 'inStim'
                                                    ), 0, 3000).to_dict('list'),
            'outStim': computeFrDict(conditionSelect(neurons_df[it],
                                                     'outStim'), 0, 3000).to_dict('list'),
            'inNoStim': computeFrDict(conditionSelect(neurons_df[it],
                                                      'inNoStim'), 0, 3000).to_dict('list'
                                                                                    ),
            'outNoStim': computeFrDict(conditionSelect(neurons_df[it],
                                                       'outNoStim'), 0, 3000).to_dict('list'
                                                                                      ),
        }, 'saccade': {
            'inStim': computeFrDict(inStimDF, 0,
                                    saccade_data_frame[it].columns.get_loc('Cond'
                                                                           ) - 1).to_dict('list'),
            'outStim': computeFrDict(outStimDF, 0,
                                     saccade_data_frame[it].columns.get_loc('Cond'
                                                                            ) - 1).to_dict('list'),
            'inNoStim': computeFrDict(inNoStimDF, 0,
                                      saccade_data_frame[it].columns.get_loc('Cond'
                                                                             ) - 1).to_dict('list'),
            'outNoStim': computeFrDict(outNoStimDF, 0,
                                       saccade_data_frame[it].columns.get_loc('Cond'
                                                                              ) - 1).to_dict('list'),
        }})
    return sep_by_cond


def computeSpikeCountALLDict(neurons_df):
    lend = len(neurons_df)
    saccade_data_frame = saccade_df(neurons_df)
    sep_by_cond = []

    for it in range(lend):
        inStimDF = conditionSelect(saccade_data_frame[it], 'inStim')
        outStimDF = conditionSelect(saccade_data_frame[it], 'outStim')
        inNoStimDF = conditionSelect(saccade_data_frame[it], 'inNoStim')
        outNoStimDF = conditionSelect(saccade_data_frame[it],
                                      'outNoStim')

        sep_by_cond.append({'visual': {
            'inStim': computeSpikeCountDict(conditionSelect(neurons_df[it], 'inStim'
                                                            ), 0, 3000).to_dict('list'),
            'outStim': computeSpikeCountDict(conditionSelect(neurons_df[it],
                                                             'outStim'), 0, 3000).to_dict('list'),
            'inNoStim': computeSpikeCountDict(conditionSelect(neurons_df[it],
                                                              'inNoStim'), 0, 3000).to_dict('list'
                                                                                            ),
            'outNoStim': computeSpikeCountDict(conditionSelect(neurons_df[it],
                                                               'outNoStim'), 0, 3000).to_dict('list'
                                                                                              ),
        }, 'saccade': {
            'inStim': computeSpikeCountDict(inStimDF, 0,
                                            saccade_data_frame[it].columns.get_loc('Cond'
                                                                                   ) - 1).to_dict('list'),
            'outStim': computeSpikeCountDict(outStimDF, 0,
                                             saccade_data_frame[it].columns.get_loc('Cond'
                                                                                    ) - 1).to_dict('list'),
            'inNoStim': computeSpikeCountDict(inNoStimDF, 0,
                                              saccade_data_frame[it].columns.get_loc('Cond'
                                                                                     ) - 1).to_dict('list'),
            'outNoStim': computeSpikeCountDict(outNoStimDF, 0,
                                               saccade_data_frame[it].columns.get_loc('Cond'
                                                                                      ) - 1).to_dict('list'),
        }})

    return sep_by_cond



# Graph processing functions
def rand_iterator(G, niter, seed, i):
    randMetrics = {}
    Gr = random_reference(G, niter=niter, seed=seed)
    Gl = lattice_reference(G, niter=niter, seed=seed)
    randMetrics["C"] = nx.transitivity(Gr)
    randMetrics["Co"] = nx.transitivity(Gl)
    randMetrics["L"] = nx.average_shortest_path_length(Gr)
    return randMetrics
