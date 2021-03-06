import os

import numpy as np
import pandas as pd
import scipy.io as sio
import networkx as nx
import scipy.signal as sig
import scipy.ndimage as sin
import logging

from networkx import random_reference, lattice_reference
from scipy.stats import ttest_1samp, wilcoxon
from commons.tools.signal_processing import gen_fx_gsmooth

from typing import Dict, List, Tuple


def generatorTemp(size: int) -> str:
    """
    A function to generate Time point names T0,T1,...
    :param size: Time Sequence length
    :rtype: str
    """
    for ii in range(size):
        yield 'T' + str(ii)


def _tolist(ndarray: np.ndarray) -> list:
    """
    A recursive function which constructs lists from cellarrays
       (which are loaded as numpy ndarrays), recursion into the elements
       if they contain matobjects.
       :type ndarray: np.ndarray
   """
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list


# generate eye data
def generateEyeDF(eye, ed):
    tempList = {}
    for i in np.arange(6):
        tempList[i] = pd.DataFrame(data=eye[0][0][0][0][i])
    eye_data_df = pd.DataFrame(_tolist(ed[0][0][0]))
    tdf = pd.DataFrame(data=pd.concat([tempList[0], tempList[1],
                                       tempList[2], tempList[3],
                                       tempList[4], tempList[5],
                                       eye_data_df], axis=1))
    colsName = [
        'SacStartTime',
        'SacStopTime',
        'SacStartPosX',
        'SacStartPosY',
        'SacStopPosX',
        'SacStopPosY',
        'R',
        'Theta',
        'X',
        'Y'
    ]
    tdf.columns = colsName
    return tdf


def generateEyeDF2(eye):
    tempList = {}
    eye_data_names = list(eye.dtype.names)
    for i, name in enumerate(eye_data_names):
        tempList[i] = pd.DataFrame(data=eye[name][0][0])
    tdf = pd.DataFrame(data=pd.concat([tempList[0], tempList[1],
                                       tempList[2], tempList[3],
                                       tempList[4], tempList[5]], axis=1))
    colsName = [
        'SacStartTime',
        'SacStopTime',
        'SacStartPosX',
        'SacStartPosY',
        'SacStopPosX',
        'SacStopPosY',
        'R',
        'Theta'
    ]

    tdf.columns = colsName

    # setting all start time below 3000 equal to mean of the value above 3000

    condi = np.where(tdf['SacStartTime'] > 3000)
    condil = np.where(tdf['SacStartTime'] < 3000)
    value = np.ceil(tdf['SacStartTime'].loc[condi].mean())
    tdf['SacStartTime'].loc[condil] = value

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
            logging.info('File:' + filename)
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
                                                          generateEyeDF(Eye, dictVal['eyeData'])], axis=1)
                            logging.info("Neuron " + str(iter))
                            iter = iter + 1
                        else:
                            iter = iter
                            logging.info("Neuron " + str(iter) + " " + "got few action potentials, skipping...")
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
                logging.info("Neuron " + str(iter))
            else:
                logging.info("Neuron " + str(iter) + " " + "got few action potentials, skipping...")

    return neurons


def assembleData2(directory):
    dirr = directory
    os.chdir(dirr)
    filename = os.fsdecode(os.listdir(dirr).__getitem__(0))
    neurons = {}
    iter = 0
    iterp = 0
    mat_data = sio.loadmat(filename)
    mat_data = mat_data[list(filter(lambda x: not x.startswith("__"), mat_data.keys())).__getitem__(0)]
    mat_size = list(mat_data.shape)
    mat_names = list(mat_data.dtype.names)

    for iter0 in range(mat_size[0]):
        for iter1 in range(mat_size[1]):
            struct_data = mat_data[iter0, iter1]
            conds = struct_data[mat_names.__getitem__(2)].T
            spike_data = struct_data[mat_names.__getitem__(0)]
            eye_data = struct_data[mat_names.__getitem__(3)]
            spike_data_shape = list(spike_data.shape)
            if conds.size > 0:
                if len(spike_data_shape) > 2:
                    for iter3 in range(spike_data_shape.__getitem__(2)):
                        df = pd.DataFrame(spike_data[:, :, iter3])
                        if sum(df.sum()) > 2000:
                            colName = generatorTemp(df.shape[1])
                            df.columns = colName
                            neurons[iter] = pd.concat([df, pd.DataFrame(conds, columns=['Cond'])], axis=1)
                            neurons[iter]['stimStatus'] = np.where(neurons[iter]['Cond'] > 8, 0, 1)
                            neurons[iter]['inOutStatus'] = np.where(neurons[iter]['Cond'] % 2 == 1, 1, 0)
                            neurons[iter] = pd.concat([neurons[iter], generateEyeDF2(eye_data)], axis=1)
                            logging.info("Neuron " + str(iterp))
                            iter = iter + 1
                            iterp = iterp + 1
                        else:
                            logging.info("Neuron " + str(iterp) + " " + "got few action potentials, skipping...")
                            iter = iter
                            iterp = iterp + 1
                else:
                    df = pd.DataFrame(spike_data)
                    if sum(df.sum()) > 2000:
                        colName = generatorTemp(df.shape[1])
                        df.columns = colName
                        neurons[iter] = pd.concat([df, pd.DataFrame(conds, columns=['Cond'])], axis=1)
                        neurons[iter]['stimStatus'] = np.where(neurons[iter]['Cond'] > 8, 0, 1)
                        neurons[iter]['inOutStatus'] = np.where(neurons[iter]['Cond'] % 2 == 1, 1, 0)
                        neurons[iter] = pd.concat([neurons[iter], generateEyeDF2(eye_data)], axis=1)
                        logging.info("Neuron " + str(iterp))
                        iter = iter + 1
                        iterp = iterp + 1
                    else:
                        logging.info("Neuron " + str(iterp) + " " + "got few action potentials, skipping...")
                        iterp = iterp + 1
                        iter = iter
            else:
                logging.info("Neuron " + str(iterp) + " " + "is empty")
                iterp = iterp + 1

    return neurons


def assembleData2_gen_fx_gsmooth(directory, sigma):
    dirr = directory
    os.chdir(dirr)
    filename = os.fsdecode(os.listdir(dirr).__getitem__(0))
    neurons = {}
    iter = 0
    iterp = 0
    mat_data = sio.loadmat(filename)
    mat_data = mat_data[list(filter(lambda x: not x.startswith("__"), mat_data.keys())).__getitem__(0)]
    mat_size = list(mat_data.shape)
    mat_names = list(mat_data.dtype.names)

    for iter0 in range(mat_size[0]):
        for iter1 in range(mat_size[1]):
            struct_data = mat_data[iter0, iter1]
            conds = struct_data[mat_names.__getitem__(2)].T
            spike_data = struct_data[mat_names.__getitem__(0)]
            eye_data = struct_data[mat_names.__getitem__(3)]
            spike_data_shape = list(spike_data.shape)
            if conds.size > 0:
                if len(spike_data_shape) > 2:
                    for iter3 in range(spike_data_shape.__getitem__(2)):
                        df = pd.DataFrame(gen_fx_gsmooth(spike_data[:, :, iter3], sigma))
                        if sum(df.sum()) > 200:
                            colName = generatorTemp(df.shape[1])
                            df.columns = colName
                            neurons[iter] = pd.concat([df, pd.DataFrame(conds, columns=['Cond'])], axis=1)
                            neurons[iter]['stimStatus'] = np.where(neurons[iter]['Cond'] > 8, 0, 1)
                            neurons[iter]['inOutStatus'] = np.where(neurons[iter]['Cond'] % 2 == 1, 1, 0)
                            neurons[iter] = pd.concat([neurons[iter], generateEyeDF2(eye_data)], axis=1)
                            logging.info("Neuron " + str(iterp))
                            iter = iter + 1
                            iterp = iterp + 1
                        else:
                            logging.info("Neuron " + str(iterp) + " " + "got few action potentials, skipping...")
                            iter = iter
                            iterp = iterp + 1
                else:
                    df = pd.DataFrame(gen_fx_gsmooth(spike_data, sigma))
                    if sum(df.sum()) > 200:
                        colName = generatorTemp(df.shape[1])
                        df.columns = colName
                        neurons[iter] = pd.concat([df, pd.DataFrame(conds, columns=['Cond'])], axis=1)
                        neurons[iter]['stimStatus'] = np.where(neurons[iter]['Cond'] > 8, 0, 1)
                        neurons[iter]['inOutStatus'] = np.where(neurons[iter]['Cond'] % 2 == 1, 1, 0)
                        neurons[iter] = pd.concat([neurons[iter], generateEyeDF2(eye_data)], axis=1)
                        logging.info("Neuron " + str(iterp))
                        iter = iter + 1
                        iterp = iterp + 1
                    else:
                        logging.info("Neuron " + str(iterp) + " " + "got few action potentials, skipping...")
                        iterp = iterp + 1
                        iter = iter
            else:
                logging.info("Neuron " + str(iterp) + " " + "is empty")
                iterp = iterp + 1

    return neurons


def saccade_df(neurons_df, align_point=3000):
    saccade_df = {}
    tmp_list1 = []
    tmp_list2 = []

    neurons_df1 = {}

    for numerator in range(len(neurons_df)):
        saccade_time = neurons_df[numerator]['SacStartTime']

        neurons_df1[numerator] = neurons_df[numerator].iloc[:, 0:(neurons_df[numerator].columns.get_loc("Cond") - 1)]
        for i in range(len(saccade_time)):
            pre_col_index = np.arange((saccade_time[i] - align_point), saccade_time[i])

            # Check for end point selection
            if (2 * saccade_time[i] - align_point) <= (neurons_df[numerator].columns.get_loc("Cond") - 1):
                end_point = (2 * saccade_time[i] - align_point)
            else:
                end_point = (neurons_df[numerator].columns.get_loc("Cond") - 1)

            post_col_index = np.arange((saccade_time[i] + 1), end_point)

            series1 = neurons_df1[numerator].iloc[i, pre_col_index].reset_index(drop=True)
            series2 = neurons_df1[numerator].iloc[i, post_col_index].reset_index(drop=True)

            tmp_list1.append(series1)
            tmp_list2.append(series2)

        df1 = pd.DataFrame(tmp_list1)
        df1.columns = list(np.arange(0, df1.shape[1]))
        df2 = pd.DataFrame(tmp_list2)
        df2.columns = list(np.arange(df1.shape[1], df1.shape[1] + df2.shape[1]))

        # Fill na with zero
        saccade_df[numerator] = pd.concat([df1.fillna(0), df2.fillna(0),
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

def computeFr(df, minimum=None, maximum=None):
    if minimum is not None and maximum is not None:
        dtemp = np.mean(df.iloc[:, minimum:maximum]) * 1000
    else:
        minimum = 0
        maximum = df.shape[1]
        dtemp = np.mean(df.iloc[:, minimum:maximum]) * 1000
    return dtemp


def evoked_response(df, base_line):
    evoked = (df - np.mean(base_line))
    return evoked


def evoked_response_count(df, base_line):
    evoked = np.ceil((df - np.mean(base_line)))
    return set_neg_to_zero(evoked)


def computeSpikeCount(df, minimum=None, maximum=None):
    if minimum is not None and maximum is not None:
        dtemp = np.sum(df.iloc[:, minimum:maximum])
    else:
        minimum = 0
        maximum = df.shape[1]
        dtemp = np.sum(df.iloc[:, minimum:maximum])
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


# select different conditions based on:

# cond1= target in, stimulation during fixation
# cond2= target out, stimulation during fixation
# cond3= target in, stimulation during visual period
# cond4= target out, stimulation during visual period
# cond5= target in, stimulation during memory period
# cond6= target out, stimulation during memory period
# cond7= target in, stimulation during saccade period
# cond8= target out, stimulation during saccade period
# cond9= target in, no stimulation
# cond10= target out, no stimulation
# cond11= target in, no stimulation
# cond12= target out, no stimulation
# cond13= target in, no stimulation
# cond14= target out, no stimulation
# cond15= target in, no stimulation
# cond16= target out, no stimulation


def conditionSelect(df, subStatus):
    if subStatus == 'inStimFixation':
        dfNew = df[(df['stimStatus'] == 1) & (df['inOutStatus'] == 1) & (df['Cond'] == 1)]
    elif subStatus == 'inStimVis':
        dfNew = df[(df['stimStatus'] == 1) & (df['inOutStatus'] == 1) & (df['Cond'] == 3)]
    elif subStatus == 'inStimMem':
        dfNew = df[(df['stimStatus'] == 1) & (df['inOutStatus'] == 1) & (df['Cond'] == 5)]
    elif subStatus == 'inStimSac':
        dfNew = df[(df['stimStatus'] == 1) & (df['inOutStatus'] == 1) & (df['Cond'] == 7)]

    elif subStatus == 'outStimFixation':
        dfNew = df[(df['stimStatus'] == 1) & (df['inOutStatus'] == 0) & (df['Cond'] == 2)]
    elif subStatus == 'outStimVis':
        dfNew = df[(df['stimStatus'] == 1) & (df['inOutStatus'] == 0) & (df['Cond'] == 4)]
    elif subStatus == 'outStimMem':
        dfNew = df[(df['stimStatus'] == 1) & (df['inOutStatus'] == 0) & (df['Cond'] == 6)]
    elif subStatus == 'outStimSac':
        dfNew = df[(df['stimStatus'] == 1) & (df['inOutStatus'] == 0) & (df['Cond'] == 8)]

    elif subStatus == 'inNoStimFixation':
        dfNew = df[(df['stimStatus'] == 0) & (df['inOutStatus'] == 1) & (df['Cond'] == 9)]
    elif subStatus == 'inNoStimVis':
        dfNew = df[(df['stimStatus'] == 0) & (df['inOutStatus'] == 1) & (df['Cond'] == 11)]
    elif subStatus == 'inNoStimMem':
        dfNew = df[(df['stimStatus'] == 0) & (df['inOutStatus'] == 1) & (df['Cond'] == 13)]
    elif subStatus == 'inNoStimSac':
        dfNew = df[(df['stimStatus'] == 0) & (df['inOutStatus'] == 1) & (df['Cond'] == 15)]

    elif subStatus == 'inStim':
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
        saccade_data_frame = saccade_df(neurons_df)
        for it in range(lend):
            inStimDF = conditionSelect(saccade_data_frame[it], 'inStim')
            outStimDF = conditionSelect(saccade_data_frame[it], 'outStim')
            inNoStimDF = conditionSelect(saccade_data_frame[it], 'inNoStim')
            outNoStimDF = conditionSelect(saccade_data_frame[it], 'outNoStim')

            sep_by_cond[it] = [computeFr(inStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond"))),
                               computeFr(outStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond"))),
                               computeFr(inNoStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond"))),
                               computeFr(outNoStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond")))]
    return sep_by_cond


def complete_fr_df(neurons_df, neuron, condition, filter_method, min_time, max_time):
    out_data_frame = pd.DataFrame()
    raw_fr = computeFr(conditionSelect(neurons_df[neuron], condition), min_time, max_time)
    out_data_frame['raw_fr'] = raw_fr

    if filter_method == 'gen_fx':
        out_data_frame['filtered_fr'] = gen_fx_gsmooth(raw_fr, 60)
    elif filter_method == 'savgol':
        out_data_frame['filtered_fr'] = sig.savgol_filter(raw_fr, 215, 3)
    else:
        out_data_frame['filtered_fr'] = sin.gaussian_filter(raw_fr, 11)

    out_data_frame['condition'] = condition
    out_data_frame['filter'] = filter_method
    out_data_frame['neuron'] = str('N' + str(neuron + 1))
    recording_time_ms = np.arange(len(out_data_frame)) + 1
    out_data_frame['time'] = recording_time_ms

    return out_data_frame


def computerFrAllDataFrame(neurons_df, period):
    lend = len(neurons_df)
    condition_list = ['inStim', 'outStim', 'inNoStim', 'outNoStim']
    filter_method_list = ['gen_fx', 'savgol', 'gaussian']

    if period == 'vis':
        fr_list = [complete_fr_df(neurons_df, x, y, z, 0, 3000)
                   for x in range(lend) for y in condition_list for z in filter_method_list]
    else:
        saccade_data_frame = saccade_df(neurons_df)
        fr_list = [complete_fr_df(saccade_data_frame, x, y, z, 0, saccade_data_frame[x].columns.get_loc("Cond"))
                   for x in range(lend) for y in condition_list for z in filter_method_list]

    fr_long_df = pd.concat(fr_list, axis=0)

    return fr_long_df


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


def evaluate_neuron_significance_based_on_fixation(all_neuron_df: dict,
                                                   saccade_data_frame: dict,
                                                   fixation_min_time_point: int,
                                                   fixation_max_time_point: int) -> \
                                                   Tuple[dict, dict, dict, pd.core.frame.DataFrame]:
    """
    Evalute which neuron is significant based on epoch comparision with fixation period of the neuron
    :param all_neuron_df:
    :param saccade_data_frame:
    :param fixation_min_time_point:
    :param fixation_max_time_point:
    :return: dictionary of significant neurons.
    """
    def df_slicer(df, s_df, fixation_min_time_point, fixation_max_time_point, row):
        fixation = np.arange(fixation_min_time_point, fixation_max_time_point)
        visual = np.arange(1051, 1251)
        memory = np.arange(2500, 2700)
        saccade = np.arange(2750, 2950)

        sliced_df = pd.DataFrame()
        sliced_df['Fixation'] = pd.Series(df.iloc[row, fixation].sum())
        sliced_df['Visual'] = pd.Series(df.iloc[row, visual].sum())
        sliced_df['Memory'] = pd.Series(df.iloc[row, memory].sum())
        sliced_df['Saccade'] = pd.Series(s_df.iloc[row, saccade].sum())

        return sliced_df

    # def sig_index_df(indexe, epoch):
    #
    #     sig_DF = pd.DataFrame()
    #     sig_DF['sig_neuron'] = pd.Series(str('N' + str(indexe + 1)))
    #     sig_DF['epoch'] = pd.Series(epoch)
    #
    #     return sig_DF


    allNeuronsFiltered = {}
    p_value_DF = pd.DataFrame(columns=['Vis', 'Mem', 'Sac', 'Is_significant'])

    for iterate in range(len(all_neuron_df)):
        df_cond_filter = conditionSelect(all_neuron_df[iterate], 'inNoStim')
        s_df_cond_filter = conditionSelect(saccade_data_frame[iterate], 'inNoStim')

        df_epoch_list = [df_slicer(df_cond_filter, s_df_cond_filter, fixation_min_time_point,
                                   fixation_max_time_point, i) for i in range(len(df_cond_filter))]
        df_epoch = pd.concat(df_epoch_list, axis=0)
        df_epoch = df_epoch.reset_index(drop=True)

        vis_p = wilcoxon(df_epoch['Fixation'], df_epoch['Visual'])[1]
        mem_p = wilcoxon(df_epoch['Fixation'], df_epoch['Memory'])[1]
        sac_p = wilcoxon(df_epoch['Fixation'], df_epoch['Saccade'])[1]
        p_val_counter = (vis_p < 0.05) | (mem_p < 0.05) | (sac_p < 0.05)

        p_value_DF.loc[iterate] = pd.Series({'Vis': vis_p, 'Mem': mem_p, 'Sac': sac_p, 'Is_significant': p_val_counter})

        allNeuronsFiltered[iterate] = df_epoch

    l0 = list(p_value_DF[(p_value_DF['Vis'] < 0.05)].index.values)
    l1 = list(p_value_DF[(p_value_DF['Mem'] < 0.05)].index.values)
    l2 = list(p_value_DF[(p_value_DF['Sac'] < 0.05)].index.values)

    vis_indexes = pd.DataFrame(np.array(list(map(lambda x: 'N' + str(x + 1), l0))))
    mem_indexes = pd.DataFrame(np.array(list(map(lambda x: 'N' + str(x + 1), l1))))
    sac_idexes = pd.DataFrame(np.array(list(map(lambda x: 'N' + str(x + 1), l2))))

    vis_indexes['epoch'] = 'Visual'
    mem_indexes['epoch'] = 'Memory'
    sac_idexes['epoch'] = 'Saccade'

    # sig_df_list = [sig_index_df(x, y) for x in [vis_indexes, mem_indexes, sac_idexes] for y in ['Visual', 'Memory', 'Saccade']]
    sig_df = pd.concat([vis_indexes, mem_indexes, sac_idexes], axis=0)

    significant_neuron_list_vis = [all_neuron_df[x] for x in l0]
    significant_neuron_list_mem = [all_neuron_df[x] for x in l1]
    significant_neuron_list_sac = [all_neuron_df[x] for x in l2]

    significant_neuron_dict_vis = {x: significant_neuron_list_vis[x] for x in range(len(significant_neuron_list_vis))}
    significant_neuron_dict_mem = {x: significant_neuron_list_mem[x] for x in range(len(significant_neuron_list_mem))}
    significant_neuron_dict_sac = {x: significant_neuron_list_sac[x] for x in range(len(significant_neuron_list_sac))}

    return significant_neuron_dict_vis, significant_neuron_dict_mem, significant_neuron_dict_sac, sig_df


# Graph processing functions
def rand_iterator(G, niter, seed, i):
    randMetrics = {}
    Gr = random_reference(G, niter=niter, seed=seed)
    Gl = lattice_reference(G, niter=niter, seed=seed)
    randMetrics["C"] = nx.transitivity(Gr)
    randMetrics["Co"] = nx.transitivity(Gl)
    if nx.is_connected(Gr):
        randMetrics["L"] = nx.average_shortest_path_length(Gr)
    else:
        SG = [Gr.subgraph(c) for c in nx.connected_components(Gr)]
        randMetrics["L"] = np.mean([nx.average_shortest_path_length(g) for g in SG])
    return randMetrics


def set_neg_to_zero(data):
    data_positive = np.array(data)
    dim = data_positive.shape
    if len(dim) > 1:
        for i in range(dim[0]):
            for j in range(dim[1]):
                if data_positive[i, j] <= 0:
                    data_positive[i, j] = 0
    else:
        for i in range(dim[0]):
            if data_positive[i] <= 0:
                data_positive[i] = 0
    return data_positive


def epoch_extractor(neurons_data: pd.core.frame.DataFrame, condition: str, time_period: np.ndarray, typ: str,
                    lag: int) -> np.ndarray:
    """
    A function to extract neuronal data with spicifics epochs and various condition and evaluate either evoked
    response, spike count or raw spikes
    :param neurons_data: data set of all neurons
    :param condition: With stimmualtion -> {inStimVis, inStimMem, inStimSac, outStimVis, outStimMem, outStimSac},
     No stimmulation -> {inNoStim, outNoStim}
    :param time_period: visual, memory and saccade
    :param typ: response type -> evoked, evokedFr, count, Fr, all_trials
    :param lag: time lag negative and positive for epochs
    :return: array of responses for all neurons across time
    """
    if typ == 'evoked':
        array_data = np.array([evoked_response_count(
            computeSpikeCount(conditionSelect(neurons_data[b], condition).iloc[:, time_period + lag]),
            computeSpikeCount(conditionSelect(neurons_data[b], condition).iloc[:, base_line(time_period + lag)]))
            for b in range(len(neurons_data))], dtype=int).transpose()
    elif typ == 'evokedFr':
        array_data = np.array([evoked_response(
            computeFr(conditionSelect(neurons_data[b], condition).iloc[:, time_period + lag]),
            computeFr(conditionSelect(neurons_data[b], condition).iloc[:, base_line(time_period + lag)]))
            for b in range(len(neurons_data))], dtype=float).transpose()
    elif typ == 'count':
        array_data = np.array([computeSpikeCount(conditionSelect(neurons_data[b], condition).iloc[:, time_period + lag])
                               for b in range(len(neurons_data))], dtype=int).transpose()
    elif typ == 'Fr':
        array_data = np.array([computeFr(conditionSelect(neurons_data[b], condition).iloc[:, time_period + lag])
                               for b in range(len(neurons_data))], dtype=float).transpose()
    else:
        array_data = np.array([np.array(conditionSelect(neurons_data[b], condition).iloc[:, time_period + lag]) for b in
                               range(len(neurons_data))], dtype=int)

    return array_data


def generate_lagged_epochs(vis_df: pd.core.frame.DataFrame, mem_df: pd.core.frame.DataFrame,
                           saccad_df: pd.core.frame.DataFrame, epoch: str, lag: int, typ: str) -> np.ndarray:
    """
    Generate epochs based on epoch_extractor function with specified time lags based on time period
    :param vis_df: data set of all neurons in visual epoch
    :param mem_df: data set of all neurons in memory epoch
    :param saccad_df: data set of all saccade aligned neurons
    :param epoch: 'Enc-In-NoStim', 'Mem-In-NoStim', 'Sac-In-NoStim',
                  'Enc-Out-NoStim', 'Mem-Out-NoStim', 'Sac-Out-NoStim',
                  'Enc-In-Stim', 'Mem-In-Stim', 'Sac-In-Stim',
                  'Enc-Out-Stim', 'Mem-Out-Stim', 'Sac-Out-Stim',
                  'Enc-In-Diff', 'Mem-In-Diff', 'Sac-In-Diff',
                  'Enc-Out-Diff', 'Mem-Out-Diff', 'Sac-Out-Diff'.
    :param lag: time lag negative and positive for epochs
    :param typ: response type -> evoked, evokedFr, count, Fr, all_trials
    :return:
    """
    array_data = None
    # slicing time to decompose Enc, Memory and saccade times
    # Visual start time
    visual = np.arange(1051, 1251)
    # Memory start time
    memory = np.arange(2500, 2700)
    # Saccade start time
    saccade = np.arange(2750, 2950)

    if epoch == 'Enc-In-NoStim':
        array_data = epoch_extractor(vis_df, 'inNoStim', visual, typ, lag)
    elif epoch == 'Mem-In-NoStim':
        array_data = epoch_extractor(mem_df, 'inNoStim', memory, typ, lag)
    elif epoch == 'Sac-In-NoStim':
        array_data = epoch_extractor(saccad_df, 'inNoStim', saccade, typ, lag)

    elif epoch == 'Enc-Out-NoStim':
        array_data = epoch_extractor(vis_df, 'outNoStim', visual, typ, lag)
    elif epoch == 'Mem-Out-NoStim':
        array_data = epoch_extractor(mem_df, 'outNoStim', memory, typ, lag)
    elif epoch == 'Sac-Out-NoStim':
        array_data = epoch_extractor(saccad_df, 'outNoStim', saccade, typ, lag)

    elif epoch == 'Enc-In-Stim':
        array_data = epoch_extractor(vis_df, 'inStimVis', visual, typ, lag)
    elif epoch == 'Mem-In-Stim':
        array_data = epoch_extractor(mem_df, 'inStimMem', memory, typ, lag)
    elif epoch == 'Sac-In-Stim':
        array_data = epoch_extractor(saccad_df, 'inStimSac', saccade, typ, lag)

    elif epoch == 'Enc-Out-Stim':
        array_data = epoch_extractor(vis_df, 'outStimVis', visual, typ, lag)
    elif epoch == 'Mem-Out-Stim':
        array_data = epoch_extractor(mem_df, 'outStimMem', memory, typ, lag)
    elif epoch == 'Sac-Out-Stim':
        array_data = epoch_extractor(saccad_df, 'outStimSac', saccade, typ, lag)

    elif epoch == 'Enc-In-Diff':
        if typ != 'all_trials':
            array_data = epoch_extractor(vis_df, 'inStimVis', visual, typ, lag) - epoch_extractor(vis_df,
                                                                                                      'inNoStim',
                                                                                                      visual, typ, lag)
        else:
            array_data = epoch_extractor(vis_df, 'inStimVis', visual, typ, lag).sum(axis=1) - epoch_extractor(
                vis_df, 'inNoStim', visual, typ, lag).sum(axis=1)
    elif epoch == 'Mem-In-Diff':
        if typ != 'all_trials':
            array_data = epoch_extractor(mem_df, 'inStimMem', memory, typ, lag) - epoch_extractor(mem_df,
                                                                                                      'inNoStim',
                                                                                                      visual, typ, lag)
        else:
            array_data = epoch_extractor(mem_df, 'inStimMem', memory, typ, lag).sum(axis=1) - epoch_extractor(
                mem_df, 'inNoStim', visual, typ, lag).sum(axis=1)
    elif epoch == 'Sac-In-Diff':
        if typ != 'all_trials':
            array_data = epoch_extractor(saccad_df, 'inStimSac', saccade, typ, lag) - epoch_extractor(saccad_df,
                                                                                                      'inNoStim',
                                                                                                      saccade, typ, lag)
        else:
            array_data = epoch_extractor(saccad_df, 'inStimSac', saccade, typ, lag).sum(axis=1) - epoch_extractor(
                saccad_df, 'inNoStim', saccade, typ, lag).sum(axis=1)

    elif epoch == 'Enc-Out-Diff':
        if typ != 'all_trials':
            array_data = epoch_extractor(vis_df, 'outStimVis', visual, typ, lag) - epoch_extractor(vis_df,
                                                                                                       'outNoStim',
                                                                                                       visual, typ, lag)
        else:
            array_data = epoch_extractor(vis_df, 'outStimVis', visual, typ, lag).sum(axis=1) - epoch_extractor(
                vis_df, 'outNoStim', visual, typ, lag).sum(axis=1)
    elif epoch == 'Mem-Out-Diff':
        if typ != 'all_trials':
            array_data = epoch_extractor(mem_df, 'outStimMem', memory, typ, lag) - epoch_extractor(mem_df,
                                                                                                       'outNoStim',
                                                                                                       memory, typ, lag)
        else:
            array_data = epoch_extractor(mem_df, 'outStimMem', memory, typ, lag).sum(axis=1) - epoch_extractor(
                mem_df, 'outNoStim', memory, typ, lag).sum(axis=1)
    elif epoch == 'Sac-Out-Diff':
        if typ != 'all_trials':
            array_data = epoch_extractor(saccad_df, 'outStimSac', saccade, typ, lag) - epoch_extractor(saccad_df,
                                                                                                       'outNoStim',
                                                                                                       saccade, typ,
                                                                                                       lag)
        else:
            array_data = epoch_extractor(saccad_df, 'outStimSac', saccade, typ, lag).sum(axis=1) - epoch_extractor(
                saccad_df, 'outNoStim', saccade, typ, lag).sum(axis=1)

    else:
        raise ValueError("Epoch not recognized")

    array_dim = array_data.shape
    if len(array_dim) > 2:
        return array_data
    else:
        return set_neg_to_zero(array_data)


def generate_lagged_all_epoch_dict(vis_df: pd.core.frame.DataFrame, mem_df: pd.core.frame.DataFrame,
                                   saccad_df: pd.core.frame.DataFrame, lag: int, typ: str, return_type: str,
                                   epoch_list: List[str] = None) -> dict:
    """
    Create a dictionary of all epochs
    :param vis_df: data set of all neurons in visual epoch
    :param mem_df: data set of all neurons in memory epoch
    :param saccad_df: data set of all saccade aligned neurons
    :param lag: time lag negative and positive for epochs
    :param typ: response type -> evoked, evokedFr, count, Fr, all_trials
    :param return_type: Specify either numpy.ndarray or pandas.dataFrame
    :param epoch_list: list of desired epochs to select
    :return: A dict of either numpyArray or pandas dataFrame
    """
    if epoch_list is None:

        epoch_list = ['Enc-In-NoStim', 'Mem-In-NoStim', 'Sac-In-NoStim',
                      'Enc-Out-NoStim', 'Mem-Out-NoStim', 'Sac-Out-NoStim',
                      'Enc-In-Stim', 'Mem-In-Stim', 'Sac-In-Stim',
                      'Enc-Out-Stim', 'Mem-Out-Stim', 'Sac-Out-Stim',
                      'Enc-In-Diff', 'Mem-In-Diff', 'Sac-In-Diff',
                      'Enc-Out-Diff', 'Mem-Out-Diff', 'Sac-Out-Diff']

    out_dict = {}
    if return_type == 'numpyArray':
        for epoch in epoch_list:
            out_dict[epoch] = generate_lagged_epochs(vis_df, mem_df, saccad_df, epoch, lag, typ)
    else:
        for epoch in epoch_list:
            out_dict[epoch] = pd.DataFrame(generate_lagged_epochs(vis_df, mem_df, saccad_df, epoch, lag, typ))
    return out_dict


def fill_dict(dicts, totalValues):
    modified_dict = dicts
    diffs = list(set(totalValues) - set(dicts.keys()))
    for x in diffs:
        modified_dict[x] = 0
    return modified_dict


def symmetrical(data):
    return (np.array(data) + np.array(data).T) / 2


# Base line start time
def base_line(x): return np.arange((x[0] - 1) - len(x), (x[0] - 1))


# get Mean of MCMC chains
def get_mean_over_all_chains(path, epoch, chain_number):
    res = np.array([np.array(pd.read_csv(path + epoch + "__" + str(y) + ".csv")) for y in chain_number]).mean(axis=0)
    return res


# A function to check significacy of a network measure
def check_significance(data, neuron, method):
    test_resualt = ttest_1samp(np.unique(data[data.neuron == neuron][method]), 0)
    return test_resualt[1]


# A function to compute error measure -- .95 confidence interval
def compute_error(data, neuron, method):
    values = np.unique(data[data.neuron == neuron][method])
    error_resualt = 1.96 * (np.std(values) / len(values))
    return error_resualt


# A function to pick color based on significancy of a network measure
def significance_color(data, neuron, method, thresh):
    defualtColor = 'rgba(204,204,204,1)'
    significantColor = 'rgba(222,45,38,0.8)'

    if check_significance(data, neuron, method) < thresh:
        return significantColor
    else:
        return defualtColor
