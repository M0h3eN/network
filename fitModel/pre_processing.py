import os
import pandas as pd

from bokeh.io import export_png
from pymongo import MongoClient
from commons.tools.basicFunctions import (computerFrAll, computerFrAllDict, computerFrAllDataFrame,
                                          computeSpikeCountALLDict, saccade_df)
from commons.selectivityMethods.mi import computeMI, plotScat, plotBar
from commons.plotRelatedFunctions.FiringRateRelatedPlotFunctions import createPlotDF, plotFun



def raw_neuronal_data_info_compute(data, args):
    # path config
    writePathFr = args.write + 'Firing_Rate' + '/'
    writePathMi = args.write + 'Mutual_Information' + '/'
    fratePath = args.write + 'Firing Rate' + '/'

    if not (os.path.exists(writePathFr) & os.path.exists(writePathMi)):
        os.makedirs(writePathFr)
        os.makedirs(writePathMi)

    if not os.path.exists(fratePath):
        os.makedirs(fratePath)

    # separating spikes epochs
    visualAndDelay = computerFrAll(data, 'vis')
    saccade = computerFrAll(data, 'saccade')

    visualAndDelayDf = computerFrAllDataFrame(data, 'vis')
    saccadeDf = computerFrAllDataFrame(data, 'saccade')

    visualAndDelayDf.to_csv(index=False, path_or_buf=fratePath + 'VisAndMemFrAll' + '.csv')
    saccadeDf.to_csv(index=False, path_or_buf=fratePath + 'SaccadeFrAll' + '.csv')

    # Mongo Configs
    client = MongoClient("mongodb://" + args.host + ':' + args.port)
    pre_proccess_db = client.neuralDataInfo

    # inserting Raw data
    # TODO must create an schema for raw data to insert many based on time or trials

    # inserting Firing rates
    firingRate = computerFrAllDict(data)
    pre_proccess_db['firing_rate' + '_' + args.write.split('/')[-2]].insert_many(firingRate)

    # saving PSTHs charts
    for iterator in range(len(saccade)):
        export_png(plotFun(createPlotDF(DF=visualAndDelay, period='vis', ind=iterator),
                           createPlotDF(DF=saccade, period='sac', ind=iterator)),
                   filename=writePathFr + str(iterator) + '.png')

    # inserting Spike Counts
    spikeCounts = computeSpikeCountALLDict(data)
    pre_proccess_db['spike_count' + '_' + args.write.split('/')[-2]].insert_many(spikeCounts)

    # inserting mutual information

    saccade_data_set = saccade_df(data)
    mivaluesDict = dict(Stim=computeMI(data, saccade_data_set, 'Stim').to_dict('list'),
                        NoStim=computeMI(data, saccade_data_set, 'NoStim').to_dict('list'))

    pre_proccess_db['mutual_information' + '_' + args.write.split('/')[-2]].insert_one(mivaluesDict)

    # saving Mutual information charts

    mivaluesNoStim = computeMI(data, saccade_data_set, "Stim")
    mivaluesStim = computeMI(data, saccade_data_set, "NoStim")

    export_png(plotScat(mivaluesNoStim), filename=writePathMi + 'scatNoStim.png')
    export_png(plotScat(mivaluesStim), filename=writePathMi + 'scatWithStim.png')

    plotBar(mivaluesNoStim, writePathMi + 'barNo')
    plotBar(mivaluesStim, writePathMi + 'barWithStim')


def firing_rate_writer(data, args):
    # path config
    writePathMi = args.write + 'Mutual_Information' + '/'
    fratePath = args.write + 'FiringRateFlatted' + '/'

    if not os.path.exists(writePathMi):
        os.makedirs(writePathMi)

    if not os.path.exists(fratePath):
        os.makedirs(fratePath)

    visualAndDelayDf = computerFrAllDataFrame(data, 'vis')
    saccadeDf = computerFrAllDataFrame(data, 'saccade')

    visualAndDelayDf.to_csv(index=False, path_or_buf=fratePath + 'VisAndMemFrAll' + '.csv')
    saccadeDf.to_csv(index=False, path_or_buf=fratePath + 'SaccadeFrAll' + '.csv')


def split_epoch_condition(data_fr, data_sc, args):
    period_fr, data_fr = zip(*data_fr.items())
    period_sc, data_sc = zip(*data_sc.items())
    writePath = args.write

    if not os.path.exists(writePath):
        os.makedirs(writePath)

    tempPath = writePath
    fratePath = tempPath + 'Firing Rate' + '/'
    scountPath = tempPath + 'Spike Count' + '/'

    if not os.path.exists(fratePath):
        os.makedirs(fratePath)
    if not os.path.exists(scountPath):
        os.makedirs(scountPath)

    for per in range(len(period_fr)):
        firing_rate = pd.DataFrame(data_fr[per])
        spike_count = pd.DataFrame(data_sc[per])
        firing_rate.to_csv(index=False, path_or_buf=fratePath + period_fr[per] + '.csv')
        spike_count.to_csv(index=False, path_or_buf=scountPath + period_sc[per] + '.csv')
