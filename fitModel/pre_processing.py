import os
from bokeh.io import export_png

from pymongo import MongoClient
from dataImport.commons.basicFunctions import computerFrAll, computerFrAllDict, computeSpikeCountDict, createPlotDF, \
    plotFun, saccade_df
from dataImport.selectivityMethods.mi import computeMI, plotScat, plotBar


def raw_neuronal_data_info_compute(data, args):

    # path config
    writePathFr = args.write + 'Firing_Rate' + '/'
    writePathMi = args.write + 'Mutual_Information' + '/'

    if not (os.path.exists(writePathFr) & os.path.exists(writePathMi)):
        os.makedirs(writePathFr)
        os.makedirs(writePathMi)

    # separating spikes epochs
    visualAndDelay = computerFrAll(data, 'vis')
    saccade = computerFrAll(data, 'saccade')

    # Mongo Configs
    client = MongoClient("mongodb://" + args.host + ':' + args.port)
    pre_proccess_db = client.neuronalDataInfo

    # inserting Raw data
    #TODO must create an schema for raw data to insert many based on time or trials

    # inserting Firing rates
    firingRate = computerFrAllDict(data)
    pre_proccess_db['firing_rate'].insert_many(firingRate)

    # saving PSTHs charts
    for iterator in range(len(saccade)):
        export_png(plotFun(createPlotDF(DF=visualAndDelay, DF2=data[0], period='vis', ind=iterator),
                           createPlotDF(DF=saccade, DF2=data[iterator], period='sac', ind=iterator)),
                   filename=writePathFr + str(iterator) + '.png')


    # inserting Spike Counts
    spikeCounts = computeSpikeCountDict(data)
    pre_proccess_db['spike_count'].insert_many(spikeCounts)

    # inserting mutual information

    saccade_data_set = saccade_df(data)
    mivaluesDict = dict(Stim=computeMI(data, saccade_data_set, 'Stim').to_dict('list'),
                        NoStim=computeMI(data, saccade_data_set, 'NoStim').to_dict('list'))

    pre_proccess_db['mutual_information'].insert_one(mivaluesDict)

    # saving Mutual information charts

    mivaluesNoStim = computeMI(data, saccade_data_set, "noStim")
    mivaluesStim = computeMI(data, saccade_data_set, "withStim")

    export_png(plotScat(mivaluesNoStim), filename=writePathMi + 'scatNoStim.png')
    export_png(plotScat(mivaluesStim), filename=writePathMi + 'scatWithStim.png')

    plotBar(mivaluesNoStim, writePathMi + 'barNo')
    plotBar(mivaluesStim, writePathMi + 'barWithStim')



