import os
import pandas as pd
import numpy as np
import networkx as nx

from bokeh.io import export_png
from pymongo import MongoClient
from commons.tools.basicFunctions import (computerFrAll, computerFrAllDict, computeSpikeCountALLDict, saccade_df,\
                                          extract_from_dict, normalize)
from commons.selectivityMethods.mi import computeMI, plotScat, plotBar
from commons.plotRelatedFunctions.FiringRateRelatedPlotFunctions import createPlotDF, plotFun
from commons.selectivityMethods.general_information_calculator import info, set_threshold


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
    pre_proccess_db = client.neuralDataInfo

    # inserting Raw data
    # TODO must create an schema for raw data to insert many based on time or trials

    # inserting Firing rates
    firingRate = computerFrAllDict(data)
    pre_proccess_db['firing_rate'].insert_many(firingRate)

    # saving PSTHs charts
    for iterator in range(len(saccade)):
        export_png(plotFun(createPlotDF(DF=visualAndDelay, DF2=data[0], period='vis', ind=iterator),
                           createPlotDF(DF=saccade, DF2=data[iterator], period='sac', ind=iterator)),
                   filename=writePathFr + str(iterator) + '.png')

    # inserting Spike Counts
    spikeCounts = computeSpikeCountALLDict(data)
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


def network_info_writer(args, filename, method):

    readPath = args.write + 'Firing Rate/'
    writePath = args.write + 'NetworkInformations/'

    if not os.path.exists(writePath):
        os.makedirs(writePath)

    data = pd.read_csv(readPath + filename)
    # Read correlation, mutual information and correlation p_values
    if method == 'pearson':

        infoPath = writePath + 'Correlation/'

        if not os.path.exists(infoPath):
            os.makedirs(infoPath)

        network, p_values = info(data=data, method='pearson')
        # Set threshold in connectivity matrix based on average p_values
        thresh_network = set_threshold(network, p_values)

    elif method == 'mutual':
        infoPath = writePath + 'MutualInformation/'

        if not os.path.exists(infoPath):
            os.makedirs(infoPath)

        network, p_values = info(data=data, method='pearson')
        # Set threshold in connectivity matrix based on average p_values
        thresh_network = set_threshold(network, p_values)

    labels = list(map(lambda x: 'N' + str(x + 1), range(thresh_network.shape[1])))
    labels = dict(zip(np.arange(0, len(labels)), labels))
    G = nx.from_numpy_matrix(thresh_network)
    G = nx.relabel_nodes(G, labels, copy=False)
    # graph infos
    # Average shortest path
    asp = nx.average_shortest_path_length(G)
    # Clustering coefficient
    co = nx.average_clustering(G)
    # smallworldness index 1-Omega: values near zero indicates small world property,
    # values near -1 indicate lattice shape, value near to 1 indicate random graph
    omega = nx.algorithms.omega(G)
    # smallworldness index 2-Sigma: values greater than 1 indicate small world value property,
    # specifically when its greater or equal than 3
    sigma = nx.algorithms.sigma(G)
    # density
    dens = nx.density(G)
    # degree distribution
    degrees = dict([(n, G.degree(n)) for n in G.nodes()])
    # centrality measures
    close = nx.centrality.closeness_centrality(G)
    close = dict(zip(extract_from_dict(close)[0], normalize(extract_from_dict(close)[1], 0, 1)))
    eigen = nx.centrality.eigenvector_centrality(G)
    eigen = dict(zip(extract_from_dict(eigen)[0], normalize(extract_from_dict(eigen)[1], 0, 1)))
    between = nx.centrality.betweenness_centrality(G)
    between = dict(zip(extract_from_dict(between)[0], normalize(extract_from_dict(between)[1], 0, 1)))
    harmon = nx.centrality.harmonic_centrality(G)
    harmon = dict(zip(extract_from_dict(harmon)[0], normalize(extract_from_dict(harmon)[1], 0, 1)))
    loads = nx.centrality.load_centrality(G)
    loads = dict(zip(extract_from_dict(loads)[0], normalize(extract_from_dict(loads)[1], 0, 1)))
    infoc = nx.centrality.information_centrality(G)
    infoc = dict(zip(extract_from_dict(infoc)[0], normalize(extract_from_dict(infoc)[1], 0, 1)))
    cf_between = nx.centrality.current_flow_betweenness_centrality(G)
    cf_between = dict(zip(extract_from_dict(cf_between)[0], normalize(extract_from_dict(cf_between)[1], 0, 1)))
    cf_close = nx.centrality.current_flow_closeness_centrality(G)
    cf_close = dict(zip(extract_from_dict(cf_close)[0], normalize(extract_from_dict(cf_close)[1], 0, 1)))
    graph_centrality_measures = {'closeness_centrality': close, 'eigenvector_centrality': eigen,
                                 'betweenness_centrality': between, 'harmonic_centrality': harmon,
                                 'load_centrality': loads, 'information_centrality': infoc,
                                 'current_flow_betweenness_centrality': cf_between,
                                 'current_flow_closeness_centrality': cf_close, 'degrees': degrees,
                                 'average_shortest_path': asp, 'average_clustering': co,
                                 'omega': omega, 'sigma': sigma, 'density': dens}
    centrality_data_frame = pd.DataFrame.from_dict(graph_centrality_measures)
    centrality_data_frame['indexNumber'] = centrality_data_frame.index.to_series().str.split('N').str[-1].astype(
        int).sort_values()
    centrality_data_frame['neuron'] = centrality_data_frame.index
    centrality_data_frame.sort_values(['indexNumber'], ascending=True, inplace=True)
    centrality_data_frame.drop('indexNumber', 1, inplace=True)
    centrality_data_frame.to_csv(infoPath + filename, index=False)



