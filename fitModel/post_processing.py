import os
import pandas as pd
import numpy as np
import networkx as nx

from commons.selectivityMethods.general_information_calculator import info, set_threshold
from commons.tools import graph_processing as gp
from commons.tools.basicFunctions import extract_from_dict, normalize


def network_info_writer(args, referencePath, quant, method, chain, filename):

    global infoPath, thresh_network, network
    readPath = args.write + 'Spike Count/'
    writePath = args.write + 'NetworkInformations/'

    if not os.path.exists(writePath):
        os.makedirs(writePath)

    data = pd.read_csv(readPath + filename)

    referenceFiles = list(filter(lambda x: x.endswith("__" + str(chain) + ".npy"), os.listdir(referencePath)))
    referenceFileName = list(filter(lambda x: (x.startswith(filename.split(".")[0]) & x.endswith("__" + str(chain) +".npy")), referenceFiles))[0]
    referenceData = np.load(referencePath + referenceFileName)
    referenceDataShape = referenceData.shape[0]
    # select data after burn-in and the quantf as estimated values
    referenceDataMean = np.mean(referenceData[referenceDataShape // 2:, :, :],  axis=0)
    referenceDataQuantiled = np.quantile(referenceData[referenceDataShape//2:, :, :], quant, axis=0)


    # Read correlation, mutual information and correlation p_values
    if method == 'pearson':

        infoPath = writePath + 'Correlation/'

        if not os.path.exists(infoPath):
            os.makedirs(infoPath)

        network = info(datax=data, datay=data, method='pearson')
        # Set threshold in connectivity matrix based on average p_values
        thresh_network = set_threshold(network, referenceDataQuantiled)

    elif method == 'mutual':

        infoPath = writePath + 'MutualInformation/'

        if not os.path.exists(infoPath):
            os.makedirs(infoPath)

        network = info(datax=data, datay=data, method='mutual')
        # Set threshold in connectivity matrix based on average p_values
        thresh_network = set_threshold(network, referenceDataQuantiled)

    elif method == 'mutualScore':
        infoPath = writePath + 'MutualInformation/'

        if not os.path.exists(infoPath):
            os.makedirs(infoPath)

        network = info(datax=data, datay=data, method='mutualScore')
        # Set threshold in connectivity matrix based on average p_values
        thresh_network = set_threshold(network, referenceDataQuantiled)

    elif method == 'hawkes':
        infoPath = writePath + 'Hawkes/'

        if not os.path.exists(infoPath):
            os.makedirs(infoPath)

        network = referenceDataMean
        # Set threshold in connectivity matrix based on average p_values
        thresh_network = set_threshold(network, referenceDataQuantiled)

    labels = list(map(lambda x: 'N' + str(x + 1), range(thresh_network.shape[1])))
    labels = dict(zip(np.arange(0, len(labels)), labels))
    G = nx.from_numpy_matrix(thresh_network)
    G = nx.relabel_nodes(G, labels, copy=False)
    # Write graphs
    nx.write_gml(G, infoPath + str(filename).split('.')[0] + "__" + str(chain) + ".gml")
    # graph infos
    # Average shortest path
    if nx.is_connected(G):
        asp = nx.average_shortest_path_length(G)
    else:
        asp = np.mean([nx.average_shortest_path_length(g) for g in nx.connected_component_subgraphs(G)])
    # Clustering coefficient
    co = nx.average_clustering(G)
    # smallworldness index 1-Omega: values near zero indicates small world property,
    # values near -1 indicate lattice shape, value near to 1 indicate random graph
    # smallworldness index 2-Sigma: values greater than 1 indicate small world value property,
    # specifically when its greater or equal than 3
    sigma, omega = gp.small_world_index(G, niter=100, nrand=16)
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

    graph_centrality_measures = {'closeness_centrality': close, 'eigenvector_centrality': eigen,
                                 'betweenness_centrality': between, 'harmonic_centrality': harmon,
                                 'load_centrality': loads, 'degrees': degrees,
                                 'average_shortest_path': asp, 'average_clustering': co,
                                 'omega': omega, 'sigma': sigma, 'density': dens}
    centrality_data_frame = pd.DataFrame.from_dict(graph_centrality_measures)
    centrality_data_frame['indexNumber'] = centrality_data_frame.index.to_series().str.split('N').str[-1].astype(
        int).sort_values()
    centrality_data_frame['neuron'] = centrality_data_frame.index
    centrality_data_frame.sort_values(['indexNumber'], ascending=True, inplace=True)
    centrality_data_frame.drop('indexNumber', 1, inplace=True)
    # Writing network info data
    centrality_data_frame.to_csv(infoPath + str(filename).split('.')[0] + "__" + str(chain) + ".csv", index=False)
    # Writing Raw data
    pd.DataFrame(network, columns=labels).to_csv(infoPath + 'Raw-' + str(filename).split('.')[0] + "__" + str(chain) + ".csv", index=False)
    pd.DataFrame(thresh_network, columns=labels).to_csv(infoPath + 'thresh-' + str(filename).split('.')[0] + "__" + str(chain) + ".csv", index=False)


def complete_df(path, method, epoch, chain):
    df = pd.read_csv(path + method + '/' + epoch + "__" + str(chain) + ".csv")
    df['method'] = method
    epoch_name_full = str(epoch).split('.')[0]
    epoch_name_sp = epoch_name_full.split('-')
    if(epoch_name_sp[0] == 'Enc'):
        epoch_name_sp[0] = 'Vis'
    if(len(epoch_name_sp) > 2):
           epoch_name = epoch_name_sp[0] + '-' + epoch_name_sp[2]
    else:
        epoch_name = epoch_name_sp[0] + '-' + epoch_name_sp[1]
    df['epoch'] = epoch_name
    df['chain'] = chain
    return df








