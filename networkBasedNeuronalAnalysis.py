import os
import time
import numpy as np
import pandas as pd
import tqdm

from argparse import ArgumentParser
from commons.tools.basicFunctions import (assembleData2, saccade_df, generate_lagged_all_epoch_dict)
from fitModel.fit_discrete_Hawkes import fit_model_discrete_time_network_hawkes_spike_and_slab
from fitModel.pre_processing import raw_neuronal_data_info_compute, split_epoch_condition
from fitModel.post_processing import network_info_writer, complete_df
from fitModel.GelmanRubin_convergence import compute_gelman_rubin_convergence
from fitModel.fitVlmc import fit_VLMC
from multiprocessing import Pool, cpu_count
from functools import partial

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

parser.add_argument('-s', '--sparsity', action='store',
                    default=0.5,  dest='sparsity', help='Initial sparsity of the network', type=float)

parser.add_argument('-S', '--self', action='store_true',
                    default=False, dest='self', help='Allow self connection')

parser.add_argument('-l', '--lag', action='store',
                    default=50, dest='lag', help='Impulse response lag', type=int)

parser.add_argument('-i', '--iter', action='store',
                    dest='iter', help='Number of MCMC iteration', type=int)

parser.add_argument('-c', '--chain', action='store',
                    dest='chain', help='Number of MCMC chain', type=int)

parser.add_argument('-v', '--version', action='version',
                    dest='', version='%(prog)s 0.1')

args = parser.parse_args()

# parallel computing config
pool = Pool(cpu_count())

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

start_time = time.time()

print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Start Variable Length Markov Chains Fitting ' +
      '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
allNeuronsVLMC = fit_VLMC(allNeurons, minTime, number_of_column_added)
saccad_df_VLMC = saccade_df(allNeuronsVLMC, 3000)

print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ VLMC fit completed in' +
      " %s hours " % round((time.time() - start_time) / (60*60), 4) + '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

sorted_total_spike_count = sorted(tuple([(x, sum(allNeurons[x].iloc[:, 0:(minTime - number_of_column_added)].sum()))
                                         for x in range(len(allNeurons))]), key=lambda tup: tup[1])
sorted_total_spike_count_vlmc = sorted(
    tuple([(x, sum(allNeuronsVLMC[x].iloc[:, 0:(minTime - number_of_column_added)].sum()))
           for x in range(len(allNeuronsVLMC))]), key=lambda tup: tup[1])

print('**** Total spike count across neurons sorted ascending ****')
for x in range(len(sorted_total_spike_count)):
    print('Neuron: ' + str(sorted_total_spike_count[x][0]) +
          ' Total Spike Count: ' + str(sorted_total_spike_count[x][1]))

print('**** Total VLMC fitted spike count across neurons sorted ascending ****')
for x in range(len(sorted_total_spike_count_vlmc)):
    print('Neuron: ' + str(sorted_total_spike_count_vlmc[x][0]) +
          ' Total Spike Count: ' + str(sorted_total_spike_count_vlmc[x][1]))

all_neuron_dict_list = [allNeurons, allNeuronsVLMC]
all_saccade_dict_list = [saccad_df, saccad_df_VLMC]
parrent_write_paths = ['Raw', 'VLMC']
writeArg = args.write


for i in range(len(parrent_write_paths)):

    neuronSetTime = time.time()
    print('************************************ Start Neuronal Network Computation for ' + parrent_write_paths[i] +
          ' neuron sets. ************************************')

    args.write = writeArg + parrent_write_paths[i] + '/'

    # Epoch discriminated spike dict data

    firingRate = generate_lagged_all_epoch_dict(all_neuron_dict_list[i], all_saccade_dict_list[i], 0, 'evokedFr',
                                                'pd.DataFrame')
    spikeCounts = generate_lagged_all_epoch_dict(all_neuron_dict_list[i], all_saccade_dict_list[i], 0, 'evoked',
                                                 'pd.DataFrame')
    neuronalData = generate_lagged_all_epoch_dict(all_neuron_dict_list[i], all_saccade_dict_list[i], 0, 'evoked',
                                                  'numpyArray')

    # network hyper parameter definition

    network_hypers = {"p": args.sparsity, "allow_self_connections": args.self}

    # get neural data information
    start_time = time.time()
    raw_neuronal_data_info_compute(all_neuron_dict_list[i], args)
    split_epoch_condition(firingRate, spikeCounts, args)
    print('************ Raw data ingestion completed in' +
          " %s minutes " % round((time.time() - start_time) / 60, 4) + ' ************')

    # Extract data and periods
    period, data = zip(*neuronalData.items())

    # create fit_par partial function
    fit_par = partial(fit_model_discrete_time_network_hawkes_spike_and_slab, *[args, network_hypers, period, data])

    start_time = time.time()
    list(tqdm.tqdm(pool.imap(fit_par, list(range(args.chain))), total=args.chain))
    print('************ MCMC estimation completed in'
          + " %s hours " % round((time.time() - start_time) / (60 * 60), 4) + 'and for each chain in '
          + " %s hours. " % round((time.time() - start_time) / (60 * 60 * args.chain), 4) + ' ****')

    # Gelman-Rubin convergence statistics
    start_time = time.time()
    compute_gelman_rubin_convergence(args)
    print('************ GR statistics computation completed in' +
          " %s minutes " % round((time.time() - start_time) / 60, 4) + '************')

    # create a list of all networks
    file_names = os.listdir(args.write + 'Firing Rate/')
    referencePath = args.write + '/MCMCResults/McMcValues/'

    start_time = time.time()
    for c in range(args.chain):
        print('**** Start writing Network information of chain ' + str(c) + ' ****')

        # create fit_par partial function
        pearson_par = partial(network_info_writer, *[args, referencePath, 0.75, 'pearson', c])
        mutual_par = partial(network_info_writer, *[args, referencePath, 0.75, 'mutualScore', c])
        ncs_par = partial(network_info_writer, *[args, referencePath, 0.75, 'ncs', c])
        hawkes_par = partial(network_info_writer, *[args, referencePath, 0.75, 'hawkes', c])

        # pearson
        list(map(pearson_par, file_names))
        # mutual information
        list(map(mutual_par, file_names))
        # ncs
        list(map(ncs_par, file_names))
        # hawkes
        list(map(hawkes_par, file_names))

    print('**** Total Network information ingestion completed in'
          + " %s hours " % round((time.time() - start_time) / (60 * 60), 4) + 'and for each chain in '
          + " %s hours. " % round((time.time() - start_time) / (60 * 60 * args.chain), 4) + ' ****')

    # Assemble all network data for all chain in one file
    start_time = time.time()
    network_data_path = args.write + '/NetworkInformations/'
    files = list(filter(lambda x: (x.endswith("__0.csv") & (not x.startswith("Raw")) &
                                   (not x.startswith("thresh"))), os.listdir(network_data_path + "Correlation")))
    files = list(map(lambda x: x.split("__")[0], files))

    method_list = ['Correlation', 'MutualInformation', 'Ncs', 'Hawkes']
    all_df_list = [complete_df(network_data_path, x, y, z) for x in method_list for y in files for z in
                   range(args.chain)]
    all_data = pd.concat(all_df_list)

    all_data.to_csv(network_data_path + "all_chain_network_info.csv", index=False)
    print('************ Concatenating all network information completed in' +
          " %s seconds " % round((time.time() - start_time), 2) + '************')

    print('************************************ ' + parrent_write_paths[i] +
          ' neuron sets computations' + 'completed in' +
          " %s hours " % round((time.time() - neuronSetTime) / (60 * 60), 4) +
          ' ************************************')

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% All evaluations completed in ' +
      " %s hours " % round((time.time() - start_time_initial) / (60 * 60), 4) +
      ' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
