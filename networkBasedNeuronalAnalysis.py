import os
import time
import numpy as np
import pandas as pd
import tqdm
import logging

from argparse import ArgumentParser
from commons.tools.basicFunctions import (assembleData2, saccade_df, generate_lagged_all_epoch_dict,
                                          evaluate_neuron_significance_based_on_fixation)
from fitModel.fit_discrete_Hawkes import fit_model_discrete_time_network_hawkes_spike_and_slab
from fitModel.pre_processing import split_epoch_condition, firing_rate_writer
from fitModel.post_processing import network_info_writer, complete_df
from fitModel.fitVlmc import fit_VLMC
from multiprocessing import Pool
from functools import partial

log_path = os.path.dirname(os.path.abspath(__file__)) + '/log'

if not os.path.exists(log_path):
    os.makedirs(log_path)

logging.basicConfig(filename=log_path + '/' + 'app.log', filemode='w',
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = ArgumentParser(description='This is a Python program for analysis on network of neurons to '
                                    'detect functional connectivity between neurons')

parser.add_argument('-d', '--data', action='store',
                    dest='data', help='Raw data directory')

# parser.add_argument('-H', '--host', action='store',
#                     default='localhost', dest='host', help='MongoDB host name')
#
# parser.add_argument('-p', '--port', action='store',
#                     default='27017', dest='port', help='MongoDB port number')

parser.add_argument('-w', '--write', action='store',
                    dest='write', help='Output directory')

parser.add_argument('-s', '--sparsity', action='store',
                    default=0.5, dest='sparsity', help='Initial sparsity of the network', type=float)

parser.add_argument('-S', '--self', action='store_true',
                    default=False, dest='self', help='Allow self connection')

parser.add_argument('-l', '--lag', action='store',
                    default=50, dest='lag', help='Impulse response lag', type=int)

parser.add_argument('-i', '--iter', action='store',
                    dest='iter', help='Number of MCMC iteration', type=int)

parser.add_argument('-c', '--chain', action='store',
                    dest='chain', help='Number of MCMC chain', type=int)

parser.add_argument('-p', '--pool', action='store',
                    dest='pool', help='Number of cpu', type=int)

parser.add_argument('-sp', '--spiking', action='store',
                    dest='spiking', help='Spiking data', type=str)

parser.add_argument('-v', '--version', action='version',
                    dest='', version='%(prog)s 0.1')

args = parser.parse_args()

# parallel computing config
pool = Pool(args.pool)

spiking_data = args.spiking
# Data preparation
start_time_initial = time.time()

start_time = time.time()
logging.info("Start assembling neurons from raw data")
# read all neurons
dirr = os.fsencode(args.data)
allNeurons = assembleData2(dirr)
# align end date
minTime = np.min([allNeurons[x].shape[1] for x in range(len(allNeurons))])
# Find total spike count across all neurons
number_of_column_added = len(list(filter(lambda x: not x.startswith("T"), allNeurons.get(0).columns)))
logging.info('Neurons assemblies completed in:' + " %s minutes " % round((time.time() - start_time) / 60, 4))

sorted_total_spike_count = sorted(tuple([(x, sum(allNeurons[x].iloc[:, 0:(minTime - number_of_column_added)].sum()))
                                         for x in range(len(allNeurons))]), key=lambda tup: tup[1])
logging.info('Total spike count across neurons sorted ascending:')
for x in range(len(sorted_total_spike_count)):
    logging.info('Neuron: ' + str(sorted_total_spike_count[x][0]) +
                 ' ---> Total Spike Count: ' + str(sorted_total_spike_count[x][1]))

epoch_list = ['Enc-In-NoStim', 'Mem-In-NoStim', 'Sac-In-NoStim']
writeArg = args.write

if spiking_data == 'VLMC':

    logging.info('Start Variable Length Markov Chains Fitting')
    start_time = time.time()

    allNeurons = fit_VLMC(allNeurons, minTime, number_of_column_added, pool)
    saccad_df = saccade_df(allNeurons, 3000)
    logging.info('VLMC fit completed in:' + " %s minutes " % round((time.time() - start_time) / 60, 4))
    args.write = writeArg + spiking_data + '/'

else:

    saccad_df = saccade_df(allNeurons, 3000)
    args.write = writeArg + spiking_data + '/'

if not os.path.exists(args.write):
    os.makedirs(args.write)

logging.info("Start evaluating significant neurons")
start_time = time.time()

all_neurons_vis_filtered, all_neurons_mem_filtered, all_neurons_sac_filtered, sig_indexes = \
    evaluate_neuron_significance_based_on_fixation(allNeurons, saccad_df, 800, 1000)

saccad_df_filtered = saccade_df(all_neurons_vis_filtered, 3000)

sig_indexes.to_csv(args.write + '.csv', index=False)

logging.info("Total number of " + "%s neuron" % len(all_neurons_vis_filtered) +
             " out of " + "%s neuron" % len(allNeurons) + " selected as significant in Visual epoch.")

logging.info("Total number of " + "%s neuron" % len(all_neurons_mem_filtered) +
             " out of " + "%s neuron" % len(allNeurons) + " selected as significant in Memory epoch.")

logging.info("Total number of " + "%s neuron" % len(all_neurons_sac_filtered) +
             " out of " + "%s neuron" % len(allNeurons) + " selected as significant in Saccade epoch.")

allNeuronsVis = all_neurons_vis_filtered
allNeuronsMem = all_neurons_mem_filtered
saccad_df = saccad_df_filtered

sorted_total_spike_count_vis = sorted(
    tuple([(x, sum(allNeuronsVis[x].iloc[:, 0:(minTime - number_of_column_added)].sum()))
           for x in range(len(allNeuronsVis))]), key=lambda tup: tup[1])
sorted_total_spike_count_mem = sorted(
    tuple([(x, sum(allNeuronsMem[x].iloc[:, 0:(minTime - number_of_column_added)].sum()))
           for x in range(len(allNeuronsMem))]), key=lambda tup: tup[1])
sorted_total_spike_count_sac = sorted(
    tuple([(x, sum(all_neurons_sac_filtered[x].iloc[:, 0:(minTime - number_of_column_added)].sum()))
           for x in range(len(all_neurons_sac_filtered))]), key=lambda tup: tup[1])

logging.info('Total spike count across significant neurons sorted ascending ------> VISUAL:')
for x in range(len(sorted_total_spike_count_vis)):
    logging.info('Neuron: ' + str(sorted_total_spike_count_vis[x][0]) +
                 ' ---> Total Spike Count: ' + str(sorted_total_spike_count_vis[x][1]))

logging.info('Total spike count across significant neurons sorted ascending ------> MEMORY:')
for x in range(len(sorted_total_spike_count_mem)):
    logging.info('Neuron: ' + str(sorted_total_spike_count_mem[x][0]) +
                 ' ---> Total Spike Count: ' + str(sorted_total_spike_count_mem[x][1]))

logging.info('Total spike count across significant neurons sorted ascending ------> SACCADE:')
for x in range(len(sorted_total_spike_count_sac)):
    logging.info('Neuron: ' + str(sorted_total_spike_count_sac[x][0]) +
                 ' ---> Total Spike Count: ' + str(sorted_total_spike_count_sac[x][1]))

logging.info('Neurons significance check completed in:' + " %s minutes " % round((time.time() - start_time) / 60, 4))

neuronSetTime = time.time()
logging.info('Start Neuronal Network Computation for ' + "%s neuron set." % spiking_data)

# Epoch discriminated spike dict data

firingRate = generate_lagged_all_epoch_dict(allNeuronsVis, allNeuronsMem, saccad_df, 0, 'evokedFr',
                                            'pd.DataFrame', epoch_list)
spikeCounts = generate_lagged_all_epoch_dict(allNeuronsVis, allNeuronsMem, saccad_df, 0, 'evoked',
                                             'pd.DataFrame', epoch_list)
neuronalData = generate_lagged_all_epoch_dict(allNeuronsVis, allNeuronsMem, saccad_df, 0, 'evoked',
                                              'numpyArray', epoch_list)

# network hyper parameter definition

network_hypers = {"p": args.sparsity, "allow_self_connections": args.self}

# get neural data information
start_time = time.time()
firing_rate_writer(allNeurons, args)
split_epoch_condition(firingRate, spikeCounts, args)
logging.info('Raw data ingestion completed in:' + " %s minutes " % round((time.time() - start_time) / 60, 4))

# Extract data and periods
period, data = zip(*neuronalData.items())

# create fit_par partial function
fit_par = partial(fit_model_discrete_time_network_hawkes_spike_and_slab, *[args, network_hypers, period, data])

start_time = time.time()
list(tqdm.tqdm(pool.imap(fit_par, list(range(args.chain))), total=args.chain))
logging.info('MCMC estimation completed in'
             + " %s hours " % round((time.time() - start_time) / (60 * 60), 4) + 'and for each chain in'
             + " %s hours. " % round((time.time() - start_time) / (60 * 60 * args.chain), 4))

# Gelman-Rubin convergence statistics
# start_time = time.time()
# compute_gelman_rubin_convergence(args)
# logging.info('GR statistics computation completed in' +
#              " %s minutes " % round((time.time() - start_time) / 60, 4))

# create a list of all networks
file_names = os.listdir(args.write + 'Firing Rate/')
referencePath = args.write + '/MCMCResults/McMcValues/'

start_time = time.time()
for c in range(args.chain):
    logging.info('Start writing Network information of chain ' + "%s " % c)

    # create fit_par partial function
    pearson_par = partial(network_info_writer, *[args, referencePath, 0.75, 'pearson', c, pool])
    mutual_par = partial(network_info_writer, *[args, referencePath, 0.75, 'mutualScore', c, pool])
    ncs_par = partial(network_info_writer, *[args, referencePath, 0.75, 'ncs', c, pool])
    hawkes_par = partial(network_info_writer, *[args, referencePath, 0.75, 'hawkes', c, pool])

    # pearson
    list(map(pearson_par, file_names))
    # mutual information
    list(map(mutual_par, file_names))
    # ncs
    list(map(ncs_par, file_names))
    # hawkes
    list(map(hawkes_par, file_names))

logging.info('Total Network information ingestion completed in'
             + " %s hours " % round((time.time() - start_time) / (60 * 60), 4) + 'and for each chain in'
             + " %s hours. " % round((time.time() - start_time) / (60 * 60 * args.chain), 4))

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
logging.info('Concatenating all network information completed in' +
             " %s seconds " % round((time.time() - start_time), 2))

logging.info('' + spiking_data + ' neuron sets computations' + 'completed in' +
             " %s hours " % round((time.time() - neuronSetTime) / (60 * 60), 4))
