import logging
import pandas as pd
import tqdm
import sys

from commons.selectivityMethods.general_information_calculator import compute_vlmc_for_each_trial
from commons.tools.basicFunctions import generatorTemp
from functools import partial
from pyvlmc.internals.vlmc import simulate

sys.setrecursionlimit(10000)

def fit_VLMC(all_neurons, min_time, number_of_column_added, pool):
    vlmc_fit_neurons = {}
    for iterat in range(len(all_neurons)):
        neuron = all_neurons[iterat]
        # Create partial function for iterating over each trial
        vlmc_partial = partial(compute_vlmc_for_each_trial, *[neuron, min_time, number_of_column_added])
        # fit VLMC for every trial on each neuron (all recording time considered)
        vlmc_list_pool = list(tqdm.tqdm(pool.imap(vlmc_partial, list(range(len(neuron)))), total=len(neuron)))
        # generate spikes based on fitted VLMC model
        simul_list = [pd.DataFrame(simulate(x, (min_time - number_of_column_added) + 1)).transpose() for x in
                      vlmc_list_pool]

        simul_data_frame = pd.concat(simul_list)
        simul_data_frame.columns = generatorTemp(simul_data_frame.shape[1])
        simul_data_frame = simul_data_frame.reset_index(drop=True)
        added_column_df = neuron.iloc[:, neuron.columns.get_loc("Cond"):neuron.shape[1]]\
            .reset_index(drop=True)

        vlmc_fitted_data_frame = pd.concat([simul_data_frame, added_column_df], axis=1)
        vlmc_fit_neurons[iterat] = vlmc_fitted_data_frame
        logging.info('Fitted VLMC for neuron ' + str(iterat) + '.')

    return vlmc_fit_neurons

