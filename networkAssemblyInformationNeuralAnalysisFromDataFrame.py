import os
import time
import pandas as pd

from argparse import ArgumentParser

from commons.selectivityMethods.general_information_calculator import complete_info_df_2

parser = ArgumentParser(description='This is a Python program for analysis on network of neurons to '
                                    'detect functional connectivity between neurons')

parser.add_argument('-d', '--data', action='store',
                    dest='data', help='Raw data directory')

parser.add_argument('-w', '--write', action='store',
                    dest='write', help='Output directory')

parser.add_argument('-v', '--version', action='version',
                    dest='', version='%(prog)s 0.1')

args = parser.parse_args()

# prepare data
start_time = time.time()

readPath = args.data
writePath = args.write

if not os.path.exists(writePath):
    os.makedirs(writePath)

spiking_data_list = ['Raw', 'VLMC']
method_list = ['Correlation', 'MutualInformation', 'Ncs', 'Hawkes']
epoch_list = ['Enc-In-NoStim', 'Mem-In-NoStim', 'Sac-In-NoStim']
chains_list = range(10)

# Flatten the lists
all_df_list_evoked = [complete_info_df_2(readPath, x, 'NetworkInformations', y, w, str(z))
                      for x in spiking_data_list
                      for y in method_list
                      for w in epoch_list
                      for z in chains_list]

all_data_evoked = pd.concat(all_df_list_evoked)

all_data_evoked.to_csv(writePath + 'AllInfoAssembled' + '.csv', index=False)

print('************ Network assembly information evaluator completed in' +
      " %s seconds " % round((time.time() - start_time), 2) + '************')

