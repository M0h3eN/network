import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from commons.plotRelatedFunctions.plot_network import plot_connectivity, plot_all_cen, graph_indexes_grouped_bar

parser = ArgumentParser(description='This is a Python program for plotting the final result')
parser.add_argument('-w', '--write', action='store',
                    dest='write', help='Output directory')

args = parser.parse_args()


# Write connectivity heat maps
lis = ['Correlation', 'MutualInformation', 'Hawkes']

epochs_all = list(filter(lambda x: (x.startswith("Raw") & x.endswith("csv")), os.listdir(args.write+lis[0])))
epochs = list(np.unique(list(map(lambda x: x.split("__")[0], epochs_all))).tolist())
chain_number = list(np.sort(np.unique(list(map(lambda x: int(x.split("__")[1].split(".")[0]), epochs_all)))).tolist())


# corr
plot_connectivity(lis[0], epochs, args.write, chain_number,  -0.75, 0.75)
# mutt
plot_connectivity(lis[1], epochs, args.write, chain_number, 0, 0.5)
# hawkes
plot_connectivity(lis[2], epochs, args.write, chain_number, 0, 0.5)


# Write centrality
all_data = pd.read_csv(args.write + "all_chain_network_info.csv")
centrality_list = ['closeness_centrality', 'eigenvector_centrality', 'betweenness_centrality', 'harmonic_centrality','load_centrality']
centrality_write_path = args.write + '/' + 'CentralityPlots' + '/'

if not os.path.exists(centrality_write_path):
    os.makedirs(centrality_write_path)

for cen in centrality_list:
    each_cen_path = centrality_write_path + '/' + cen.split('_')[0] + '/'

    if not os.path.exists(each_cen_path):
        os.makedirs(each_cen_path)

    plot_all_cen(all_data, cen, each_cen_path)

# Write network indexes

noStim = all_data[all_data.epoch.isin(["Vis-NoStim", "Sac-NoStim", "Mem-NoStim"])]
stim = all_data[all_data.epoch.isin(["Vis-Stim", "Sac-Stim", "Mem-Stim"])]
diff = all_data[all_data.epoch.isin(["Vis-diff", "Sac-diff", "Mem-diff"])]

network_indexes_write_path = args.write + 'NetworkIndexes/'

if not os.path.exists(network_indexes_write_path):
    os.makedirs(network_indexes_write_path)

graph_indexes_grouped_bar(noStim, network_indexes_write_path + "leg", True)
graph_indexes_grouped_bar(noStim, network_indexes_write_path + "no-stim-index", False)
graph_indexes_grouped_bar(stim, network_indexes_write_path + "stim-index", False)
graph_indexes_grouped_bar(diff, network_indexes_write_path + "diff-index", False)
