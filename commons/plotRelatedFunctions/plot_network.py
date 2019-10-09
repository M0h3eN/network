from typing import List

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import pandas as pd
import os

from commons.tools.basicFunctions import symmetrical, get_mean_over_all_chains, compute_error, significance_color


def plot_network(G, figname):
    # Creates a copy of the graph
    H = G.copy()

    # crates a list for edges and for the weights
    edges, weights = zip(*nx.get_edge_attributes(H, 'weight').items())

    # calculates the degree of each node
    d = nx.degree(H)
    # creates list of nodes and a list their degrees that will be used later for their sizes
    nodelist, node_sizes = zip(*dict(d).items())

    # positions
    positions = nx.circular_layout(H)

    # Figure size
    plt.figure(figsize=(15, 15))

    # draws nodes
    nx.draw_networkx_nodes(H, positions, node_color='#DA70D6', nodelist=nodelist,
                           # the node size will be now based on its degree
                           node_size=tuple([x ** 3 for x in node_sizes]), alpha=0.8)

    # Styling for labels
    nx.draw_networkx_labels(H, positions, font_size=8,
                            font_family='sans-serif')

    # draws the edges
    nx.draw_networkx_edges(H, positions, edge_list=edges, style='solid',
                           # adds width=weights and edge_color = weights
                           # so that edges are based on the weight parameter
                           # edge_cmap is for the color scale based on the weight
                           # edge_vmin and edge_vmax assign the min and max weights for the width
                           width=weights, edge_color=weights, edge_cmap=plt.cm.PuRd,
                           edge_vmin=min(weights), edge_vmax=max(weights))

    dc = nx.degree_centrality(G)
    a1, a2 = zip(*dc.items())

    # displays the graph without axis
    plt.axis('off')
    # saves image
    plt.savefig(figname + '.svg', format='svg', dpi=1200)
    # , plt.show()
    return np.argmax(a2)


def plot_connectivity(method: str, epochs: List[str], parent_path: str, chain_number: List[int], minn: float, maxx: float) -> None:
    """
    Network Connectivity heatmap based on connectivity method
    :param method: Functional connectivity method - Correlation, MutualInformation, Hawkes
    :param epochs: all List of all combination of epochs and conditions
    :param parent_path: Estimated MCMC connectivity measures based on different method path
    :param chain_number: Ordered list of chain numbers
    :param minn: min value of connectivity measure
    :param maxx: max value of connectivity measure
    """
    for ep in epochs:

        read_path = parent_path + method + '/'
        write_path = parent_path + method + '/' + 'plot' + '/'

        if not os.path.exists(write_path):
            os.makedirs(write_path)

        rate = get_mean_over_all_chains(read_path, ep, chain_number)
        rate = np.array(symmetrical(rate))
        labels = list(map(lambda x: 'N' + str(x), np.arange(1, len(rate) + 1, 2)))
        plt.matshow(symmetrical(rate), vmin=minn, vmax=maxx, cmap=plt.cm.jet)
        ticks = np.arange(0, rate.shape[1], 2)
        plt.xticks(ticks, labels, rotation='vertical')
        plt.yticks(ticks, labels)
        plt.colorbar()
        plt.savefig(write_path + ep.split('.')[0] + '.svg', dpi=1000)
        plt.close()


def graph_centrality(data: pd.core.frame.DataFrame, cen_method: str, period: str, status: str, write_path: str) -> None:
    """
    Network centrality plot with significance check
    :param data: long format centrality pandas data frame
    :param cen_method: centrality method - closeness_centrality, eigenvector_centrality, betweenness_centrality,
    harmonic_centrality, load_centrality
    :param period: epochs - Vis, Mem, Sac
    :param status: conditions - Stim, NoStim, diff
    :param write_path: save directory
    """
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    DF = data[(data.epoch.str.strip() == period + '-' + status)].groupby(['neuron', 'chain'], as_index=False, sort=False).mean()

    neuron_list = list(data['neuron'][0:23])
    # Create Error bar measure based on .95 confidence interval
    error_bar_measure = list(map(lambda x: compute_error(DF, x, cen_method), neuron_list))
    colorList = list(map(lambda x: significance_color(DF, x, cen_method, 0.05), neuron_list))

    # Create grouped data frame on neuron and take average on all measures
    DF_grouped = DF.groupby(['neuron'], as_index=False, sort=False).mean()

    trace0 = go.Bar(
        x=DF_grouped.loc[:, 'neuron'],
        y=DF_grouped.loc[:, cen_method],
        error_y=dict(
            type='data',
            array=error_bar_measure,
            visible=True
        ),
        marker=dict(
            color=colorList),
    )

    data = [trace0]
    layout = go.Layout(
        width=1350,
        height=752,
        barmode='group')

    fig = go.Figure(data=data, layout=layout)

    # Axis Config
    fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=35))
    fig.update_yaxes(tickfont=dict(family='Rockwell', color='crimson', size=35))

    pio.write_image(fig, write_path + str(cen_method) + '__' + str(period) + '-' + str(status) + '.svg')


def information_assembly(data: pd.core.frame.DataFrame, info_method: str, period: str, status: str, write_path: str) -> None:
    """
    Network averaged information in neurons assemble based on correlation and mutual information with significance check
    :param data: long format information pandas data frame
    :param info_method: connectivity information method - correlation or mutual information
    :param period: epochs - Vis, Mem, Sac
    :param status: conditions - Stim, NoStim, diff
    :param write_path: save directory
    """
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    DF = data[(data.epoch.str.strip() == period + '-' + status) & (data.method.str.strip() == info_method)]


    trace0 = go.Bar(
        x=DF.loc[:, 'lag'],
        y=DF.loc[:, 'info'],
        error_y=dict(
            type='data',
            array=DF.loc[:, 'SEM'],
            visible=True
        ),
        marker=dict(
            color=DF.loc[:, 'colorList']),
    )

    data = [trace0]
    layout = go.Layout(
        width=1350,
        height=752,
        barmode='group')

    fig = go.Figure(data=data, layout=layout)

    # Axis Config
    fig.update_xaxes(tickangle=45, dtick=10, tickfont=dict(family='Rockwell', color='crimson', size=35))
    fig.update_yaxes(tickfont=dict(family='Rockwell', color='crimson', size=35))

    pio.write_image(fig, write_path + str(info_method) + '__' + str(period) + '-' + str(status) + '.svg')

    
def plot_all_cen(data: pd.core.frame.DataFrame, method: str, write_path) -> None:
    """
    Plot graph centrality for all epochs and conditions
    :param data: long format centrality pandas data frame
    :param method: centrality method - closeness_centrality, eigenvector_centrality, betweenness_centrality,
    harmonic_centrality, load_centrality
    :param write_path: save directory
    """
    graph_centrality(data, method, 'Vis', 'NoStim', write_path)
    graph_centrality(data, method, 'Mem', 'NoStim', write_path)
    graph_centrality(data, method, 'Sac', 'NoStim', write_path)

    graph_centrality(data, method, 'Vis', 'Stim', write_path)
    graph_centrality(data, method, 'Mem', 'Stim', write_path)
    graph_centrality(data, method, 'Sac', 'Stim', write_path)

    graph_centrality(data, method, 'Vis', 'diff', write_path)
    graph_centrality(data, method, 'Mem', 'diff', write_path)
    graph_centrality(data, method, 'Sac', 'diff', write_path)


def plot_all_info(data: pd.core.frame.DataFrame, method: str, write_path) -> None:
    """
    Plot graph centrality for all epochs and conditions
    :param data: long format centrality pandas data frame
    :param method: centrality method - closeness_centrality, eigenvector_centrality, betweenness_centrality,
    harmonic_centrality, load_centrality
    :param write_path: save directory
    """
    information_assembly(data, method, 'Vis', 'NoStim', write_path)
    information_assembly(data, method, 'Mem', 'NoStim', write_path)
    information_assembly(data, method, 'Sac', 'NoStim', write_path)

    information_assembly(data, method, 'Vis', 'Stim', write_path)
    information_assembly(data, method, 'Mem', 'Stim', write_path)
    information_assembly(data, method, 'Sac', 'Stim', write_path)

    information_assembly(data, method, 'Vis', 'diff', write_path)
    information_assembly(data, method, 'Mem', 'diff', write_path)
    information_assembly(data, method, 'Sac', 'diff', write_path)


def graph_indexes_grouped_bar(df: pd.core.frame.DataFrame, file_name: str, leg: bool) -> None:
    """

    :param df: Filtered centrality data frame based on conditions - Stim, NoStim, diff
    :param file_name: file name to write
    :param leg: whether legend is visible or not
    """
    filt = lambda st, data_list: list(filter(lambda x: st in x, data_list))[0]
    columnList = ['average_shortest_path', 'average_clustering', 'density', 'sigma']

    colls = df.loc[:, 'epoch'].unique()
    x_axis_names = [filt("Vis", colls), filt("Mem", colls), filt("Sac", colls)]

    trace1 = go.Bar(
        x=list(map(lambda x: x.split("-")[0], x_axis_names)),
        y=list(map(lambda x: df[(df.epoch == x)][columnList[0]].mean(), x_axis_names)),
        name=columnList[0].replace("_", " "),
        error_y=dict(
            type='data',
            array=list(map(lambda x: 1.96 * (
                        df[(df.epoch == x)][columnList[0]].std() / df[(df.epoch == x)][columnList[0]].size),
                           x_axis_names)),
            visible=True
        )
    )
    trace2 = go.Bar(
        x=list(map(lambda x: x.split("-")[0], x_axis_names)),
        y=list(map(lambda x: df[(df.epoch == x)][columnList[1]].mean(), x_axis_names)),
        name=columnList[1].replace("_", " "),
        error_y=dict(
            type='data',
            array=list(map(lambda x: 1.96 * (
                        df[(df.epoch == x)][columnList[1]].std() / df[(df.epoch == x)][columnList[1]].size),
                           x_axis_names)),
            visible=True
        )
    )
    trace3 = go.Bar(
        x=list(map(lambda x: x.split("-")[0], x_axis_names)),
        y=list(map(lambda x: df[(df.epoch == x)][columnList[2]].mean(), x_axis_names)),
        name=columnList[2].replace("_", " "),
        error_y=dict(
            type='data',
            array=list(map(lambda x: 1.96 * (
                        df[(df.epoch == x)][columnList[2]].std() / df[(df.epoch == x)][columnList[2]].size),
                           x_axis_names)),
            visible=True
        )
    )
    trace4 = go.Bar(
        x=list(map(lambda x: x.split("-")[0], x_axis_names)),
        y=list(map(lambda x: df[(df.epoch == x)][columnList[3]].mean(), x_axis_names)),
        name=columnList[3].replace("_", " "),
        error_y=dict(
            type='data',
            array=list(map(lambda x: 1.96 * (
                        df[(df.epoch == x)][columnList[3]].std() / df[(df.epoch == x)][columnList[3]].size),
                           x_axis_names)),
            visible=True
        )
    )

    data = [trace1, trace2, trace3, trace4]
    layout = go.Layout(
        width=1350,
        height=752,
        barmode='group',
    )

    fig = go.Figure(data=data, layout=layout)
    # Axis Config
    fig.update_xaxes(tickfont=dict(family='Rockwell', color='crimson', size=30))
    fig.update_yaxes(tickfont=dict(family='Rockwell', color='crimson', size=30))

    # Legend config
    fig.layout.update(
        showlegend=leg,
        legend=go.layout.Legend(
            font=dict(
                family="Courier New, monospace",
                size=35,
                color="black")
        )
    )

    pio.write_image(fig, str(file_name) + '.svg')
