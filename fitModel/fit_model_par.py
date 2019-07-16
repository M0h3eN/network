from commons.selectivityMethods.mi import plotBarGraphCentrality, plotBarGraphCentralityCompare
from commons.tools.basicFunctions import computerFrAll, createPlotDF, plotFun
from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab
from commons.plotRelatedFunctions.plot_network import plot_network
from bokeh.io import export_png
from pymongo import MongoClient
from networkx.readwrite import json_graph


import os
import numpy as np
import pandas as  pd
import networkx as nx


def generate_matrix_indices(k, b):
    Akeys = []
    Wkeys = []
    WeKeys = []
    for i in range(k):
        for j in range(k):
            Akeys.append("a_" + str(i + 1) + ',' + str(j + 1))
            Wkeys.append("w_" + str(i + 1) + ',' + str(j + 1))
            WeKeys.append("we_" + str(i + 1) + ',' + str(j + 1))

    Imkeys = []
    for i in range(k):
        for j in range(k):
            for p1 in range(b):
                Imkeys.append("im_" + str(i + 1) + ',' + str(j + 1) + ',' + str(p1 + 1))

    LamKeys = []
    for i in range(k):
        LamKeys.append("la_" + str(i + 1))

    allKeys = Akeys + Wkeys + WeKeys + Imkeys + LamKeys

    return allKeys, Akeys, Wkeys, WeKeys, Imkeys, LamKeys



def fit_model_discrete_time_network_hawkes_spike_and_slab(dtmax, hypers, itter, period,
                                                          data, completeData, tempPath,
                                                          args, mivalues,
                                                          chain, per):
    # MongoDB connection config

    client = MongoClient("mongodb://" + args.host + ':' + args.port)
    paramValuesDB = client.MCMC_param
    diagnosticValuesDB = client.MCMC_diag
    GraphDB = client.Graph
    EstimatedGrapgDB = client.Estimation

    writePath = tempPath + '/' + 'Plots' '/' + period[per] + '/'
    graphPah = tempPath + '/' + 'Network' + '/'
    ratePath = tempPath + '/' + 'EstimatedRate' + '/'

    if not (os.path.exists(writePath)):
        os.makedirs(writePath)

    if not(os.path.exists(graphPah)):
        os.makedirs(graphPah)

    if not(os.path.exists(ratePath)):
        os.makedirs(ratePath)

    k = data[per].shape[1]

    model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
        K=k, dt_max=dtmax,
        network_hypers=hypers)
    assert model.check_stability()

    model.add_data(data[per])

    ###########################################################
    # Fit the test model with Gibbs sampling
    ###########################################################

    samples = []
    lps = []

    for itr in range(itter):
        # print("Gibbs iteration ", itr)
        model.resample_model()
        lps.append(model.log_probability())
        samples.append(model.copy_sample())

    # def analyze_samples(true_model, samples, lps):
    N_samples = len(samples)
    B = samples[1].impulse_model.B

    # Ingestion each model MCMC samples to mongoDB

    allKeys, Akeys, Wkeys, WeKeys, Imkeys, LamKeys = generate_matrix_indices(k, B)

    All_samples_param = (
        [dict(zip(allKeys, (
                list(np.array(s.weight_model.A.flatten(), "float")) +
                list(np.array(s.weight_model.W.flatten(), "float")) +
                list(np.array(s.weight_model.W_effective.flatten(), "float")) +
                list(np.array(np.reshape(s.impulse_model.g, (k, k, s.impulse_model.B)).flatten(),
                              "float")) +
                list(np.array(s.bias_model.lambda0.flatten(), "float"))
        )
                  )) for s in samples])

    colName = period[per] + '___' + str(chain)
    paramValuesDB[colName].insert_many(All_samples_param)

    A_samples = np.array([s.weight_model.A for s in samples])
    W_samples = np.array([s.weight_model.W for s in samples])
    W_effective_sample = np.array([s.weight_model.W_effective for s in samples])
    LambdaZero_sample = np.array([s.bias_model.lambda0 for s in samples])
    ImpulseG_sample = np.array([np.reshape(s.impulse_model.g, (k, k, s.impulse_model.B)) for s in samples])
    Rate_samples = np.array([s.compute_rate(S=data[per]) for s in samples])

    # DIC evaluation

    # theta_bar evaluation

    A_mean = A_samples[:, ...].mean(axis=0)
    W_mean = W_samples[:, ...].mean(axis=0)
    LambdaZero_mean = LambdaZero_sample[:, ...].mean(axis=0)
    ImpulseG_mean = ImpulseG_sample[:, ...].mean(axis=0)

    logLik = np.array(lps)

    # D_hat evaluation

    D_hat = -2 * (model.weight_model.log_likelihood(tuple((A_mean, W_mean))) +
                  model.impulse_model.log_likelihood(ImpulseG_mean) +
                  model.bias_model.log_likelihood(LambdaZero_mean))
    D_bar = -2 * np.mean(logLik)
    pD = D_bar - D_hat
    pV = np.var(-2 * logLik) / 2

    DIC = pD + D_bar

    modelDiag = {'Model': str(model.__class__).split(".")[2].split("'")[0],
                 'logLik': lps,
                 'D_hat': D_hat,
                 'D_bar': D_bar,
                 'pD': pD,
                 'pV': pV,
                 'DIC': DIC}

    colNameDiag = period[per] + '___' + str(chain)
    diagnosticValuesDB[colNameDiag].insert_one(modelDiag)
    # Compute sample statistics for second half of samples

    offset = N_samples // 2
    W_effective_mean = np.median(W_effective_sample[offset:, ...], axis=0)
    rate_mean = pd.DataFrame(np.mean(Rate_samples[offset:, ...], axis=0))

    # insert estimated rate in a csv file of dimension T*N

    rate_mean.to_csv(index=False, path_or_buf= ratePath + period[per] + '.csv')

    # Insert estimated graph after burnIn phase

    EstimatedGrapgDB[colNameDiag].insert_one(dict(zip(WeKeys,
                                                      list(np.array(W_effective_mean.flatten(), "float")))))

    # Create Graph Objects
    typ = nx.DiGraph()
    G0 = nx.from_numpy_matrix(W_effective_mean, create_using=typ)
    G = nx.from_numpy_matrix(15 * W_effective_mean, create_using=typ)

    dataGraph = json_graph.adjacency_data(G0)

    nx.write_gml(G0, graphPah + period[per] + ".gml")

    colNameGraph = period[per] + '___' + str(chain)
    GraphDB[colNameGraph].insert_one(dataGraph)

    fv = plot_network(G, writePath + 'Network')

    plotBarGraphCentrality(mivalues[period[per].split("-")[2]], fv, writePath + 'MutualInformation')
    plotBarGraphCentralityCompare(mivalues[period[per].split("-")[2]], fv, writePath + 'MutualInformationCompare')

    # PSTH

    visualAndDelay = computerFrAll(completeData, 'vis')
    saccade = computerFrAll(completeData, 'saccade')

    export_png(plotFun(createPlotDF(DF=visualAndDelay, DF2=completeData[0], period='vis', ind=fv),
                       createPlotDF(DF=saccade, DF2=completeData[fv], period='sac', ind=fv)),
               filename=writePath + 'FiringRate' + '.png')


