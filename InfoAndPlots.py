import os
from dataImport.commons.basicFunctions import assembleData, computerFrAll, createPlotDF,\
 plotFun, saccade_df
from bokeh.io import export_png
from dataImport.selectivityMethods.mi import computeMI, plotScat, plotBar, compute_mi_stim_v_nostim, compute_m_index


dirr = os.fsencode("/home/mohsen/projects/neuroScienceWork/data/")

tmp = assembleData(dirr)
visualAndDelay = computerFrAll(tmp, 'vis')
saccade = computerFrAll(tmp, 'saccade')

completeData=tmp

saccade_data_set = saccade_df(tmp)
mivaluesDict = dict(Stim=computeMI(completeData, saccade_data_set, 'Stim').to_dict('list'),
                                 NoStim=computeMI(completeData, saccade_data_set, 'NoStim').to_dict('list'))


# PSTHs

for iterator in range(len(saccade)):
    export_png(plotFun(createPlotDF(DF=visualAndDelay, DF2=tmp[0], period='vis', ind=iterator),
                       createPlotDF(DF=saccade, DF2=tmp[iterator], period='sac', ind=iterator)),
               filename=str(iterator) + '.png')
    
# MI
    
saccade_data_set = saccade_df(tmp)

# Mutual information In-Out

mivaluesNoStim = computeMI(tmp, saccade_data_set, "noStim")
mivaluesStim = computeMI(tmp, saccade_data_set, "withStim")

# Mutual information Stim-NoStim

mivaluesIn = compute_mi_stim_v_nostim(tmp, saccade_data_set, "IN")
mivaluesOut = compute_mi_stim_v_nostim(tmp, saccade_data_set, "Out")

# modularity Index

mindexvaluesNoStim = compute_m_index(tmp, saccade_data_set, "noStim")
mindexvaluesStim = compute_m_index(tmp, saccade_data_set, "withStim")


export_png(plotScat(mivaluesNoStim), filename="scatNoStim.png")
export_png(plotScat(mivaluesStim), filename="scatWithStim.png")

plotBar(mivaluesNoStim, "barNo")
plotBar(mivaluesStim, "barWithStim")

