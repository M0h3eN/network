from sklearn.metrics import mutual_info_score
from dataImport.commons.basicFunctions import conditionSelect, np, pd, figure, computeSpikeCount
from scipy.stats import entropy
import plotly.graph_objs as go
import plotly.io as pio


'''
0 -> inStim
1 -> outStim
2 -> inNoStim
3 -> outNoStim
'''


def computeMI(DF, saccad_data, stim_status):
    mutualVisual = []
    mutualMem = []
    mutualSac = []
    # miOutNoStim = []
    for ni in range(len(DF)):
        # With Stim
        if stim_status == "Stim":

            # in

            inVisual = computeSpikeCount(conditionSelect(DF[ni], subStatus='inStim'), 1050, 1250)
            inMem = computeSpikeCount(conditionSelect(DF[ni], subStatus='inStim'), 2500, 2700)
            inSac = computeSpikeCount(conditionSelect(saccad_data[ni], subStatus='inStim'), 2700, 2950)

            # out

            outVisual = computeSpikeCount(conditionSelect(DF[ni], subStatus='outStim'), 1050, 1250)
            outMem = computeSpikeCount(conditionSelect(DF[ni], subStatus='outStim'), 2500, 2700)
            outSac = computeSpikeCount(conditionSelect(saccad_data[ni], subStatus='outStim'), 2700, 2950)

            # Mutual Information Score

            mutualVisual.append(mutual_info_score(inVisual, outVisual))
            mutualMem.append(mutual_info_score(inMem, outMem))
            mutualSac.append(mutual_info_score(inSac, outSac))

            # Without Stim

        else:
            # in

            inVisual = computeSpikeCount(conditionSelect(DF[ni], subStatus='inNoStim'), 1050, 1250)
            inMem = computeSpikeCount(conditionSelect(DF[ni], subStatus='inNoStim'), 2500, 2700)
            inSac = computeSpikeCount(conditionSelect(saccad_data[ni], subStatus='inNoStim'), 2700, 2950)

            # out

            outVisual = computeSpikeCount(conditionSelect(DF[ni], subStatus='outNoStim'), 1050, 1250)
            outMem = computeSpikeCount(conditionSelect(DF[ni], subStatus='outNoStim'), 2500, 2700)
            outSac = computeSpikeCount(conditionSelect(saccad_data[ni], subStatus='outNoStim'), 2700, 2950)

            # Mutual Information Score

            mutualVisual.append(mutual_info_score(inVisual, outVisual))
            mutualMem.append(mutual_info_score(inMem, outMem))
            mutualSac.append(mutual_info_score(inSac, outSac))

    MI_Values = pd.DataFrame(dict(mi_visual=mutualVisual,
                                      mi_mem=mutualMem,
                                      mi_sac=mutualSac))
    return MI_Values


def compute_m_index(DF, saccad_data, stim_status):
    mIndexVisual = []
    mIndexMem = []
    mIndexSac = []
    # miOutNoStim = []
    for ni in range(len(DF)):
        # With Stim
        if stim_status == "Stim":

            allVisual = computeSpikeCount(conditionSelect(DF[ni], subStatus="allStim"), 1050, 1250)
            allMem = computeSpikeCount(conditionSelect(DF[ni], subStatus="allStim"), 2500, 2700)
            allSac = computeSpikeCount(conditionSelect(saccad_data[ni], subStatus="allStim"), 2700, 2950)
            # allfrNoStim = computeSpikeCount(conditionSelect(DF[ni], subStatus="allNoStim"), 0, 3798)
            # With Stim
            # in

            inVisual = computeSpikeCount(conditionSelect(DF[ni], subStatus='inStim'), 1050, 1250)
            inMem = computeSpikeCount(conditionSelect(DF[ni], subStatus='inStim'), 2500, 2700)
            inSac = computeSpikeCount(conditionSelect(saccad_data[ni], subStatus='inStim'), 2700, 2950)

            # out

            outVisual = computeSpikeCount(conditionSelect(DF[ni], subStatus='outStim'), 1050, 1250)
            outMem = computeSpikeCount(conditionSelect(DF[ni], subStatus='outStim'), 2500, 2700)
            outSac = computeSpikeCount(conditionSelect(saccad_data[ni], subStatus='outStim'), 2700, 2950)

            # Index

            mIndexVisual.append((np.mean(inVisual) - np.mean(outVisual))/np.mean(allVisual))
            mIndexMem.append((np.mean(inMem) - np.mean(outMem))/np.mean(allMem))
            mIndexSac.append((np.mean(inSac) - np.mean(outSac))/np.mean(allSac))

            # Without Stim

        else:

            allVisual = computeSpikeCount(conditionSelect(DF[ni], subStatus="allNoStim"), 1050, 1250)
            allMem = computeSpikeCount(conditionSelect(DF[ni], subStatus="allNoStim"), 2500, 2700)
            allSac = computeSpikeCount(conditionSelect(saccad_data[ni], subStatus="allNoStim"), 2700, 2950)
            # allfrNoStim = computeSpikeCount(conditionSelect(DF[ni], subStatus="allNoStim"), 0, 3798)
            # With Stim
            # in

            inVisual = computeSpikeCount(conditionSelect(DF[ni], subStatus='inNoStim'), 1050, 1250)
            inMem = computeSpikeCount(conditionSelect(DF[ni], subStatus='inNoStim'), 2500, 2700)
            inSac = computeSpikeCount(conditionSelect(saccad_data[ni], subStatus='inNoStim'), 2700, 2950)

            # out

            outVisual = computeSpikeCount(conditionSelect(DF[ni], subStatus='outNoStim'), 1050, 1250)
            outMem = computeSpikeCount(conditionSelect(DF[ni], subStatus='outNoStim'), 2500, 2700)
            outSac = computeSpikeCount(conditionSelect(saccad_data[ni], subStatus='outNoStim'), 2700, 2950)

            # Index

            mIndexVisual.append((np.mean(inVisual) - np.mean(outVisual)) / np.mean(allVisual))
            mIndexMem.append((np.mean(inMem) - np.mean(outMem)) / np.mean(allMem))
            mIndexSac.append((np.mean(inSac) - np.mean(outSac)) / np.mean(allSac))

    MIndex_Values = pd.DataFrame(dict(mIndex_visual=mIndexVisual,
                                  mIndex_mem=mIndexMem,
                                  mIndex_sac=mIndexSac))
    return MIndex_Values


def compute_mi_stim_v_nostim(DF, saccad_data, in_status):
    miVisual = []
    miMem = []
    miSac = []
    # miOutNoStim = []
    for ni in range(len(DF)):
        # In
        if in_status == "IN":
            # IN

            stimVisual = computeSpikeCount(conditionSelect(DF[ni], subStatus='inStim'), 1050, 1250)
            stimMem = computeSpikeCount(conditionSelect(DF[ni], subStatus='inStim'), 2500, 2700)
            stimSac = computeSpikeCount(conditionSelect(saccad_data[ni], subStatus='inStim'), 2700, 2950)

            # out

            nostimVisual = computeSpikeCount(conditionSelect(DF[ni], subStatus='inNoStim'), 1050, 1250)
            nostimMem = computeSpikeCount(conditionSelect(DF[ni], subStatus='inNoStim'), 2500, 2700)
            nostimSac = computeSpikeCount(conditionSelect(saccad_data[ni], subStatus='inNoStim'), 2700, 2950)

            # Mutual Information Score

            miVisual.append(mutual_info_score(stimVisual, nostimVisual))
            miMem.append(mutual_info_score(stimMem, nostimMem))
            miSac.append(mutual_info_score(stimSac, nostimSac))

        else:

            # OUT

            stimVisual = computeSpikeCount(conditionSelect(DF[ni], subStatus='outStim'), 1050, 1250)
            stimMem = computeSpikeCount(conditionSelect(DF[ni], subStatus='outStim'), 2500, 2700)
            stimSac = computeSpikeCount(conditionSelect(saccad_data[ni], subStatus='outStim'), 2700, 2950)

            # out

            nostimVisual = computeSpikeCount(conditionSelect(DF[ni], subStatus='outNoStim'), 1050, 1250)
            nostimMem = computeSpikeCount(conditionSelect(DF[ni], subStatus='outNoStim'), 2500, 2700)
            nostimSac = computeSpikeCount(conditionSelect(saccad_data[ni], subStatus='outNoStim'), 2700, 2950)

            # Mutual Information Score

            miVisual.append(mutual_info_score(stimVisual, nostimVisual))
            miMem.append(mutual_info_score(stimMem, nostimMem))
            miSac.append(mutual_info_score(stimSac, nostimSac))

    MI_Values = pd.DataFrame(dict(mi_visual=miVisual,
                                  mi_mem=miMem,
                                  mi_sac=miSac))
    return MI_Values


'''
def plotHist(DF, title, col):
    p1 = figure(title=str(title), toolbar_location=None,
                x_axis_label="Mutual information(bit)",
                y_axis_label="Frequency")
    hist, edges = np.histogram(DF[str(col)], density=True, bins=15)
    p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], alpha=0.3)
    #        fill_color="#036564", line_color="#033649")
    return p1
'''

def plotScat(DF):
    p1 = figure(title="Mutual information between in and out", toolbar_location=None,
                x_axis_label="Mutual information(bit)[Visual]",
                y_axis_label="Mutual information(bit)[Memory]")
    # DF1 = resample(DF, n_samples=1000)
    p1.scatter(DF['mi_visual'], DF['mi_mem'], fill_alpha=0.6,
               line_color=None)
    return p1


def plotBar(DF, file_name):
    
    trace1 = go.Bar(
    x = ['Visual', 'Memory', 'Saccade'],
    y = [DF['mi_visual'].median(), DF['mi_mem'].median(),
       DF['mi_sac'].median()],
    name='Control',
    error_y=dict(
        type='data',
        array=[DF['mi_visual'].mean()-DF['mi_visual'].std(),
               DF['mi_mem'].mean()-DF['mi_mem'].std(),
               DF['mi_mem'].mean()-DF['mi_mem'].std()],
        visible=True
        )
    )
    data = [trace1]
    layout = go.Layout(
    width=1350,
    height=752,
    barmode='group',
     yaxis=dict(
        title='Mutual information(bit)',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        ))
    )
    
    fig = go.Figure(data=data, layout=layout)
    pio.write_image(fig, file = str(file_name) + '.svg')
    
    
def plotBarGraphCentrality(DF, mostCenIndex, file_name):
    trace1 = go.Bar(
    x=['All Neurons'],
    y=[DF['mi_visual'].mean()],
    name='Avrage'
    )
    trace2 = go.Bar(
    x=['Neuron' + ' ' + str(mostCenIndex)],
    y=[DF['mi_visual'][mostCenIndex]],
    name='Most degree centrality'
    )

    data = [trace1, trace2]
    layout = go.Layout(
    width=1350,
    height=752,
    barmode='group',
     yaxis=dict(
        title='Mutual information(bit)',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        ))
    )
    fig = go.Figure(data=data, layout=layout)
    pio.write_image(fig, str(file_name) + '.svg')
    
    
def plotBarGraphCentralityCompare(DF, mostCenIndex, file_name):
    
    defualtColor = 'rgba(204,204,204,1)'
    mostCenDegColor = 'rgba(222,45,38,0.8)'
    
    colorList = list(np.repeat(defualtColor, DF.shape[0], axis = 0))
    colorList[mostCenIndex] = mostCenDegColor

    trace0 = go.Bar(
    x=['Neuron' + ' ' + str(it) for it in range(DF.shape[0])],
    y=list(DF['mi_visual']),
    marker=dict(
        color=colorList),
            )

    data = [trace0]
    layout = go.Layout(
    width=1350,
    height=752,
    barmode='group',
     yaxis=dict(
        title='Mutual information(bit)',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        ))
    )

    fig = go.Figure(data=data, layout=layout)
    pio.write_image(fig, str(file_name) + '.svg')


def plotGroupedBar(df, file_name):

    columnList = list(df.columns)
    trace1 = go.Bar(
        x=list(df[columnList[0]].index.map(str)),
        y=list(df[columnList[0]]),
        name=columnList[0].split("_")[1]
    )
    trace2 = go.Bar(
        x=list(df[columnList[1]].index.map(str)),
        y=list(df[columnList[1]]),
        name=columnList[1].split("_")[1]
    )
    trace3 = go.Bar(
        x=list(df[columnList[2]].index.map(str)),
        y=list(df[columnList[2]]),
        name=columnList[2].split("_")[1]
    )

    data = [trace1, trace2, trace3]
    layout = go.Layout(
        width=1350,
        height=752,
        barmode='group',
        xaxis=dict(
            dtick=1,
            tickcolor='#000',
            title='Neuron',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            tickcolor='#000',
            title='Modularity Index',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            ))
    )

    fig = go.Figure(data=data, layout=layout)
    pio.write_image(fig, str(file_name) + '.svg')

