import scipy.signal as sig
import numpy as np
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Span, Label
from bokeh.palettes import Reds4, Blues4
from bokeh.plotting import figure

"""
0 -> inStim
1 -> outStim
2 -> inNoStim
3 -> outNoStim
"""


def createPlotDF(DF, period, ind):
    if period == 'sac':
        inStim = sig.savgol_filter(DF[ind][0], 315, 3)
        x = np.linspace(0, 3700, len(inStim)) - 3000
        inNoStim = sig.savgol_filter(DF[ind][2], 315, 3)
        outStim = sig.savgol_filter(DF[ind][1], 315, 3)
        outNoStim = sig.savgol_filter(DF[ind][3], 315, 3)
        df = pd.DataFrame(data=dict(x=x,
                                    inStim=inStim,
                                    inNoStim=inNoStim,
                                    outStim=outStim,
                                    outNoStim=outNoStim))
    else:
        inStim = sig.savgol_filter(DF[ind][0], 415, 3)
        inNoStim = sig.savgol_filter(DF[ind][2], 415, 3)
        outStim = sig.savgol_filter(DF[ind][1], 415, 3)
        outNoStim = sig.savgol_filter(DF[ind][3], 415, 3)
        x = np.linspace(0, 3000, len(inStim)) - 1000
        df = pd.DataFrame(data=dict(x=x,
                                    inStim=inStim,
                                    inNoStim=inNoStim,
                                    outStim=outStim,
                                    outNoStim=outNoStim))
    return df


def plotVisDel(DF, s1, s2, s3, xlab, ylab):
    source = ColumnDataSource(DF.reset_index())
    p = figure(title=str(s3), x_axis_label=str(xlab),
               y_axis_label=str(ylab), toolbar_location=None)

    p.line(x='x', y=str(s1), color=Blues4[0],
           source=source, legend="")
    p.line(x='x', y=str(s2), color=Reds4[0],
           source=source, legend="")

    vline = Span(location=1000, dimension='height', line_dash='dashed',
                 line_color='black', line_width=2)
    vline0 = Span(location=0, dimension='height', line_dash='dashed',
                  line_color='grey', line_width=2)

    maxY = max([max(DF.inStim), max(DF.inStim)])
    text0 = Label(x=300, y=maxY, text='Visual period', render_mode='css',
                  border_line_color='black', border_line_alpha=1.5,
                  background_fill_color='white', background_fill_alpha=1.5)
    text1 = Label(x=1300, y=maxY, text='Delay period', render_mode='css',
                  border_line_color='black', border_line_alpha=1.5,
                  background_fill_color='white', background_fill_alpha=1.5)
    text2 = Label(x=-700, y=maxY, text='Baseline', render_mode='css',
                  border_line_color='black', border_line_alpha=1.5,
                  background_fill_color='white', background_fill_alpha=1.5)
    p.add_layout(vline)
    p.add_layout(vline0)
    p.add_layout(text0)
    p.add_layout(text1)
    p.add_layout(text2)
    return p


def plotSac(DF, s1, s2, s3, xlab, ylab):
    source1 = ColumnDataSource(DF.reset_index())
    s = figure(title=str(s3),
               x_axis_label=str(xlab),
               y_axis_label=str(ylab), toolbar_location=None)
    s.line(x='x', y=str(s1), color=Blues4[0],
           source=source1, legend="in")
    s.line(x='x', y=str(s2), color=Reds4[0],
           source=source1, legend="out")
    vline = Span(location=0, dimension='height', line_dash='dashed',
                 line_color='black', line_width=2)
    s.add_layout(vline)
    # s.x_range=Range1d(-400, 350)
    return s


def plotFun(DF1, DF2):
    wpin = plotVisDel(DF1, 'inStim', 'outStim', 'With stimmulation',
                      xlab='Time from stimulus onset(ms)',
                      ylab='Firing rate(Hz)')
    wsin = plotSac(DF2, 'inStim', 'outStim', 'Saccade period',
                   xlab='Time from saccade onset(ms)', ylab='')
    wpout = plotVisDel(DF1, 'inNoStim', 'outNoStim', 'Without stimmulation',
                       xlab='Time from stimulus onset(ms)',
                       ylab='Firing rate(Hz)')
    wsout = plotSac(DF2, 'inNoStim', 'outNoStim', '',
                    xlab='Time from saccade onset(ms)', ylab='')
    grid = gridplot([wpin, wsin, wpout, wsout], ncols=2,
                    sizing_mode='stretch_both', toolbar_location=None)
    return grid