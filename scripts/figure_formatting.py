# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:26:11 2020
@author: ebbek
Function "AUcolor defines the Aarhus University colors.
Function "figure_formatting" creates a matplotlib figure with latex and AU fonts.
Input:
    xlab: x-label
    ylab: y-label
    nrows: number of rows 
    ncols: number of columns
    color: figure color (optional)
    figsiz: figure size (optional)
    ylab_twin: Label of twinx axis (optional)
    color_twin: Color of twinx axis (optional)
    twin: Whether twinx axis is used (True) or not (False)
"""
def AUcolor():
    AU_Blue = [0,61,115]
    AU_Purple = [101,90,159]
    AU_Cyan = [55,160,203]
    AU_Green =  [139,173,63]
    AU_Yellow = [250,187,0]
    AU_Orange = [238,127,0]
    AU_Red = [226,0,26]
    AU_Magenta = [226,0,122]
    AU_Grey = [135,135,135]
    AU_Turkis = [0,171,164]
    AU_Dark_Blue = [0,37,70]
    AU_Dark_Purple = [40,28,65]
    AU_Dark_Cyan = [0,62,92]
    AU_Dark_Green = [66,88,33]
    AU_Dark_Yellow =  [99,75,3]
    AU_Dark_Orange = [95,52,8]
    AU_Dark_Red = [91,12,12]
    AU_Dark_Magenta = [95,0,48]
    AU_Dark_Grey = [75,75,74]
    AU_Dark_Turkis = [0,69,67]
    
    color_list = [AU_Blue,
                  AU_Purple,
                  AU_Green,
                  AU_Yellow,
                  AU_Orange,
                  AU_Cyan,
                  AU_Red,
                  AU_Magenta,
                  AU_Grey,
                  AU_Turkis,
                  AU_Dark_Blue,
                  AU_Dark_Purple,
                  AU_Dark_Cyan,
                  AU_Dark_Green,
                  AU_Dark_Yellow,
                  AU_Dark_Orange,
                  AU_Dark_Red,
                  AU_Dark_Magenta,
                  AU_Dark_Grey,
                  AU_Dark_Turkis]
    
    color_names = ['AU_Blue',
                  'AU_Purple',
                  'AU_Green',
                  'AU_Yellow',
                  'AU_Orange',
                  'AU_Cyan',
                  'AU_Red',
                  'AU_Magenta',
                  'AU_Grey',
                  'AU_Turkis',
                  'AU_Dark_Blue',
                  'AU_Dark_Purple',
                  'AU_Dark_Cyan',
                  'AU_Dark_Green',
                  'AU_Dark_Yellow',
                  'AU_Dark_Orange',
                  'AU_Dark_Red',
                  'AU_Dark_Magenta',
                  'AU_Dark_Grey',
                  'AU_Dark_Turkis']
    
    import numpy as np
    colors = np.array(color_list)/255
    return colors, color_names

def figure_formatting(xlab,ylab,nrows,ncols,color=[0.,0.239, 0.451],figsiz = 0,ylab_twin = False, color_twin=False,twin=False):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import ScalarFormatter
    import numpy as np
    colors,color_names = AUcolor()
    from matplotlib import font_manager as fm, rcParams
    figure_font = fm.FontProperties(fname='AUfonts/AUPassata_Bold.ttf',size=28)
    ticks_font = fm.FontProperties(fname='AUfonts/AUPassata_Rg.ttf',size=28)
    if nrows == 1 and ncols == 3:
        if figsiz == 0:
            figsiz = (17,5)
        fig = plt.figure(figsize = figsiz)
        plt.subplots_adjust(wspace=0.2)
    elif nrows > 1 and ncols > 1:
        if figsiz == 0:
            figsiz = (17,10)
        fig = plt.figure(figsize=figsiz)
        plt.subplots_adjust(hspace=0.6)
        plt.subplots_adjust(wspace=0.15)
    else:
        if figsiz == 0:
            figsiz = (10,5)
        fig = plt.figure(figsize=figsiz)
    gs = GridSpec(nrows, ncols, figure=fig)
    ax = [0]*nrows*ncols
    ax_twin = [0]*nrows*ncols
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            ax[k] = fig.add_subplot(gs[i, j])
            ax[k].set_ylabel(ylab,fontproperties=figure_font,color=color)
            ax[k].set_xlabel(xlab,fontproperties=figure_font,color=color)
            ax[k].tick_params(axis='y', colors=color)
            ax[k].set_yticklabels(ax[k].get_yticks(), fontProperties = ticks_font)
            ax[k].set_xticklabels(ax[k].get_xticks(), fontProperties = ticks_font)
            ax[k].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
            ax[k].xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
            if twin == True:
                ax_twin[k] = ax[k].twinx()
                ax_twin[k].tick_params(axis='y', colors=color_twin)
                ax_twin[k].set_yticklabels(ax_twin[k].get_yticks(), fontProperties = ticks_font)
                ax_twin[k].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
                ax_twin[k].set_ylabel(ylab_twin,fontproperties=figure_font,color=color_twin)
            k = k + 1
    if twin == True:
        return fig,ax,ax_twin
    else:
        return fig,ax