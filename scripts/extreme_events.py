# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:49:00 2021

@author: ebbek
"""
def extreme_events_count(country_codes,scen,hist_woa_EM,cc_woa_EM):
    import numpy as np
    import itertools
    mean_dur_boc = [0]*len(country_codes)
    mean_nb_seq_boc = [0]*len(country_codes)
    mean_dur_eoc = [0]*len(country_codes)
    mean_nb_seq_eoc = [0]*len(country_codes)
    for c in range(len(country_codes)):
        infl_BOC_d = hist_woa_EM[country_codes[c]]
        infl_EOC_d = cc_woa_EM[country_codes[c]]
        infl_BOC_d_array = np.array(infl_BOC_d)
        infl_EOC_d_array = np.array(infl_EOC_d)
        if scen == 'drought':
            extreme_BOC = np.percentile(infl_BOC_d_array,10)
            condition = infl_EOC_d_array <= extreme_BOC
            condition_boc = infl_BOC_d_array <= extreme_BOC
        else:
            extreme_BOC = np.percentile(infl_BOC_d_array,90)
            condition = infl_EOC_d_array >= extreme_BOC
            condition_boc = infl_BOC_d_array >= extreme_BOC
            
        cons_days_eoc = [ sum( 1 for _ in group ) for key, group in itertools.groupby( condition ) if key ]
        cons_days_boc = [ sum( 1 for _ in group ) for key, group in itertools.groupby( condition_boc ) if key ]
        if scen == 'drought':
            consec_days_eoc = infl_EOC_d[infl_EOC_d < extreme_BOC]
        else:
            consec_days_eoc = infl_EOC_d[infl_EOC_d > extreme_BOC]
        cons_groups = [0]*len(cons_days_eoc)
        j = 0
        for i in range(len(cons_days_eoc)):
            if i == 0:
                cons_groups[i] = consec_days_eoc.iloc[0:cons_days_eoc[i]]
            else:
                cons_groups[i] = consec_days_eoc.iloc[j:j+cons_days_eoc[i]]
            j += cons_days_eoc[i]
        mean_dur_boc[c] = sum(cons_days_boc)/len(cons_days_boc) # Mean drought duration
        mean_nb_seq_boc[c] = len(np.array(cons_days_boc)[np.array(cons_days_boc) > 1])/len(infl_BOC_d_array)*365
        mean_dur_eoc[c] = sum(cons_days_eoc)/len(cons_days_eoc) # Mean drought duration
        mean_nb_seq_eoc[c] = len(np.array(cons_days_eoc)[np.array(cons_days_eoc) > 1])/len(infl_EOC_d_array)*365
    return mean_dur_boc,mean_nb_seq_boc,mean_dur_eoc,mean_nb_seq_eoc
      
def extreme_events_plot(country_codes,countries,scen,fig,ax,mean_dur_boc,mean_dur_eoc,mean_nb_seq_boc,mean_nb_seq_eoc,rcp_dic,rcp):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    import matplotlib.ticker as plticker
    import numpy as np
    from AUcolor import AUcolor
    colors,color_names = AUcolor()
    from matplotlib import font_manager as fm
    ticks_font = fm.FontProperties(fname='AUfonts/AUPassata_Rg.ttf',size=20)
    color_BOC = 'k'
    color_EOC = rcp_dic[rcp]
    ax[0].scatter(-1,-1,label='EOC RCP' + str(int(rcp)/10),zorder=1,color=color_EOC,marker='s')
    ax[0].scatter(-1,-1,label='BOC',zorder=1,color=color_BOC,marker='s')
    z = np.array(mean_dur_eoc)*np.array(mean_nb_seq_eoc)
    # ax[0].set_ylim([min(mean_nb_seq_eoc),max(mean_nb_seq_eoc)*1.4])
    # ax[0].set_xlim([min(mean_dur_eoc),max(mean_dur_eoc)*1.3])
    ax[0].set_xlim([2.08,37.54])
    ax[0].set_ylim([0.23,21.91])
    x1 = np.arange(0,40,0.01)
    y1 = np.arange(0,40,0.01)
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    xv, yv = np.meshgrid(x1, y1)
    zv = xv*yv
    #zv[zv > z.max()] = z.max()
    if scen == 'drought':
        IPCCpreccolors = [(245/255,245/255,245/255),(84/255,48/255,5/255)]
        colormap = LinearSegmentedColormap.from_list(
        'IPCCprec', IPCCpreccolors, N=50)
        #colormap = plt.cm.Reds
    else:
        #colormap = plt.cm.Blues
        IPCCpreccolors = [(245/255,245/255,245/255),(0,60/255,48/255)]
        colormap = LinearSegmentedColormap.from_list(
        'IPCCprec', IPCCpreccolors, N=50)
    normalize = Normalize(vmin=0, vmax=100) #(vmin=z.min(), vmax=z.max())
    plt.contourf(xv,yv,zv,cmap=colormap,norm=normalize,levels=[1,10,20,40,60,80,100],zorder=0)#,levels=10)
    for i in range(len(countries)):
        ax[0].annotate(country_codes[i], xy=(mean_dur_boc[i],mean_nb_seq_boc[i]),color='w',
                        bbox=dict(boxstyle="circle", fc=color_BOC),fontsize=15,zorder=4)
        
        ax[0].annotate(country_codes[i], xy=(mean_dur_eoc[i],mean_nb_seq_eoc[i]),color='w',
                        bbox=dict(boxstyle="round", fc=color_EOC),fontsize=15,zorder=5)
        
    ax[0].legend(prop=ticks_font)
    #ax[0].grid(linestyle='-', linewidth='0.75', color='darkgrey',which='both',axis='both')  
    cbar = plt.colorbar(ax=ax[0],boundaries=np.linspace(0, 60, 6)) 
    cbar.ax.set_ylabel('Number of days with ' + scen + ' in a year',fontproperties=ticks_font)
    cbar.ax.tick_params(labelsize=20)
    ax[0].yaxis.set_minor_formatter(ScalarFormatter(useMathText=True, useOffset=False))
    plt.setp(ax[0].get_yminorticklabels(), visible=False)
    #ax[0].tick_params(axis='both', which='minor', labelsize=20)
    #ax[0].set_yticks([2,3,4,5,6,7,8,9,10,20])
    #ax[0].set_xticks([3,4,5,6,7,8,9,10,20,30])
    # loc = plticker.MultipleLocator(base=2)
    # ax[0].yaxis.set_minor_locator(loc)
    # ax.xaxis.set_minor_locator(MultipleLocator(5))
    return fig
