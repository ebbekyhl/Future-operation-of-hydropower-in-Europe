# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:47:56 2021

@author: ebbek
"""
import pandas as pd
import numpy as np
from inflhist import inflhist
import scipy.stats as sp
from infl_concat import infl_concat
from inflhist import inflhist
from figure_formatting import figure_formatting
from AUcolor import AUcolor
import os
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")
colors,color_names = AUcolor()
title_font = fm.FontProperties(fname='AUfonts/AUPassata_Bold.ttf',size=22)
subtitle_font = fm.FontProperties(fname='AUfonts/AUPassata_Rg.ttf',size=14)
figure_font = fm.FontProperties(fname='AUfonts/AUPassata_Bold.ttf',size=18)
ticks_font = fm.FontProperties(fname='AUfonts/AUPassata_Rg.ttf',size=18)
country_code_font = fm.FontProperties(fname='AUfonts/AUPassata_Bold.ttf',size=18)
plt.close('all')
path_parent = os.path.dirname(os.getcwd())
gendatadir = path_parent + '/gendata/' # Directory in which general data is located
moddatadir = path_parent + '/moddata/' # Directory in which modelled inflow time series are located
resdatadir = path_parent + '/resdata/' # Directory in which calibrated modelled inflow time series are located
histdatadir = path_parent + '/histdata/' # Directory in which historical inflow time series are located
figuredir = path_parent + '/figure/' # Directory in which saved figures are located
#%% ============================ INPUT ========================================
gcm_list = ['MPI-M-MPI-ESM-LR','ICHEC-EC-EARTH','CNRM-CERFACS-CNRM-CM5','MOHC-HadGEM2-ES', 'NCC-NorESM1-M'] # General Circulation Model
rcm_list = ['RCA4','HIRHAM5'] # Regional Climate Model
rcp_list = ['85'] # Representative Concentration Pathways
hydrotype = 'HDAM' # Type of hydropower plant
WI = 1 # Wattsight data included (1) or not (0)
#%% ============================= OUTPUT ======================================
if WI == 1:    
    country_name = ['Norway','France','Spain','Switzerland','Sweden','Germany','Austria','Italy',
                    'Bulgaria','Croatia','Portugal', 'Romania','Czech_Republic', 'Hungary',
                    'Bosnia_and_Herzegovina','Serbia','Slovenia','Finland','Poland','Slovakia', 
                    'North_Macedonia','Montenegro']
    country_iso_alpha_2 = ['NO','FR','ES','CH','SE','DE','AT','IT','BG','HR','PT','RO',
                            'CZ','HU','BA','RS','SI','FI','PL','SK','MK','ME'] 
    nrows = 8
    ncols = 3
    lp = -2
else:
    country_name = ['Norway','France','Spain','Switzerland','Sweden','Austria','Italy', 'Romania',
                    'Bulgaria','Portugal', 'Montenegro', 'Serbia']
    country_iso_alpha_2 = ['NO','FR','ES','CH','SE','AT','IT','RO','BG','PT', 'ME', 'RS'] 
    nrows = 4
    ncols = 3
    lp = 0
    
# nrows = 1
# ncols = 1
# country_name = ['Norway'] 
# country_iso_alpha_2 = ['NO']

if nrows > 1:
    if len(country_iso_alpha_2) % 3 != 0:
        fig,ax = figure_formatting('','',1,1,figsiz = (17,14))
        gs = GridSpec(nrows, ncols)
        ax = [0]*len(country_iso_alpha_2)
        ax_twin = [0]*len(country_iso_alpha_2)
        counter = 0
        for row in range(nrows-1):
            for col in range(ncols):
                ax[counter] = plt.subplot(gs[row,col])
                ax[counter].tick_params(axis='y', colors=colors[0])
                ax[counter].set_yticklabels(ax[counter].get_yticks(), fontProperties = ticks_font)
                ax[counter].set_xticklabels(ax[counter].get_xticks(), fontProperties = ticks_font)
                ax[counter].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
                ax[counter].xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
                ax_twin[counter] = ax[counter].twinx()
                ax_twin[counter].tick_params(axis='y', colors=colors[6])
                ax_twin[counter].set_yticklabels(ax_twin[counter].get_yticks(), fontProperties = ticks_font)
                ax_twin[counter].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
                counter += 1
        if len(country_iso_alpha_2) % 3 == 1:
            ax[counter] = plt.subplot(gs[row+1,0])
            ax[counter].tick_params(axis='y', colors=colors[0])
            ax[counter].set_yticklabels(ax[counter].get_yticks(), fontProperties = ticks_font)
            ax[counter].set_xticklabels(ax[counter].get_xticks(), fontProperties = ticks_font)
            ax[counter].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
            ax[counter].xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
            ax_twin[counter] = ax[counter].twinx()
            ax_twin[counter].tick_params(axis='y', colors=colors[6])
            ax_twin[counter].set_yticklabels(ax_twin[counter].get_yticks(), fontProperties = ticks_font)
            ax_twin[counter].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        elif len(country_iso_alpha_2) % 3 == 2:
            ax[counter] = plt.subplot(gs[row+1,0])
            ax[counter].tick_params(axis='y', colors=colors[0])
            ax[counter].set_yticklabels(ax[counter].get_yticks(), fontProperties = ticks_font)
            ax[counter].set_xticklabels(ax[counter].get_xticks(), fontProperties = ticks_font)
            ax[counter].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
            ax[counter].xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
            ax_twin[counter] = ax[counter].twinx()
            ax_twin[counter].tick_params(axis='y', colors=colors[6])
            ax_twin[counter].set_yticklabels(ax_twin[counter].get_yticks(), fontProperties = ticks_font)
            ax_twin[counter].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
            ax[counter+1] = plt.subplot(gs[row+1,1])
            ax[counter+1].tick_params(axis='y', colors=colors[0])
            ax[counter+1].set_yticklabels(ax[counter+1].get_yticks(), fontProperties = ticks_font)
            ax[counter+1].set_xticklabels(ax[counter+1].get_xticks(), fontProperties = ticks_font)
            ax[counter+1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
            ax[counter+1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
            ax_twin[counter+1] = ax[counter+1].twinx()
            ax_twin[counter+1].tick_params(axis='y', colors=colors[6])
            ax_twin[counter+1].set_yticklabels(ax_twin[counter+1].get_yticks(), fontProperties = ticks_font)
            ax_twin[counter+1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.4)
    fig1,ax1 = figure_formatting('','',1,1,color='k',figsiz = (17,14))
    gs1 = GridSpec(nrows, ncols)
    ax1 = [0]*len(country_iso_alpha_2)
    counter = 0
    for row in range(nrows-1):
        for col in range(ncols):
            ax1[counter] = plt.subplot(gs1[row,col])
            ax1[counter].tick_params(axis='y', colors=colors[0])
            ax1[counter].set_yticklabels(ax1[counter].get_yticks(), fontProperties = ticks_font)
            ax1[counter].set_xticklabels(ax1[counter].get_xticks(), fontProperties = ticks_font)
            ax1[counter].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
            ax1[counter].xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
            counter += 1
    if len(country_iso_alpha_2) % 3 == 1:
        ax1[counter] = plt.subplot(gs[row+1,0])
        ax1[counter].tick_params(axis='y', colors='k')
        ax1[counter].set_yticklabels(ax1[counter].get_yticks(), fontProperties = ticks_font)
        ax1[counter].set_xticklabels(ax1[counter].get_xticks(), fontProperties = ticks_font)
        ax1[counter].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        ax1[counter].xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
    elif len(country_iso_alpha_2) % 3 == 2:
        ax1[counter] = plt.subplot(gs[row+1,0])
        ax1[counter].tick_params(axis='y', colors='k')
        ax1[counter].set_yticklabels(ax1[counter].get_yticks(), fontProperties = ticks_font)
        ax1[counter].set_xticklabels(ax1[counter].get_xticks(), fontProperties = ticks_font)
        ax1[counter].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        ax1[counter].xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        ax1[counter+1] = plt.subplot(gs[row+1,1])
        ax1[counter+1].tick_params(axis='y', colors='k')
        ax1[counter+1].set_yticklabels(ax1[counter+1].get_yticks(), fontProperties = ticks_font)
        ax1[counter+1].set_xticklabels(ax1[counter+1].get_xticks(), fontProperties = ticks_font)
        ax1[counter+1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        ax1[counter+1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
    fig1.subplots_adjust(wspace=0.3)
    fig1.subplots_adjust(hspace=0.4)
else:
    fig,ax,ax_twin = figure_formatting('','',nrows,ncols,figsiz = (18,12.5),ylab_twin='',color_twin=colors[6],twin=True)
    fig1,ax1 = figure_formatting('','',nrows,ncols,color='k',figsiz = (18,12.5))
rcp = rcp_list[0]
for c in range(len(country_name)):
    gcm_it = 0
    count = 0
    country = country_iso_alpha_2[c]
    country_l = country_name[c]
    hist = inflhist(histdatadir,1991,2020,country,country_l)*1e-3
    hist = hist[(hist.T != 0).any()] # Removing rows with zeros
    hist_m_train = hist #.drop(columns='month')
    hist_m_train['month'] = hist_m_train.index.month
    hist_m_train_seasonal = hist_m_train.groupby('month').mean()
    ind = pd.date_range('2016/01/01','2016/12/31',freq='MS')
    hist_m_train_seasonal.set_index(ind,inplace=True)
    ax[c].text(0.01, 0.7,country, transform=ax[c].transAxes,fontproperties=ticks_font,zorder=10)
    ax1[c].text(0.01, 0.7,country, transform=ax1[c].transAxes,fontproperties=ticks_font,zorder=10)
    pearson = [0]*len(gcm_list)*len(rcm_list)
    if c == 0:
        ax[c].plot(hist_m_train_seasonal,label='Historical',color=colors[0],zorder=20,lw=2,linestyle='--')
        # ax[c].set_ylabel('Historical inflow [TWh]')
    else:
        ax[c].plot(hist_m_train_seasonal,color=colors[0],zorder=5,lw=2,linestyle='--')
    for gcm in gcm_list:
        rcm_it = 0
        gcm = gcm_list[gcm_it]
        for rcm in rcm_list:
            rcm = rcm_list[rcm_it]
            infl_cal,infl_EOC = infl_concat(country,gendatadir,moddatadir,gcm,rcm,rcp)
            infl_cal = (infl_cal*(3.6e12)**-1)*1e-3 # Unit conversion from Joule to TWh 
            time_dt = pd.to_datetime(infl_cal.index)
            infl_cal['date'] = time_dt
            infl_cal.set_index('date',inplace=True)
            infl_cal_monthly = infl_cal.groupby(pd.Grouper(freq='MS')).sum()
            infl_cal_monthly['month'] = infl_cal_monthly.index.month
            infl_cal_m_train_seasonal = infl_cal_monthly.groupby('month').mean()
            infl_cal_m_train_seasonal.set_index(ind,inplace=True)
            # ax_twin[c].set_ylabel('Modelled inflow [TWh]')                
            pearson[count] = hist_m_train_seasonal.corrwith(infl_cal_m_train_seasonal).inflow.round(2)
            RF = hist_m_train_seasonal.values/infl_cal_m_train_seasonal.values
            if c == 0:
                ax_twin[c].plot(infl_cal_m_train_seasonal,color=colors[1+count],label=gcm + '-' + rcm,lw=1)
                ax1[c].plot(infl_cal_m_train_seasonal.index,RF,marker='o', linestyle='dashed',markersize=4,color=colors[1+count],label=gcm + '-' + rcm,lw=1)
            else:
                ax_twin[c].plot(infl_cal_m_train_seasonal,color=colors[1+count],lw=1)
                ax1[c].plot(infl_cal_m_train_seasonal.index,RF,marker='o', linestyle='dashed',markersize=4,color=colors[1+count],lw=1)
            count += 1
            rcm_it += 1
        gcm_it += 1
    if np.mean(pearson) >= 0:
        ax[c].text(0.8, 0.7,'r=' + str(np.mean(pearson).round(2)), transform=ax[c].transAxes,fontproperties=ticks_font,zorder=10)
    else:
        ax[c].text(0.77, 0.7,'r=' + str(np.mean(pearson).round(2)), transform=ax[c].transAxes,fontproperties=ticks_font,zorder=10)
    ax[c].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax[c].set_xlim([min(ind),max(ind)])
    ax[c].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax[c].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1[c].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1[c].set_xlim([min(ind),max(ind)])
    ax1[c].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax1[c].xaxis.set_major_locator(mdates.MonthLocator(interval=2))

fig.legend(frameon=False,loc='lower center',ncol=3,prop=ticks_font,handletextpad=0.1,labelspacing=0,borderaxespad=-0.5)
fig1.legend(frameon=False,loc='lower center',ncol=3,prop=ticks_font,handletextpad=0.1,labelspacing=0,borderaxespad=-0.5)

if len(country_name) == 2:
    fig.savefig(figuredir + 'Model_evaluation_' + country_name[0] + '_' + country_name[1] + '_' + rcp + '.png',bbox_inches='tight') 
    fig1.savefig(figuredir + 'Retain_factors_' + country_name[0] + '_' + country_name[1] + '_' + rcp + '.png',bbox_inches='tight') 
elif len(country_name) == 22:
    fig.savefig(figuredir + 'Model_evaluation_' + rcp + '.png',bbox_inches='tight') 
    fig1.savefig(figuredir + 'Retain_factors_' + rcp + '.png',bbox_inches='tight') 