# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:32:24 2021

@author: au485969
"""

import pandas as pd
import numpy as np
from inflhist import inflhist
import scipy.stats as sp
from sklearn.metrics import mean_squared_error
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
figure_font = fm.FontProperties(fname='AUfonts/AUPassata_Bold.ttf',size=28)
ticks_font = fm.FontProperties(fname='AUfonts/AUPassata_Rg.ttf',size=28)
plt.close('all')
path_parent = os.path.dirname(os.getcwd())
gendatadir = path_parent + '/gendata/' # Directory in which general data is located
moddatadir = path_parent + '/moddata/' # Directory in which modelled inflow time series are located
resdatadir = path_parent + '/resdata/' # Directory in which calibrated modelled inflow time series are located
histdatadir = path_parent + '/histdata/' # Directory in which historical inflow time series are located
figuredir = path_parent + '/figure/' # Directory in which saved figures are located
#%% Functions
def train(train_date,modelled,hist): # Uses every data which is before the split date to train
    hist_m_train = hist.loc[hist.index >= pd.to_datetime(date_train[0])].loc[hist.loc[hist.index >= pd.to_datetime(date_train[0])].index <= date_train[1]]
    hist_m_train['month'] = hist_m_train.index.month
    hist_m_train_seasonal = hist_m_train.groupby('month').mean()
    hist_m_train_seasonal.set_index(ind,inplace=True)
    RF = hist_m_train_seasonal.values/modelled
    return RF

def test(RF,modelled,date_test,hist): # Uses every data which is after the split date to test
    model_test = modelled*RF
    hist_m_test = hist.loc[hist.index >= pd.to_datetime(date_test[0])].loc[hist.loc[hist.index >= pd.to_datetime(date_test[0])].index <= date_test[1]]
    hist_m_test['month'] = hist_m_test.index.month
    hist_m_test_seasonal = hist_m_test.groupby('month').mean()
    hist_m_test_seasonal.set_index(ind,inplace=True)
    return model_test,hist_m_test_seasonal

#%% ============================ INPUT ========================================
gcm_list = ['MPI-M-MPI-ESM-LR','ICHEC-EC-EARTH','CNRM-CERFACS-CNRM-CM5','MOHC-HadGEM2-ES', 'NCC-NorESM1-M'] # General Circulation Model
rcm_list = ['RCA4','HIRHAM5'] # Regional Climate Model
rcp = '85' # Representative Concentration Pathways
date_trains = [["1991-1-1","1994-12-31"],["1995-1-1","1998-12-31"],["1999-1-1","2002-12-31"]]
date_tests = [["2003-1-1","2006-12-31"],["2007-1-1","2010-12-31"],["2011-1-1","2014-12-31"],["2015-1-1","2019-12-31"]]
#%% ============================= OUTPUT ======================================
plt.close('all')
country_name = ['Norway','Spain','Sweden']
country_iso_alpha_2 = ['NO','ES','SE'] 
color_list = [colors[0],colors[2],colors[3],colors[4],colors[5],colors[6],colors[7]]
nrmse = np.zeros([len(country_name),len(date_trains + date_tests)])
nrmse_1y =  np.zeros([len(country_name),len(date_trains + date_tests)])  #np.zeros([len(country_name),len(date_trains_1y + date_tests_1y)])
Z = []
df = []
for c in range(len(country_name)):
    fig1,ax1 = figure_formatting('','',1,1,color='k',figsiz = (18,12))
    gcm_it = 0
    count = 0
    country = country_iso_alpha_2[c]
    country_l = country_name[c]
    hist = inflhist(histdatadir,1991,2020,country,country_l)*1e-3
    hist = hist[(hist.T != 0).any()] # Removing rows with zeros
    ind = pd.date_range('2016/01/01','2016/12/31',freq='MS')
    hist_m_30y = hist
    hist_m_30y['month'] = hist_m_30y.index.month
    hist_m_30y_seasonal = hist_m_30y.groupby('month').mean()
    hist_m_30y_seasonal.set_index(ind,inplace=True)
    
    hist_m_15y = hist.loc[hist.index >= pd.to_datetime("2003-1-1")].loc[hist.loc[hist.index >= pd.to_datetime("2003-1-1")].index <= "2019-12-31"]
    hist_m_15y['month'] = hist_m_15y.index.month
    hist_m_15y_seasonal = hist_m_15y.groupby('month').mean()
    hist_m_15y_seasonal.set_index(ind,inplace=True)
    #-------------------------------- Plot natural variability between the considered periods
    dates = date_trains + date_tests
    print("")
    print(country_name[c])
    for k in range(len(dates)):           
        date = dates[k]
        hist_m_test = hist.loc[hist.index >= pd.to_datetime(date[0])].loc[hist.loc[hist.index >= pd.to_datetime(date[0])].index <= date[1]]
        hist_m_test['month'] = hist_m_test.index.month
        hist_m_test_seasonal = hist_m_test.groupby('month').mean()
        hist_test = hist_m_test_seasonal.set_index(ind)
        if k == 0:
            hist_test_df = hist_test
        else:
            hist_test_df[str(k)] = hist_test
            
        nrmse_1y[c,k] = (mean_squared_error(hist_m_30y_seasonal,hist_test)**0.5)/hist_m_30y_seasonal.mean()*100                    
        ax1[0].plot(hist_test,ls='-',linewidth=2, label = date[0][0:4] + ' - ' + date[1][0:4]) # + ', ' + 'RMSE=' + str(np.round(nrmse_1y[c,k],1)) + ' %')
    hist_test_df_min = hist_test_df.min(axis=1)
    hist_test_df_max = hist_test_df.max(axis=1)
    hist_test_df_range = hist_test_df_max - hist_test_df_min
    ax1[0].plot(hist_m_30y_seasonal.inflow,color='k',ls='-',linewidth=2, label = 'Historical mean')
    ax1[0].fill_between(hist_test_df_min.index,hist_test_df_min,hist_test_df_max,alpha=0.5,label='Historical range')
    upper = ((hist_test_df_max - hist_m_30y_seasonal.inflow)/(hist_m_30y_seasonal.inflow)*100).round(1).to_list() # hist_test_df.mean(axis=1)*100).round(1).to_list()
    lower = ((hist_m_30y_seasonal.inflow - hist_test_df_min)/(hist_m_30y_seasonal.inflow)*100).round(1).to_list() # hist_test_df.mean(axis=1)*100).round(1).to_list()    
    ax1[0].set_xlim([min(hist_test.index),max(hist_test.index)])
    ax1[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    
    if country == 'NO':
        ax1[0].set_aspect(5)
    elif country == 'ES':
        ax1[0].set_aspect(25)
    elif country == 'SE':
        ax1[0].set_aspect(10)
    
    ax1[0].set_ylabel('Inflow [TWh]')
    fig1.legend(prop=ticks_font,loc='lower center',ncol=3,borderaxespad=-0.5,frameon=False)
    fig1.savefig(figuredir + 'Cross_validation_historical_' + country_name[c] + '.png',bbox_inches='tight') 
    
    fig2,ax2 = figure_formatting('','',1,1,color='k',figsiz = (18,12))
    for k in range(len(dates)):           
        date = dates[k]
        hist_m_test = hist.loc[hist.index >= pd.to_datetime(date[0])].loc[hist.loc[hist.index >= pd.to_datetime(date[0])].index <= date[1]]
        hist_m_test['month'] = hist_m_test.index.month
        hist_m_test_seasonal = hist_m_test.groupby('month').mean()
        hist_test = hist_m_test_seasonal.set_index(ind)
        if k == 0:
            hist_test_df = hist_test
        else:
            hist_test_df[str(k)] = hist_test
        nrmse[c,k] = (mean_squared_error(hist_m_30y_seasonal,hist_test)**0.5)/hist_m_30y_seasonal.mean()*100
    hist_test_df_min = hist_test_df.min(axis=1)
    hist_test_df_max = hist_test_df.max(axis=1)
    hist_test_df_range = hist_test_df_max - hist_test_df_min
    ax2[0].fill_between(hist_test_df_min.index,hist_test_df_min,hist_test_df_max,alpha=0.5,label='Historical range',zorder=5)
    upper = ((hist_test_df_max - hist_m_30y_seasonal.inflow)/(hist_m_30y_seasonal.inflow.mean())*100).round(1).to_list() # hist_test_df.mean(axis=1)*100).round(1).to_list()
    lower = ((hist_m_30y_seasonal.inflow - hist_test_df_min)/(hist_m_30y_seasonal.inflow.mean())*100).round(1).to_list() # hist_test_df.mean(axis=1)*100).round(1).to_list()
    # -----------------------------------------------------------------------------------------
    rmse = np.zeros([len(gcm_list)*len(rcm_list),len(date_tests)*len(date_trains)])
    corr = np.zeros([len(gcm_list)*len(rcm_list),len(date_tests)*len(date_trains)])
    model_it = 0
    n_within_hist = []
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
            itera = 0
            for j in range(len(date_tests)):
                for i in range(len(date_trains)):                
                    date_train = date_trains[i]
                    date_test = date_tests[j]
                    infl_cal_m_train_seasonal = infl_cal_monthly[infl_cal_monthly.index >= pd.to_datetime(date_train[0])].loc[infl_cal_monthly[infl_cal_monthly.index >= pd.to_datetime(date_train[0])].index <= pd.to_datetime(date_train[1])].groupby('month').mean()
                    infl_cal_m_train_seasonal.set_index(ind,inplace=True)
                    infl_cal_m_test_seasonal = infl_cal_monthly[infl_cal_monthly.index >= pd.to_datetime(date_test[0])].loc[infl_cal_monthly[infl_cal_monthly.index >= pd.to_datetime(date_test[0])].index <= pd.to_datetime(date_test[1])].groupby('month').mean()
                    infl_cal_m_test_seasonal.set_index(ind,inplace=True)
                    RF = train(date_train,infl_cal_m_train_seasonal.values,hist)
                    if (gcm_it == 0 and rcm_it == 0 and i == 0 and j == 0):
                        model_test,hist_test= test(RF,infl_cal_m_test_seasonal,date_test,hist)
                    model_test[str(gcm_it) +',' + str(rcm_it) + ',' + str(j) + ',' + str(i)],hist_test= test(RF,infl_cal_m_test_seasonal,date_test,hist)
                    n_within_hist.append(len(model_test[ model_test[str(gcm_it) +',' + str(rcm_it) + ',' + str(j) + ',' + str(i)].between(hist_test_df_min,hist_test_df_max)]))
                    nrmse_val = (mean_squared_error(hist_test,model_test[str(gcm_it) +',' + str(rcm_it) + ',' + str(j) + ',' + str(i)])**0.5)/hist_test.mean()*100
                    if (gcm_it == 0 and rcm_it == 0):
                        ax2[0].plot(model_test[str(gcm_it) +',' + str(rcm_it) + ',' + str(j) + ',' + str(i)],ls='--',linewidth=1.25, color=colors[itera],label = 'Model test ' + str(itera+1)) # label = gcm + '-' + rcm + ' for ' + date_tests[j][0][0:4] + '-' + date_tests[j][1][0:4] + ' trained w. ' + date_trains[i][0][0:4] + ' - ' + date_trains[i][1][0:4]) # + ', NRMSE = ' + str(np.round(nrmse_val.item(),1)) + ' %')
                    else:
                        ax2[0].plot(model_test[str(gcm_it) +',' + str(rcm_it) + ',' + str(j) + ',' + str(i)],ls='--',linewidth=1.25, color=colors[itera]) # label = gcm + '-' + rcm + ' for ' + date_tests[j][0][0:4] + '-' + date_tests[j][1][0:4] + ' trained w. ' + date_trains[i][0][0:4] + ' - ' + date_trains[i][1][0:4]) # + ', NRMSE = ' + str(np.round(nrmse_val.item(),1)) + ' %')
                    rmse[model_it,itera] = ((mean_squared_error(hist_m_15y_seasonal,model_test[str(gcm_it) +',' + str(rcm_it) + ',' + str(j) + ',' + str(i)])**0.5).item())
                    corr[model_it,itera] = model_test[str(gcm_it) +',' + str(rcm_it) + ',' + str(j) + ',' + str(i)].corr(hist_m_15y_seasonal['inflow'])                    
                    itera += 1
            model_it += 1
            count += 1
            rcm_it += 1
        gcm_it += 1
    
    for test_no in range(12):
        print('Model test ' + str(test_no+1) + ':')
        print(str((rmse[:,test_no].mean()/hist_test.mean()*100).round(1).item()) + ' %' )
        print(str(corr[:,test_no].mean().round(2).item()))

    print('MEAN TEST RESULT:')
    print(str((rmse.mean()/hist_test.mean()*100).round(1).item()) + ' %')
    ax2[0].set_xlim([min(hist_test.index),max(hist_test.index)])
    ax2[0].set_ylabel('Inflow [TWh]')
    ax2[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    model_test_min = model_test.min(axis=1)
    model_test_max = model_test.max(axis=1)
    if country == 'NO':
        ax2[0].set_aspect(3)
    elif country == 'ES':
        ax2[0].set_aspect(12)
    elif country == 'SE':
        ax2[0].set_aspect(6)
    
    Z.append(np.round(sum(n_within_hist)/(120*12)*100,1))
    ax2[0].plot(hist_m_15y_seasonal.inflow,color='k',ls='-',linewidth=2, label = 'Historical mean (test)',zorder=10)
    fig2.legend(prop=ticks_font,loc='lower center',ncol=3,borderaxespad=-0.5,frameon=False)
        
    upper_mod = (((model_test_max - hist_m_15y_seasonal.inflow)/hist_m_15y_seasonal.inflow.mean())*100).round(1).to_list() # hist_test_df.mean(axis=1)*100).round(1).to_list()
    lower_mod = (((hist_m_15y_seasonal.inflow - model_test_min)/hist_m_15y_seasonal.inflow.mean())*100).round(1).to_list() # hist_test_df.mean(axis=1)*100).round(1).to_list()
    fig2.savefig(figuredir + 'Cross_validation_' + country_name[c] + '.png',bbox_inches='tight') 

    model_test.drop(columns='inflow',inplace=True)
    hist_test_df.drop(columns='inflow',inplace=True)
    df.append((model_test.sum() - hist_test_df.sum().mean())/hist_test_df.sum().mean()*100)
        