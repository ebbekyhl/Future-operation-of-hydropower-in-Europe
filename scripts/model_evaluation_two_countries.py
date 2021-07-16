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
figure_font = fm.FontProperties(fname='AUfonts/AUPassata_Bold.ttf',size=28)
ticks_font = fm.FontProperties(fname='AUfonts/AUPassata_Rg.ttf',size=28)
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
rcp = '85' # Representative Concentration Pathways
#%% ============================= OUTPUT ======================================
country_name = ['Norway','Spain']
country_iso_alpha_2 = ['NO','ES'] 
nrows = 2
ncols = 2
fig,ax,ax_twin = figure_formatting('','',nrows,ncols,color='k',figsiz = (18,16),ylab_twin='',color_twin=colors[6],twin=True)
fig.subplots_adjust(hspace=-0.1)
fig.subplots_adjust(wspace=0.3)
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
    if c == 0:
        ax[c].plot(hist_m_train_seasonal,label='Historical',color='k',zorder=20,lw=3,linestyle='--')
        ax[c].set_ylabel('Historical inflow [TWh]')
    else:
        ax[c].plot(hist_m_train_seasonal,color='k',zorder=5,lw=3,linestyle='--')
        
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
            if c == 0:
                ax_twin[c].plot(infl_cal_m_train_seasonal,color=colors[1+count],label=gcm + '-' + rcm)
            else:
                ax_twin[c].plot(infl_cal_m_train_seasonal,color=colors[1+count])
                ax_twin[c].set_ylabel('Modelled inflow [TWh]',labelpad=8)     
            
            RF = hist_m_train_seasonal.values/infl_cal_m_train_seasonal.values
            ax[c+2].plot(infl_cal_m_train_seasonal.index,RF,color=colors[1+count],marker='o', linestyle='dashed',markersize=7.5)
            count += 1
            rcm_it += 1
        gcm_it += 1
    ax[c].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax[c].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax[c].set_title(country_l + ' (a)',fontproperties=figure_font)
    ax[c+2].set_title(country_l + ' (b)',fontproperties=figure_font)
    ax[c].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax[c+2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    # ax[c+2].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax[c+2].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    # ax[c+2].set_ylabel('Retain Factor [-]')
    ax[c+2].tick_params(axis='y', colors='k')
    ax[c].set_xlim([min(ind),max(ind)])
    ax[c+2].set_xlim([min(ind),max(ind)])
    ax_twin[c+2].remove()
    ax[c].grid(linestyle=':', linewidth='0.75', color='grey',which='both',axis='x')  
    ax[c+2].grid(linestyle=':', linewidth='0.75', color='grey',which='both',axis='x')  

    ax[c].axes.set_box_aspect(0.8) # requires matplotlib 3.4
    ax[c+2].axes.set_box_aspect(0.8) # requies matplotlib 3.4

ax[2].set_ylabel('Monthly retain Factor [-]',color='k')

fig.legend(frameon=False,loc='lower center',ncol=2,prop=ticks_font,handletextpad=1,labelspacing=0,borderaxespad=-0.5)

if len(country_name) == 2:
    fig.savefig(figuredir + 'Model_evaluation_' + country_name[0] + '_' + country_name[1] + '_' + rcp + '_TEST.png',bbox_inches='tight') 
