# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:36:00 2020
@author: ebbek
The output of this script is a map of the climate change impact of 22 European 
countries (if Wattsight data is included), given by the relative change in the 
mean annual inflow and by the change in the seasonal profile. Furthermore, a 
figure shows the probability distributions of the annual inflow at the BOC 
and EOC. Modules versions:
numpy 1.19.1, matplotlib 3.1.0, pandas 1.0.3, scipy 1.5.2

Inputs:
    gendatadir: General data directory
    moddatadir: Modelled inflow data directory
    resdatadir: Calibrated modelled inflow data directory
    histdatadir: Historical inflow data directory
    figuredir: Figure data directory
    hydrotype: Type of hydroelectric power plants included
    gcm_list: List of general circulation models
    rcm_list: List of regional climate models
    rcp_list: List of representative concentration pathways
    WI: Wattsight data included (1) or not (0)
    matrix: GCM-RCM matrix included (1) or not (0)
Output:
    Figures:
    fig_EM: Map of Europe illustrating the grand ensemble mean (GEM) prediction  
    fig: GCM-RCM inflow 5x2 matrix 
    f_d: Figure presenting the paired t-test based on the GEM
    f_cm: Annual inflow distributions at BOC and EOC for Norway
    fig3: Historical model evaulation based on seasonal profile for each climate model
    f4: Figure illustrating frequency and duration of droughts
    f5: Figure illustrating frequency and duration of overflow
    (All figures are saved in the folder "figure")

    CSV-files:
    hist_woa: Daily climate model specific inflow from 1991 to 2020
    cc_woa: Daily climate model specific inflow from 2071 to 2100
    hist_EM: Daily ensemble mean inflow time series for an average year between 1991 to 2020
    cc_EM: Daily ensemble mean inflow time series for an average year between 2071 to 2100
    (CSV-files are saved in the folder "resdata"
"""
from inflhist import inflhist
from inflforecast import inflforecast
from extreme_events import extreme_events_count
from extreme_events import extreme_events_plot
from figure_formatting import figure_formatting
from AUcolor import AUcolor
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.basemap import Basemap
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager as fm
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from datetime import datetime
from IPython import get_ipython
import pandas as pd
import scipy.stats as sp
import warnings
warnings.filterwarnings("ignore")
colors,color_names = AUcolor()
title_font = fm.FontProperties(fname='AUfonts/AUPassata_Bold.ttf',size=22)
subtitle_font = fm.FontProperties(fname='AUfonts/AUPassata_Rg.ttf',size=14)
figure_font = fm.FontProperties(fname='AUfonts/AUPassata_Bold.ttf',size=20)
ticks_font = fm.FontProperties(fname='AUfonts/AUPassata_Rg.ttf',size=14)
ticks_font_2 = fm.FontProperties(fname='AUfonts/AUPassata_Rg.ttf',size=16)
ticks_font_3 = fm.FontProperties(fname='AUfonts/AUPassata_Rg.ttf',size=20)
country_code_font = fm.FontProperties(fname='AUfonts/AUPassata_Bold.ttf',size=10)
country_code_font_2 = fm.FontProperties(fname='AUfonts/AUPassata_Bold.ttf',size=5)
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
rcp_list = ['85'] #['26','45','85'] # Representative Concentration Pathways. To save time, it is possible to select only one rcp scenario, e.g. rcp_list = ['85']
hydrotype = 'HDAM' # Type of hydropower plant (HDAM = reservoir, HROR = run-of-river, HPHS = pumped hydro, all = all three types included). So far only HDAM has been run, but executing the script "Atlite_inflow" can provide simulations for remaining types.
WI = 0 # Wattsight + ENTSO-E historial data (1) or ENTSO-E only (0). The openly available historical data will only contain ENTSO-E. 
matrix = 0 # Whether inflow 5x2 GCM-RCM matrix is created (1) or not (0). If 1, the script becomes considerably more time consuming.
#%% ============================= OUTPUT ======================================
if WI == 1:    
    country_seasonal_plot = ['NO','FR','ES','SE','DE','CH','IT','AT'] # countries for which the seasonal profile is plotted
    country_name = ['Norway','France','Spain','Switzerland','Sweden','Germany','Austria','Italy',
                    'Bulgaria','Croatia','Portugal', 'Romania','Czech_Republic', 'Hungary',
                    'Bosnia_and_Herzegovina','Serbia','Slovenia','Finland','Poland','Slovakia', 
                    'North_Macedonia','Montenegro'] # countries for which inflow timeseries are modelled
    country_iso_alpha_2 = ['NO','FR','ES','CH','SE','DE','AT','IT','BG','HR','PT','RO',
                            'CZ','HU','BA','RS','SI','FI','PL','SK','MK','ME'] 
    country_iso_alpha_3 = ['NOR','FRA','ESP','CHE','SWE','DEU','AUT','ITA','BGR',
                            'HRV','PRT','ROU','CZE','HUN',
                            'BIH','SRB','SVN','FIN','POL', 'SVK','MKD','MNE']
    nrows = 8 # Number of rows in figure
    ncols = 3 # Number of columns in figure
    lp = -3 # Distance between axis label and axis
else:
    country_seasonal_plot = ['NO','FR','ES','SE','RO','CH','IT','AT'] # countries for which the seasonal profile is plotted
    country_name = ['Norway','France','Spain','Switzerland','Sweden','Austria','Italy','Romania',
                    'Bulgaria','Portugal', 'Montenegro', 'Serbia'] # countries for which inflow timeseries are modelled
    country_iso_alpha_2 = ['NO','FR','ES','CH','SE','AT','IT','RO','BG','PT','ME','RS'] 
    country_iso_alpha_3 = ['NOR','FRA','ESP','CHE','SWE','AUT','ITA','ROU','BGR','PRT','MNE','SRB']
    nrows = 4 # Number of rows in figure
    ncols = 3 # Number of columns in figure
    lp = 0 # Distance between axis label and axis

rcp_dic = {'26':(0,52/255,102/255),'45':(112/255,160/255,205/255),'85':(153/255,0,2/255)} # RCP colors consistent with IPCC layout
IPCCpreccolors = [(84/255,48/255,5/255) ,(245/255,245/255,245/255),(0,60/255,48/255)] # Water resources colormap consistent with IPCC layout
cmap = LinearSegmentedColormap.from_list(
        'IPCCprec', IPCCpreccolors, N=50)
months = ['January','February','March','April','May','June','July',
              'August','September','October','November','December']    
country_coord = pd.read_csv(gendatadir + 'Countries_lat_lon.csv',
                            sep=';',header=None) # Country center coordinates
country_coord.columns = ['Code','Lat','Lon','Country']
inflow_annual = np.zeros((2,len(gcm_list),len(rcm_list),len(rcp_list),len(country_name),30)) # Initialise list of annual inflow
gcm_short_dic = {'MPI-M-MPI-ESM-LR':'MPI-ESM-LR','ICHEC-EC-EARTH':'EC-EARTH','CNRM-CERFACS-CNRM-CM5':'CNRM-CM5','NCC-NorESM1-M':'NorESM1-M', 'MOHC-HadGEM2-ES':'HadGEM2-ES'}
rcp_it = 0
for rcp in rcp_list:
    sns.reset_defaults()
    mpl.rc('hatch', color='k', linewidth=0.25)
    if matrix == 1:
        fig, ax = plt.subplots(5,2,figsize=(10,20)) # Initialise figure
        fig.subplots_adjust(wspace=-0.375)
        fig.subplots_adjust(hspace=0)
        norm=plt.Normalize(-70,70)
    count = 0 # Count of total loops per rcp scenario
    hist_woa = {}  # Initialise dictionary containing modelled inflow at BOC
    cc_woa = {} # Initialise dictionary containing modelled inflow at EOC
    inflow_annual_rel_change = np.zeros([len(country_name),len(gcm_list)*len(rcm_list)]) # Initialise list of predicted relative changes in annual inflow
    model_name = [0]*len(gcm_list)*len(rcm_list) # Initialise list of climate model name
    gcm_it = 0 # GCM loop iterator
    for gcm in gcm_list:
        gcm_short = gcm_short_dic[gcm]
        rcm_it = 0
        for rcm in rcm_list:
            print('Reading and plotting modelled inflow in ' + str(len(country_name)) + ' countries from climate model ' + gcm + '-' + rcm + '-rcp' + str(int(rcp)/10)) 
            inflow_plot = np.zeros([len(country_iso_alpha_2)])
            try:
                hist_woa[count],cc_woa[count] = inflforecast(histdatadir,gendatadir,moddatadir,resdatadir,gcm,rcm,rcp,country_name,country_iso_alpha_2) # modelled inflow at BOC and EOC 
                hist_woa[count] = hist_woa[count]*1e-3 # BOC inflow in TWh
                cc_woa[count] = cc_woa[count]*1e-3 # EOC inflow in TWh
                hist_woa[count]['day_of_year'] = hist_woa[count].index.dayofyear
                cc_woa[count]['day_of_year'] = cc_woa[count].index.dayofyear
                if matrix == 1:
                    m = Basemap(width=11500000/3,height=9000000/2.3,projection='laea',
                                resolution='i',lat_0=54.5,lon_0=12)
                    
                    m_plot = Basemap(width=11500000/3,height=9000000/2.3,projection='laea',
                                     resolution='i',lat_0=54.5,lon_0=12,ax=ax[gcm_it][rcm_it])
                    m_plot.drawcountries(zorder=3,color=colors[18],linewidth=0.05)
                    print("line164")
                for c in range(len(country_iso_alpha_2)):
                    country = country_iso_alpha_2[c]
                    inflow_annual[0][gcm_it][rcm_it][rcp_it][c][:] = hist_woa[count][country].groupby(pd.Grouper(freq='y')).sum().values # Annual inflow at BOC in TWh
                    inflow_annual[1][gcm_it][rcm_it][rcp_it][c][:] = cc_woa[count][country].groupby(pd.Grouper(freq='y')).sum().values # Annual inflow at EOC in TWh
                    if matrix == 1:
                        mu0 = hist_woa[count][country].groupby(pd.Grouper(freq='y')).sum().mean() # mean annual inflow BOC
                        mu1 = cc_woa[count][country].groupby(pd.Grouper(freq='y')).sum().mean() # mean annual inflow EOC
                        inflow_plot[c] = (mu1 - mu0)/mu0*100 # change in mean annual inflow at EOC relative to BOC
                        inflow_annual_rel_change[c,count] = (mu1 - mu0)/mu0*100 # change in mean annual inflow at EOC relative to BOC
                        model_name[count] = gcm_short + '-' + rcm   
                        #color countries corresponding to relative change in mean annual inflow
                        #Shape files: https://www.gadm.org/download_country_v3.html
                        m.readshapefile('shapefiles/gadm36_' + country_iso_alpha_3[c] + '_0',country_name[c],drawbounds=False)
                        patches = []
                        for info, shape in zip(eval('m.' + country_name[c] + '_info'), eval('m.' + country_name[c])):
                            patches.append(Polygon(np.array(shape), True))
                        patch1=ax[gcm_it][rcm_it].add_collection(PatchCollection(patches, facecolor= cmap(norm(inflow_plot[c])), linewidths=0, zorder=2))
                        CC_plot = country_coord[country_coord.Code == country]
                        # if inflow_plot[c] < 0:
                        #     ax[gcm_it][rcm_it].annotate(str(inflow_plot[c].round(1)) + '%',xy=m(np.array(CC_plot.Lon.item()-1), np.array(CC_plot.Lat.item())),color='k',fontproperties = country_code_font_2, zorder=13)
                        # else:
                        #     ax[gcm_it][rcm_it].annotate('+' + str(inflow_plot[c].round(1)) + '%',xy=m(np.array(CC_plot.Lon.item()-1), np.array(CC_plot.Lat.item())),color='k',fontproperties = country_code_font_2, zorder=13)
                if matrix == 1:
                    ax[gcm_it][rcm_it].text(0.01, 0.94,gcm_short + '-' + rcm, transform=ax[gcm_it][rcm_it].transAxes,fontproperties=ticks_font)
                count += 1
            except:
                if matrix == 1:
                    m = Basemap(width=11500000/3,height=9000000/2.3,projection='laea',
                                resolution='i',lat_0=54.5,lon_0=12)
                    
                    m_plot = Basemap(width=11500000/3,height=9000000/2.3,projection='laea',
                                      resolution='i',lat_0=54.5,lon_0=12,ax=ax[gcm_it][rcm_it])
                    m_plot.drawcountries(zorder=3,color=colors[18],linewidth=0.05)
                    ax[gcm_it][rcm_it].patch.set_facecolor(color='white')
                    ax[gcm_it][rcm_it].text(0.01, 0.94,gcm_short + '-' + rcm, transform=ax[gcm_it][rcm_it].transAxes,fontproperties=ticks_font,zorder=5)
                    ax[gcm_it][rcm_it].patch.set_alpha(0.5)
                    # ax[gcm_it][rcm_it].patch.set_hatch('///')     
                    print("line202")
                for c in range(len(country_iso_alpha_2)):
                    country = country_iso_alpha_2[c]
                    inflow_annual[0][gcm_it][rcm_it][rcp_it][c][:] = np.nan
                    inflow_annual[1][gcm_it][rcm_it][rcp_it][c][:] = np.nan
            rcm_it += 1
        gcm_it += 1
    if matrix == 1:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(inflow_plot)
        cax = fig.add_axes([ax[2][0].get_position().x0, 
                            0.085, 
                            ax[2][1].get_position().x1-ax[2][0].get_position().x0,
                            0.02])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label('Change in annual inflow [%]',fontproperties = ticks_font,zorder=10)
        fig.savefig(figuredir + 'Hydro_reservoir_energy_inflow_map_GCM_RCM_matrix_' + rcp + '.png',bbox_inches='tight')  
        inflow_annual_rel_change_df = pd.DataFrame(inflow_annual_rel_change).round(1)
        inflow_annual_rel_change_df['countries'] = country_name
        inflow_annual_rel_change_df.set_index('countries',inplace=True)
        inflow_annual_rel_change_df.columns = model_name
        inflow_annual_rel_change_df.to_csv(resdatadir + 'inflow_relative_changes_matrix_rcp' + rcp + '.csv')
    #%% Analysis of variance 1 (Interannual variability at BOC and EOC)
    sigma_interannual_boc_mean = [0]*len(country_name)
    sigma_interannual_eoc_mean = [0]*len(country_name)
    c_dist_it = 0
    c_dist_list = [0,2]
    if rcp == '85':
        E_mean = np.zeros([len(gcm_list),len(rcm_list)])
        # =========================== Interannual variability ==========================
        sigma_interannual_boc = np.zeros([len(country_name),len(gcm_list),len(rcm_list)])
        sigma_interannual_eoc = np.zeros([len(country_name),len(gcm_list),len(rcm_list)])
        for c in range(len(country_name)): 
            for i in range(len(gcm_list)):
                for j in range(len(rcm_list)):
                    sigma_interannual_boc[c,i,j] = np.std(inflow_annual[0][i][j][rcp_it][c][:]).round(1) 
                    sigma_interannual_eoc[c,i,j] = np.std(inflow_annual[1][i][j][rcp_it][c][:]).round(1)  
            sigma_interannual_boc_mean[c] = np.mean(sigma_interannual_boc[c]).round(1)
            sigma_interannual_eoc_mean[c] = np.mean(sigma_interannual_eoc[c]).round(1)
        f_cm,ax_cm = figure_formatting('TWh','PDF',1,2,figsiz = (18,8))
        # =============================== Distribution plot =============================
        for c_dist in c_dist_list:
            for i in range(len(gcm_list)):
                for j in range(len(rcm_list)):
                    if j == 0 and i == 0:
                        sns.distplot(inflow_annual[0][i][j][rcp_it][c_dist][:],ax=ax_cm[c_dist_it], label='BOC',hist_kws={"color":colors[0]}, kde_kws={"color":colors[0],"linewidth":1.5}) #,color='royalblue')
                        sns.distplot(inflow_annual[1][i][j][rcp_it][c_dist][:],ax=ax_cm[c_dist_it],label='EOC ' + gcm_list[i] + '-' + rcm_list[j],hist_kws={"color":colors[4]}, kde_kws={"color":colors[4],"linewidth":1.5}) #color='orange')
                    else:
                        sns.distplot(inflow_annual[0][i][j][rcp_it][c_dist][:],ax=ax_cm[c_dist_it],hist_kws={"color":colors[0]}, kde_kws={"color":colors[0],"linewidth":1.5})
                        sns.distplot(inflow_annual[1][i][j][rcp_it][c_dist][:],ax=ax_cm[c_dist_it],hist_kws={"color":colors[4]}, kde_kws={"color":colors[4],"linewidth":1.5})                    
                    E_mean[i,j] = inflow_annual[1][i][j][rcp_it][c_dist][:].mean().round(1)
            sigma_interannual_rel_boc = np.mean(sigma_interannual_boc[c_dist]/E_mean*100).round(1)
            sigma_interannual_rel_eoc = np.mean(sigma_interannual_eoc[c_dist]/E_mean*100).round(1)
            # =================== Inter-climate model variability =========================
            E_list = [inflow_annual[1][0][0][rcp_it][c_dist][:].mean(),
                        inflow_annual[1][1][0][rcp_it][c_dist][:].mean(),
                        inflow_annual[1][2][0][rcp_it][c_dist][:].mean(),
                        inflow_annual[1][3][0][rcp_it][c_dist][:].mean(),
                        inflow_annual[1][4][0][rcp_it][c_dist][:].mean(),
                        inflow_annual[1][0][1][rcp_it][c_dist][:].mean(),
                        inflow_annual[1][1][1][rcp_it][c_dist][:].mean(),
                        inflow_annual[1][2][1][rcp_it][c_dist][:].mean(),
                        inflow_annual[1][3][1][rcp_it][c_dist][:].mean(),
                        inflow_annual[1][4][1][rcp_it][c_dist][:].mean()]
            sigma_interclimate_eoc = np.std(E_list).round(1)
            sigma_interclimate_rel_eoc = (sigma_interclimate_eoc/np.mean(E_list)*100).round(1)
            # =============================================================================
            ax_cm[c_dist_it].text(0.01, 0.95, r'$\bar{\sigma}_{y,BOC} = $' +  str(sigma_interannual_rel_boc) + ' %', transform=ax_cm[c_dist_it].transAxes,fontproperties=ticks_font_3)
            ax_cm[c_dist_it].text(0.01, 0.9, r'$\bar{\sigma}_{y,EOC} = $' +  str(sigma_interannual_rel_eoc) + ' %', transform=ax_cm[c_dist_it].transAxes,fontproperties=ticks_font_3)
            ax_cm[c_dist_it].text(0.01, 0.84, r'$\sigma_{GCM-RCM,EOC} = $' +  str(sigma_interclimate_rel_eoc) + ' %', transform=ax_cm[c_dist_it].transAxes,fontproperties=ticks_font_3)
            ax_cm[c_dist_it].set_aspect(1./ax_cm[c_dist_it].get_data_ratio())
            ax_cm[c_dist_it].set_title(country_name[c_dist],fontproperties=figure_font)
            c_dist_it += 1
        f_cm.legend(['BOC','EOC'],prop=ticks_font_3,frameon=False,loc='lower center',ncol=2)
        f_cm.subplots_adjust(wspace=0.15)
        f_cm.savefig(figuredir + 'Annual_inflow_distributions_1_' + country_name[c_dist_list[0]] + '_' + country_name[c_dist_list[1]] + '_' + rcp + '.png',bbox_inches='tight')  
#%% Analysis of variance 2 (Inter-RCM variability)
    # sns.set_palette(sns.color_palette(list(colors)))   
    palette = [colors[0],colors[2],colors[3],colors[5],colors[6]]
    sns.set_palette(sns.color_palette(palette))   
    c_dist_it = 0
    c_dist_list = [0,2] # 0 = Norway, 2 = Spain
    if rcp == '85':
        E_mean = np.zeros([len(gcm_list),len(rcm_list)])
        f_cm1,ax_cm1 = figure_formatting('TWh','PDF',1,2,figsiz = (18,8))
        # =============================== Distribution plot =============================
        for c_dist in c_dist_list:
            sigma_eoc_mean = sigma_interannual_eoc[c_dist].mean(axis=0).round(1)
            for i in range(len(gcm_list)):
                for j in range(len(rcm_list)):
                    if j == 0:
                        sns.distplot(inflow_annual[1][i][j][rcp_it][c_dist][:],ax=ax_cm1[c_dist_it],hist=False,kde_kws={"linewidth":1.5},label='EOC ' + gcm_list[i] + '-' + rcm_list[j])
                    else:
                        sns.distplot(inflow_annual[1][i][j][rcp_it][c_dist][:],ax=ax_cm1[c_dist_it],hist=False,kde_kws={"linewidth":1.5},label='EOC ' + gcm_list[i] + '-' + rcm_list[j])
                    E_mean[i,j] = inflow_annual[1][i][j][rcp_it][c_dist][:].mean().round(1)
                ax_cm1[c_dist_it].set_prop_cycle(None)

            ax_cm1[c_dist_it].text(0.99, 0.95, r'$\bar{\sigma}_{y,EOC,RCA4} = $' +  str(sigma_eoc_mean[0]) + ' TWh', transform=ax_cm1[c_dist_it].transAxes,fontproperties=ticks_font_3,ha='right',color=palette[0])
            ax_cm1[c_dist_it].text(0.99, 0.9, r'$\bar{\sigma}_{y,EOC,HIRHAM5} = $' +  str(sigma_eoc_mean[1]) + ' TWh', transform=ax_cm1[c_dist_it].transAxes,fontproperties=ticks_font_3,ha='right',color=palette[1])
            ax_cm1[c_dist_it].text(0.99, 0.84, r'$\bar{E}_{EOC,RCA4} = $' +  str(E_mean.mean(axis=0)[0].round(1)) + ' TWh', transform=ax_cm1[c_dist_it].transAxes,fontproperties=ticks_font_3,ha='right',color=palette[0])
            ax_cm1[c_dist_it].text(0.99, 0.78, r'$\bar{E}_{EOC,HIRHAM5} = $' +  str(E_mean.mean(axis=0)[1].round(1)) + ' TWh', transform=ax_cm1[c_dist_it].transAxes,fontproperties=ticks_font_3,ha='right',color=palette[1])
            #ax_cm1[c_dist_it].get_legend().remove()
            ax_cm1[c_dist_it].set_aspect(1./ax_cm1[c_dist_it].get_data_ratio())
            ax_cm1[c_dist_it].set_title(country_name[c_dist],fontproperties=figure_font)
            c_dist_it += 1
        f_cm1.legend([rcm_list[0],rcm_list[1]],prop=ticks_font_3,frameon=False,loc='lower center',ncol=2)
        f_cm1.subplots_adjust(wspace=0.15)
        f_cm1.savefig(figuredir + 'Annual_inflow_distributions_2_' + country_name[c_dist_list[0]] + '_' + country_name[c_dist_list[1]] + '_' + rcp + '.png',bbox_inches='tight')  
#%% Analysis of variance 3 (Inter-GCM variability)
    c_dist_it = 0
    c_dist_list = [0,2]
    if rcp == '85':
        E_mean = np.zeros([len(gcm_list),len(rcm_list)])
        f_cm2,ax_cm2 = figure_formatting('TWh','PDF',1,2,figsiz = (18,9))
        # =============================== Distribution plot =============================
        for c_dist in c_dist_list:
            sigma_eoc_mean = sigma_interannual_eoc[c_dist].mean(axis=1).round(1)
            for j in range(len(rcm_list)):
                for i in range(len(gcm_list)):
                    E_mean[i,j] = inflow_annual[1][i][j][rcp_it][c_dist][:].mean().round(1)
                    sns.distplot(inflow_annual[1][i][j][rcp_it][c_dist][:],hist=False,ax=ax_cm2[c_dist_it],kde_kws={"linewidth":1.5},label='EOC ' + gcm_list[i] + '-' + rcm_list[j])
                ax_cm2[c_dist_it].set_prop_cycle(None)

            ax_cm2[c_dist_it].text(0.98, 0.95, r'$\bar{\sigma}_{y,EOC,MPI} = $' +  str(sigma_eoc_mean[0]) + ' TWh', transform=ax_cm2[c_dist_it].transAxes,fontproperties=ticks_font_3,ha='right',color=palette[0])
            ax_cm2[c_dist_it].text(0.98, 0.90, r'$\bar{\sigma}_{y,EOC,ICHEC} = $' +  str(sigma_eoc_mean[1]) + ' TWh', transform=ax_cm2[c_dist_it].transAxes,fontproperties=ticks_font_3,ha='right',color=palette[1])
            ax_cm2[c_dist_it].text(0.98, 0.85, r'$\bar{\sigma}_{y,EOC,CNRM} = $' +  str(sigma_eoc_mean[2]) + ' TWh', transform=ax_cm2[c_dist_it].transAxes,fontproperties=ticks_font_3,ha='right',color=palette[2])
            ax_cm2[c_dist_it].text(0.98, 0.80, r'$\bar{\sigma}_{y,EOC,MOHC} = $' +  str(sigma_eoc_mean[3]) + ' TWh', transform=ax_cm2[c_dist_it].transAxes,fontproperties=ticks_font_3,ha='right',color=palette[3])
            ax_cm2[c_dist_it].text(0.98, 0.75, r'$\bar{\sigma}_{y,EOC,NCC} = $' +  str(sigma_eoc_mean[4]) + ' TWh', transform=ax_cm2[c_dist_it].transAxes,fontproperties=ticks_font_3,ha='right',color=palette[4])
            ax_cm2[c_dist_it].text(0.98, 0.68, r'$\bar{E}_{EOC,MPI} = $' +  str(E_mean.mean(axis=1)[0].round(1)) + ' TWh', transform=ax_cm2[c_dist_it].transAxes,fontproperties=ticks_font_3,ha='right',color=palette[0])
            ax_cm2[c_dist_it].text(0.98, 0.63, r'$\bar{E}_{EOC,ICHEC} = $' +  str(E_mean.mean(axis=1)[1].round(1)) + ' TWh', transform=ax_cm2[c_dist_it].transAxes,fontproperties=ticks_font_3,ha='right',color=palette[1])
            ax_cm2[c_dist_it].text(0.98, 0.58, r'$\bar{E}_{EOC,CNRM} = $' +  str(E_mean.mean(axis=1)[2].round(1)) + ' TWh', transform=ax_cm2[c_dist_it].transAxes,fontproperties=ticks_font_3,ha='right',color=palette[2])
            ax_cm2[c_dist_it].text(0.98, 0.53, r'$\bar{E}_{EOC,MOHC} = $' +  str(E_mean.mean(axis=1)[3].round(1)) + ' TWh', transform=ax_cm2[c_dist_it].transAxes,fontproperties=ticks_font_3,ha='right',color=palette[3])
            ax_cm2[c_dist_it].text(0.98, 0.48, r'$\bar{E}_{EOC,NCC} = $' +  str(E_mean.mean(axis=1)[4].round(1)) + ' TWh', transform=ax_cm2[c_dist_it].transAxes,fontproperties=ticks_font_3,ha='right',color=palette[4])
            #ax_cm2[c_dist_it].get_legend().remove()
            ax_cm2[c_dist_it].set_aspect(1./ax_cm2[c_dist_it].get_data_ratio())
            ax_cm2[c_dist_it].set_title(country_name[c_dist],fontproperties=figure_font)
            c_dist_it += 1
        f_cm2.legend([gcm_list[0],gcm_list[1],gcm_list[2],gcm_list[3],gcm_list[4]],prop=ticks_font_3,frameon=False,loc='lower center',ncol=3,labelspacing=0,borderaxespad=-0.5)
        f_cm2.subplots_adjust(wspace=0.22)
        f_cm2.savefig(figuredir + 'Annual_inflow_distributions_3_' + country_name[c_dist_list[0]] + '_' + country_name[c_dist_list[1]] + '_' + rcp + '.png',bbox_inches='tight')  
    #%%
    elif rcp == '26':
        for c in range(len(country_name)): 
            sigma_interannual_boc_mean[c] = np.mean([np.std(inflow_annual[0][0][0][rcp_it][c][:]),
                                            np.std(inflow_annual[0][1][0][rcp_it][c][:]),
                                            np.std(inflow_annual[0][3][0][rcp_it][c][:]),
                                            np.std(inflow_annual[0][4][0][rcp_it][c][:]),
                                            np.std(inflow_annual[0][1][1][rcp_it][c][:]),
                                            np.std(inflow_annual[0][3][1][rcp_it][c][:])]).round(1)
            sigma_interannual_eoc_mean[c] = np.mean([np.std(inflow_annual[1][0][0][rcp_it][c][:]),
                                            np.std(inflow_annual[1][1][0][rcp_it][c][:]),
                                            np.std(inflow_annual[1][3][0][rcp_it][c][:]),
                                            np.std(inflow_annual[1][4][0][rcp_it][c][:]),
                                            np.std(inflow_annual[1][1][1][rcp_it][c][:]),
                                            np.std(inflow_annual[1][3][1][rcp_it][c][:])]).round(1)
    elif rcp == '45':
        for c in range(len(country_name)): 
            sigma_interannual_boc_mean[c] = np.mean([np.std(inflow_annual[0][0][0][rcp_it][c][:]),
                                             np.std(inflow_annual[0][1][0][rcp_it][c][:]),
                                             np.std(inflow_annual[0][2][0][rcp_it][c][:]),
                                             np.std(inflow_annual[0][3][0][rcp_it][c][:]),
                                             np.std(inflow_annual[0][4][0][rcp_it][c][:]),
                                             np.std(inflow_annual[0][1][1][rcp_it][c][:]),
                                             np.std(inflow_annual[0][3][1][rcp_it][c][:]),
                                             np.std(inflow_annual[0][4][1][rcp_it][c][:])]).round(1)
            sigma_interannual_eoc_mean[c] = np.mean([np.std(inflow_annual[1][0][0][rcp_it][c][:]),
                                             np.std(inflow_annual[1][1][0][rcp_it][c][:]),
                                             np.std(inflow_annual[1][2][0][rcp_it][c][:]),
                                             np.std(inflow_annual[1][3][0][rcp_it][c][:]),
                                             np.std(inflow_annual[1][4][0][rcp_it][c][:]),
                                             np.std(inflow_annual[1][1][1][rcp_it][c][:]),
                                             np.std(inflow_annual[1][3][1][rcp_it][c][:]),
                                             np.std(inflow_annual[1][4][1][rcp_it][c][:])]).round(1)
    #%% Paired t-test to evaluate significance of change in mean annual inflow
    if nrows > 1:
        f_d,ax_d = figure_formatting('','',1,1,figsiz = (17,14))
        gs = GridSpec(nrows, ncols)
        ax_d = [0]*len(country_iso_alpha_2)
        counter = 0
        if len(country_iso_alpha_2) % 3 != 0:
            for row in range(nrows-1):
                for col in range(ncols):
                    ax_d[counter] = plt.subplot(gs[row,col])
                    counter += 1
            if len(country_iso_alpha_2) % 3 == 1:
                ax_d[counter] = plt.subplot(gs[row+1,0])
            elif len(country_iso_alpha_2) % 3 == 2:
                ax_d[counter] = plt.subplot(gs[row+1,0])
                ax_d[counter+1] = plt.subplot(gs[row+1,1])
        else:
            for row in range(nrows):
                for col in range(ncols):
                    ax_d[counter] = plt.subplot(gs[row,col]) 
                    counter += 1
    else:
        f_d,ax_d = figure_formatting('','',nrows,ncols,figsiz = (12,10))
    f_d.subplots_adjust(wspace=0.1)
    f_d.subplots_adjust(hspace=0.75)
    p = {}            
    for c in range(len(country_name)):
        # inflow_annual: [BOC/EOC][gcm][rcm][rcp][c][year] 
        if rcp == '26':
            inflow_tot_boc = np.concatenate((inflow_annual[0][0][0][rcp_it][c][:],inflow_annual[0][1][0][rcp_it][c][:],
                                             inflow_annual[0][3][0][rcp_it][c][:],
                                             inflow_annual[0][4][0][rcp_it][c][:],
                                             inflow_annual[0][1][1][rcp_it][c][:],
                                             inflow_annual[0][3][1][rcp_it][c][:]))
            inflow_tot_eoc = np.concatenate((inflow_annual[1][0][0][rcp_it][c][:],inflow_annual[1][1][0][rcp_it][c][:],
                                             inflow_annual[1][3][0][rcp_it][c][:],
                                             inflow_annual[1][4][0][rcp_it][c][:],
                                             inflow_annual[1][1][1][rcp_it][c][:],
                                             inflow_annual[1][3][1][rcp_it][c][:]))
        elif rcp == '45':
            inflow_tot_boc = np.concatenate((inflow_annual[0][0][0][rcp_it][c][:],inflow_annual[0][1][0][rcp_it][c][:],
                                             inflow_annual[0][2][0][rcp_it][c][:],inflow_annual[0][3][0][rcp_it][c][:],
                                             inflow_annual[0][4][0][rcp_it][c][:],
                                             inflow_annual[0][1][1][rcp_it][c][:],
                                             inflow_annual[0][3][1][rcp_it][c][:],
                                             inflow_annual[0][4][1][rcp_it][c][:]))
            inflow_tot_eoc = np.concatenate((inflow_annual[1][0][0][rcp_it][c][:],inflow_annual[1][1][0][rcp_it][c][:],
                                             inflow_annual[1][2][0][rcp_it][c][:],inflow_annual[1][3][0][rcp_it][c][:],
                                             inflow_annual[1][4][0][rcp_it][c][:],
                                             inflow_annual[1][1][1][rcp_it][c][:],
                                             inflow_annual[1][3][1][rcp_it][c][:],
                                             inflow_annual[1][4][1][rcp_it][c][:]))
        else:
            inflow_tot_boc = np.concatenate((inflow_annual[0][0][0][rcp_it][c][:],inflow_annual[0][1][0][rcp_it][c][:],
                                             inflow_annual[0][2][0][rcp_it][c][:],inflow_annual[0][3][0][rcp_it][c][:],
                                             inflow_annual[0][4][0][rcp_it][c][:],
                                             inflow_annual[0][0][1][rcp_it][c][:],inflow_annual[0][1][1][rcp_it][c][:],
                                             inflow_annual[0][2][1][rcp_it][c][:],inflow_annual[0][3][1][rcp_it][c][:],
                                             inflow_annual[0][4][1][rcp_it][c][:]))
            inflow_tot_eoc = np.concatenate((inflow_annual[1][0][0][rcp_it][c][:],inflow_annual[1][1][0][rcp_it][c][:],
                                             inflow_annual[1][2][0][rcp_it][c][:],inflow_annual[1][3][0][rcp_it][c][:],
                                             inflow_annual[1][4][0][rcp_it][c][:],
                                             inflow_annual[1][0][1][rcp_it][c][:],inflow_annual[1][1][1][rcp_it][c][:],
                                             inflow_annual[1][2][1][rcp_it][c][:],inflow_annual[1][3][1][rcp_it][c][:],
                                             inflow_annual[1][4][1][rcp_it][c][:]))
        if c == 0:
            dplot1 = sns.distplot(inflow_tot_boc, ax=ax_d[c],color='royalblue',label='BOC')
            dplot2 = sns.distplot(inflow_tot_eoc, ax=ax_d[c],color='orange',label='EOC')
        else:
            dplot1 = sns.distplot(inflow_tot_boc, ax=ax_d[c],color='royalblue')
            dplot2 = sns.distplot(inflow_tot_eoc, ax=ax_d[c],color='orange')
        kdeline1 = dplot1.lines[0]
        mean1 = inflow_tot_boc.mean()
        std1 = sigma_interannual_boc_mean[c] #np.std(inflow_tot_boc)
        median1 = np.median(inflow_tot_boc)
        height1 = np.interp(mean1, kdeline1.get_xdata(), kdeline1.get_ydata())
        dplot1.vlines(mean1, 0, height1, color='royalblue', ls='--')
        dplot1.set_ylim(ymin=0)
        kdeline2 = dplot1.lines[1]
        mean2 = inflow_tot_eoc.mean()
        std2 = sigma_interannual_eoc_mean[c] #np.std(inflow_tot_eoc)
        median2 = np.median(inflow_tot_eoc)
        height2 = np.interp(mean2, kdeline2.get_xdata(), kdeline2.get_ydata())
        dplot2.vlines(mean2, 0, height2, color='orange', ls='--')
        dplot2.set_ylim(ymin=0)
        # Null hypothesis H0: mu2 - mu1 = 0
        # Alternative hypothesis: mu2 - mu1 != 0
        #Check if BOC and EOC are normally distributed w. the Shapiro-Wilk test.
        #First value is W test value and second is the p-value (which should be larger
        #than 0.05 to meet the null hypothesis that the data is normal distributed)
        pn1 = sp.shapiro(inflow_tot_boc)
        pn2 = sp.shapiro(inflow_tot_eoc)
        p[c] = sp.ttest_rel(inflow_tot_boc,inflow_tot_eoc).pvalue
        rel_change_inflow = ((inflow_tot_eoc.mean() - inflow_tot_boc.mean())/inflow_tot_boc.mean()*100).round(1)
        rel_change_std = ((std2 - std1)/std1*100).round(1)
        if p[c] < 0.05:
            if rel_change_inflow < 0:
                ax_d[c].patch.set_facecolor(color=colors[6])
                ax_d[c].text(0.01, 0.4, '$S^E=$' + str(rel_change_inflow) + ' %', transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            elif rel_change_inflow > 0:
                ax_d[c].patch.set_facecolor(color=colors[0])
                ax_d[c].text(0.01, 0.4, '$S^E=+$' + str(rel_change_inflow) + ' %', transform=ax_d[c].transAxes,fontproperties=ticks_font_2)    
            if rel_change_std < 0:
                ax_d[c].text(0.01, 0.15, r'$S^\sigma=$' + str(rel_change_std) + ' %', transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            else:
                ax_d[c].text(0.01, 0.15, r'$S^\sigma=+$' + str(rel_change_std) + ' %', transform=ax_d[c].transAxes,fontproperties=ticks_font_2)    
        ax_d[c].patch.set_alpha(0.5)
        ax_d[c].set_xlabel('TWh',fontproperties=ticks_font_2,labelpad = 0)
        #ax_d[c].set_ylabel('PDF',fontproperties=ticks_font_2,labelpad = lp)
        ax_d[c].set_xticklabels(ax_d[c].get_xticks(), fontProperties = ticks_font_2)
        ax_d[c].set_yticklabels(ax_d[c].get_yticks(), fontProperties = ticks_font_2)
        ax_d[c].xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        ax_d[c].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        if WI == 1:
            ax_d[c].text(0.01, 0.7, country_iso_alpha_2[c], transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            if p[c] < 0.001:
                ax_d[c].text(0.76, 0.7, r'$p$' + ' < 0.001', transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            else:
                ax_d[c].text(0.76, 0.7, r'$p = $' + str(np.round(p[c],2)), transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            if pn1.pvalue < 0.001:
                ax_d[c].text(0.67, 0.45, r'$p_{n_{BOC}}$' + ' < 0.001', transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            elif pn1.pvalue < 0.1 and pn1.pvalue >= 0.001:
                ax_d[c].text(0.65, 0.45, r'$p_{n_{BOC}} = $' + '%s' % float('%.1g' % pn1.pvalue), transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            else:
                ax_d[c].text(0.69, 0.45, r'$p_{n_{BOC}} = $' + '%s' % float('%.2g' % pn1.pvalue), transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            if pn2.pvalue < 0.001:
                ax_d[c].text(0.67, 0.2, r'$p_{n_{EOC}}$' + ' < 0.001', transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            elif pn2.pvalue < 0.1 and pn2.pvalue >= 0.001:
                ax_d[c].text(0.65, 0.2, r'$p_{n_{EOC}} = $' + '%s' % float('%.1g' % pn2.pvalue), transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            else:
                ax_d[c].text(0.69, 0.2, r'$p_{n_{EOC}} = $' + '%s' % float('%.2g' % pn2.pvalue), transform=ax_d[c].transAxes,fontproperties=ticks_font_2)

            # ax_d[c].text(0.01, 0.7, country_iso_alpha_2[c], transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            # ax_d[c].text(0.68, 0.7, r'$p = $' + str(np.round(p[c],2)), transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            # ax_d[c].text(0.68, 0.45, r'$p_{n_{BOC}} = $' + str(np.round(pn1.pvalue,2)), transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            # ax_d[c].text(0.68, 0.2, r'$p_{n_{EOC}} = $' + str(np.round(pn2.pvalue,2)), transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
        else:
            ax_d[c].text(0.01, 0.85, country_iso_alpha_2[c], transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            ax_d[c].text(0.68, 0.85, r'$p = $' + str(np.round(p[c],2)), transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            ax_d[c].text(0.68, 0.5, r'$p_{n_{BOC}} = $' + str(np.round(pn1.pvalue,2)), transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            ax_d[c].text(0.68, 0.3, r'$p_{n_{EOC}} = $' + str(np.round(pn2.pvalue,2)), transform=ax_d[c].transAxes,fontproperties=ticks_font_2)
            
        ax_d[c].axes.get_yaxis().set_visible(False)
    f_d.legend(frameon=False,loc='lower center',bbox_to_anchor=(1.6, 0.05), bbox_transform=ax_d[c].transAxes,prop=ticks_font_3,ncol=2)
    f_d.savefig(figuredir + 't_test_significance_evaluation_GRAND_ENSEMBLE_MEAN_' + rcp + '.png',bbox_inches='tight') 
    #%% ENSEMBLE MEAN EUROPE MAP
    fig_EM = plt.figure(figsize=(14,10))
    gs = GridSpec(3, 3, figure=fig_EM)
    ax_EM = fig_EM.add_subplot(gs[0:3, 0:3])
    inflow_plot = np.zeros([len(country_iso_alpha_2)])
    EOC_rel_change = np.zeros([len(country_iso_alpha_2)])
    # initialise basemap   
    m = Basemap(width=12000000/2.2,height=9000000/2.2,projection='laea',
                resolution='i',lat_0=54.5,lon_0=12)
    
    m_plot = Basemap(width=12000000/2.2,height=9000000/2.2,projection='laea',
                     resolution='i',lat_0=54.5,lon_0=12,ax=ax_EM)
        
    m_plot.fillcontinents(color='lightgrey',zorder=1)
    m_plot.drawcountries(zorder=3,color=colors[18],linewidth=0.2)
    ax_plot_list = [[0.3,0.8,0.18,0.18],[0.015,0.575,0.18,0.18],[0.015,0.35,0.18,0.18],[0.015,0.8,0.18,0.18],
                    [0.8,0.8,0.18,0.18], [0.8,0.575,0.18,0.18], [0.8,0.35,0.18,0.18],
                    [0.8,0.1,0.18,0.18]]
    ax_plots = [0]*8
    c_plot = 0
    hist_woa_EM = pd.concat(hist_woa).groupby(level=1).mean() # Ensemble Mean BOC inflow in TWh
    cc_woa_EM = pd.concat(cc_woa).groupby(level=1).mean()# Ensemble Mean EOC inflow in TWh
    hist_EM = hist_woa_EM.drop(columns='day_of_year')
    hist_EM = hist_EM[~((hist_EM.index.month == 2) & (hist_EM.index.day == 29))]
    hist_EM['day'] = np.array(list(np.arange(1,366))*30)
    hist_EM = hist_EM.groupby('day').mean()
    hist_EM.index = pd.date_range('1/1/2015','12/31/2015',freq='d')
    cc_EM = cc_woa_EM.drop(columns='day_of_year')
    cc_EM = cc_EM[~((cc_EM.index.month == 2) & (cc_EM.index.day == 29))]
    cc_EM['day'] = np.array(list(np.arange(1,366))*30)
    cc_EM = cc_EM.groupby('day').mean()
    cc_EM.index = pd.date_range('1/1/2015','12/31/2015',freq='d')
    hist_EM.loc[pd.Timestamp('2016-01-01')] = 0
    cc_EM.loc[pd.Timestamp('2016-01-01')] = 0
    (hist_EM.resample('h').pad()[:-1]/24*1e6).round(1).to_csv(resdatadir + 'Hydro_inflow_BOC_ensemble_mean_' + 'rcp' + rcp + '.csv', sep=';', line_terminator='\n', float_format='%.1f')
    (cc_EM.resample('h').pad()[:-1]/24*1e6).round(1).to_csv(resdatadir + 'Hydro_inflow_EOC_ensemble_mean_' + 'rcp' + rcp + '.csv', sep=';', line_terminator='\n', float_format='%.1f')
    hist_EM = hist_EM[:-1]
    cc_EM = cc_EM[:-1]
    norm=plt.Normalize(-32,32)
    for c in range(len(country_iso_alpha_2)):
        # ========================================================================
        country = country_iso_alpha_2[c]
        BOC_results = pd.DataFrame()
        EOC_results = pd.DataFrame()
        BOC_results['index'] = hist_woa_EM[country].index
        EOC_results['index'] = cc_woa_EM[country].index
        BOC_results['inflow'] = hist_woa_EM[country].values
        EOC_results['inflow'] = cc_woa_EM[country].values
        BOC_results.set_index('index',inplace=True)
        EOC_results.set_index('index',inplace=True)
        BOC_results = BOC_results.groupby(pd.Grouper(freq='MS')).sum()
        EOC_results = EOC_results.groupby(pd.Grouper(freq='MS')).sum()
        BOC_results['month'] = BOC_results.index.month
        EOC_results['month'] = EOC_results.index.month
        BOC_results_mean = BOC_results.groupby('month').mean()
        EOC_results_mean = EOC_results.groupby('month').mean()
        BOC_results.drop(columns='month',inplace=True)
        mu0 = hist_woa_EM[country].groupby(pd.Grouper(freq='y')).sum().mean() # mean annual inflow BOC
        df_d_avg_boc = pd.DataFrame()
        df_d_avg_boc['inflow'] = hist_woa_EM[country]
        df_d_avg_boc = df_d_avg_boc[~((df_d_avg_boc.index.month == 2) & (df_d_avg_boc.index.day == 29))]
        df_d_avg_boc['day'] = np.array(list(np.arange(1,366))*30) #df_d_avg_boc.index.dayofyear
        df_d_avg_boc = df_d_avg_boc.groupby('day').mean()
        df_d_avg_boc.index = pd.date_range('1/1/2015','12/31/2015',freq='d')
        df_d_avg_eoc = pd.DataFrame()
        df_d_avg_eoc['inflow'] = cc_woa_EM[country]
        df_d_avg_eoc = df_d_avg_eoc[~((df_d_avg_eoc.index.month == 2) & (df_d_avg_eoc.index.day == 29))]
        df_d_avg_eoc['day'] = np.array(list(np.arange(1,366))*30) #df_d_avg_boc.index.dayofyear
        df_d_avg_eoc = df_d_avg_eoc.groupby('day').mean()
        df_d_avg_eoc.index = pd.date_range('1/1/2015','12/31/2015',freq='d')
        mu1 = cc_woa_EM[country].groupby(pd.Grouper(freq='y')).sum().mean() # mean annual inflow EOC
        # z-scores: https://useruploads.socratic.org/scL2sv6QVKr2i3fXhvSg_z196.jpg
        EOC_rel_change[c] = (mu1 - mu0)/mu0*100# change in mean annual inflow at EOC relative to BOC
        inflow_plot[c] = EOC_rel_change[c] # Relative change in percent
        inflow_hist_profile = BOC_results_mean # inflow_hist_mean # BOC seasonal profile
        inflow_eoc_profile = EOC_results_mean #EOC seasonal profile
        inflow_hist_profile.index = pd.date_range('1/1/2016','12/1/2016',freq='MS')
        inflow_eoc_profile.index = pd.date_range('1/1/2016','12/1/2016',freq='MS')
        #==========================================================================
        if country in country_seasonal_plot: 
            ax_plots[c_plot] = ax_EM.inset_axes(ax_plot_list[c_plot])
            if c_plot == 0:
                lpl1, = ax_plots[c_plot].plot(df_d_avg_boc*1e3,color=colors[10],label='1991 - 2020',alpha=0.75,lw=0,zorder=4)
                ax_plots[c_plot].fill_between(df_d_avg_boc.index,df_d_avg_boc.iloc[:,0]*1e3,color=colors[10],alpha=0.75,lw=0,zorder=4)
                lpl2, = ax_plots[c_plot].plot(df_d_avg_eoc*1e3,color=rcp_dic[rcp],label='2071 - 2100 RCP' + str(int(rcp)/10),alpha=0.5,lw=0,zorder=5)
                ax_plots[c_plot].fill_between(df_d_avg_eoc.index,df_d_avg_eoc.iloc[:,0]*1e3,color=rcp_dic[rcp],alpha=0.5,lw=0,zorder=5)
            else:
                ax_plots[c_plot].plot(df_d_avg_boc*1e3,color=colors[10],alpha=0.75,lw=0,zorder=4)
                ax_plots[c_plot].plot(df_d_avg_eoc*1e3,color=rcp_dic[rcp],alpha=0.5,lw=0,zorder=5)
                ax_plots[c_plot].fill_between(df_d_avg_boc.index,df_d_avg_boc.iloc[:,0]*1e3,color=colors[10],alpha=0.75,lw=0,zorder=4)
                ax_plots[c_plot].fill_between(df_d_avg_eoc.index,df_d_avg_eoc.iloc[:,0]*1e3,color=rcp_dic[rcp],alpha=0.5,lw=0,zorder=5)
            if EOC_rel_change[c] < 0:
                ax_plots[c_plot].text(0.05,0.85,country + ' (-' + str(np.round(np.abs(EOC_rel_change[c]),1)) + '%)',fontproperties=ticks_font,transform=ax_plots[c_plot].transAxes)
            else:
                ax_plots[c_plot].text(0.05,0.85,country + ' (+' + str(np.round(np.abs(EOC_rel_change[c]),1)) + '%)',fontproperties=ticks_font,transform=ax_plots[c_plot].transAxes)
            ax_plots[c_plot].set_ylabel('GWh',fontproperties=ticks_font, labelpad = -3)
            if c == 0 or c == 1 or c == 2 or c == 3:
                ax_plots[c_plot].yaxis.tick_right()
                ax_plots[c_plot].yaxis.set_label_coords(1.4,0.5)
            ax_plots[c_plot].set_yticklabels(ax_plots[c_plot].get_yticks(), fontProperties = ticks_font)
            ax_plots[c_plot].set_xticklabels(ax_plots[c_plot].get_xticks(), fontProperties = ticks_font)
            ax_plots[c_plot].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
            ax_plots[c_plot].set_xticklabels(inflow_hist_profile.index)
            ax_plots[c_plot].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax_plots[c_plot].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax_plots[c_plot].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
            ax_plots[c_plot].grid(linestyle=':', linewidth='0.5', color='grey',which='both',axis='x')
            ax_plots[c_plot].set_xlim([datetime(2015,1,1),datetime(2015,12,1)])
            ax_plots[c_plot].set_ylim([0,ax_plots[c_plot].get_ylim()[1]*1.2])
            c_plot += 1
        # =========================================================================
        # color countries corresponding to relative change in mean annual inflow
        # Shape files: https://www.gadm.org/download_country_v3.html
        m.readshapefile('shapefiles/gadm36_' + country_iso_alpha_3[c] + '_0',country_name[c],drawbounds=False)
        patches = []
        for info, shape in zip(eval('m.' + country_name[c] + '_info'), eval('m.' + country_name[c])):
            patches.append(Polygon(np.array(shape), True))
        if p[c] < 0.05:
            patch1=ax_EM.add_collection(PatchCollection(patches, facecolor= cmap(norm(inflow_plot[c])), linewidths=0, hatch='///', zorder=2))
        else:
            patch1=ax_EM.add_collection(PatchCollection(patches, facecolor= cmap(norm(inflow_plot[c])), linewidths=0, zorder=2))    
        CC_plot = country_coord[country_coord.Code == country]
        ax_EM.annotate(CC_plot.Code.item(),xy=m(np.array(CC_plot.Lon.item()), np.array(CC_plot.Lat.item())),color='w',fontproperties = country_code_font, zorder=13)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(inflow_plot)
    divider = make_axes_locatable(ax_EM)
    cax = divider.new_vertical(size="5%", pad=0.5, pack_start=True)
    fig_EM.add_axes(cax)
    cbar_EM = fig_EM.colorbar(sm, cax=cax, orientation="horizontal")
    cbar_EM.set_label('Change in annual inflow [%]',fontproperties = ticks_font)
    ax_EM.plot([0.025,0.06],[0.08,0.08],transform=ax_EM.transAxes,zorder=10,color=colors[10],ls='-',lw=10,alpha=0.75)
    ax_EM.plot([0.025,0.06],[0.04,0.04],transform=ax_EM.transAxes,zorder=10,color=rcp_dic[rcp],ls='-',lw=10,alpha=0.5)
    fpl1 = ax_EM.fill_between(x=[-0.022,-0.048],y1=[0.10,0.10],y2=[0.09,0.09],transform=ax_EM.transAxes,facecolor = 'k',alpha=0.1, zorder=10,hatch = '///',label = 'p < 0.05')
    ax_EM.legend(handles=[fpl1, lpl1, lpl2],frameon=False,loc='lower left',prop=ticks_font)
    fig_EM.savefig(figuredir + 'Hydro_reservoir_energy_inflow_map_ensemble_mean_' + rcp + '.png',bbox_inches='tight') 
    #%% Extreme events
    scen = 'drought'
    mean_dur_boc,mean_nb_seq_boc,mean_dur_eoc,mean_nb_seq_eoc = extreme_events_count(country_iso_alpha_2,scen,hist_woa_EM,cc_woa_EM)    
    f4,ax4 = figure_formatting('Mean duration [days]','Mean number of sequences [/year]',1,1,color='k',figsiz = (15,9))
    f4 = extreme_events_plot(country_iso_alpha_2,country_name,scen,f4,ax4,mean_dur_boc,mean_dur_eoc,mean_nb_seq_boc,mean_nb_seq_eoc,rcp_dic,rcp)
    ax4[0].grid(linestyle='-', linewidth='0.75', color='darkgrey',which='both',axis='both')   
    plt.margins(0,0)
    ax4[0].set_xticklabels([2, 5, 10, 20])
    ax4[0].set_yticklabels([0.5, 1, 2, 5, 10, 20])
    ax4[0].set_xticks([2, 5, 10, 20])
    ax4[0].set_yticks([0.5, 1, 2, 5, 10, 20])
    f4.savefig(figuredir + 'Extreme_events_droughts_' + rcp + '.png',bbox_inches='tight') 
    scen = 'overflow'
    mean_dur_boc,mean_nb_seq_boc,mean_dur_eoc,mean_nb_seq_eoc = extreme_events_count(country_iso_alpha_2,scen,hist_woa_EM,cc_woa_EM)    
    f5,ax5 = figure_formatting('Mean duration [days]','Mean number of sequences [/year]',1,1,color='k',figsiz = (15,9))
    f5 = extreme_events_plot(country_iso_alpha_2,country_name,scen,f5,ax5,mean_dur_boc,mean_dur_eoc,mean_nb_seq_boc,mean_nb_seq_eoc,rcp_dic,rcp)
    ax5[0].grid(linestyle='-', linewidth='0.75', color='darkgrey',which='both',axis='both')  
    ax5[0].set_xticklabels([2, 5, 10, 20])
    ax5[0].set_yticklabels([0.5, 1, 2, 5, 10, 20])
    ax5[0].set_xticks([2, 5, 10, 20])
    ax5[0].set_yticks([0.5, 1, 2, 5, 10, 20])
    f5.savefig(figuredir + 'Extreme_events_overflow_' + rcp + '.png',bbox_inches='tight')         
    
    
    #%%
    rcp_it += 1
#%% RCP comparison
def box_plot(data, edge_color, fill_color,ax,x,wi):
    bp = ax.boxplot(data, patch_artist=True, positions = x, widths=wi,showfliers=False)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)    
        
    return bp

if len(rcp_list) == 3:
    country_sort_df = pd.DataFrame()
    country_sort_df['country'] = country_iso_alpha_2
    country_sort_df['delta_E'] = EOC_rel_change
    country_sort_df.sort_values(by='delta_E',inplace=True)
    BOC_RCP26 = [0]*len(country_name)
    BOC_RCP45 = [0]*len(country_name)
    BOC_RCP85 = [0]*len(country_name)
    EOC_RCP26 = [0]*len(country_name)
    EOC_RCP45 = [0]*len(country_name)
    EOC_RCP85 = [0]*len(country_name)
    Delta_E_RCP26 = [0]*len(country_name)
    Delta_E_RCP45 = [0]*len(country_name)
    Delta_E_RCP85 = [0]*len(country_name)
    c_it = 0
    for c in country_sort_df.index: #range(len(country_name)):
        BOC_RCP26[c_it] = np.concatenate([inflow_annual[0][0][0][0][c][:],inflow_annual[0][1][0][0][c][:],
                                      inflow_annual[0][3][0][0][c][:],inflow_annual[0][4][0][0][c][:],
                                      inflow_annual[0][1][1][0][c][:],
                                      inflow_annual[0][3][1][0][c][:]])
                                  
        BOC_RCP45[c_it] = np.concatenate([inflow_annual[0][0][0][1][c][:],inflow_annual[0][1][0][1][c][:],inflow_annual[0][2][0][1][c][:],
                                      inflow_annual[0][3][0][1][c][:],inflow_annual[0][4][0][1][c][:],
                                      inflow_annual[0][1][1][1][c][:],
                                      inflow_annual[0][3][1][1][c][:],inflow_annual[0][4][1][1][c][:]])
                                      
        BOC_RCP85[c_it] = np.concatenate([inflow_annual[0][0][0][2][c][:],inflow_annual[0][1][0][2][c][:],inflow_annual[0][2][0][2][c][:],
                                      inflow_annual[0][3][0][2][c][:],inflow_annual[0][4][0][2][c][:],
                                      inflow_annual[0][0][1][2][c][:],inflow_annual[0][1][1][2][c][:],inflow_annual[0][2][1][2][c][:],
                                      inflow_annual[0][3][1][2][c][:],inflow_annual[0][4][1][2][c][:]])
        
        EOC_RCP26[c_it] = np.concatenate([inflow_annual[1][0][0][0][c][:],inflow_annual[1][1][0][0][c][:],
                                      inflow_annual[1][3][0][0][c][:],inflow_annual[1][4][0][0][c][:],
                                      inflow_annual[1][1][1][0][c][:],
                                      inflow_annual[1][3][1][0][c][:]])
                                      
        EOC_RCP45[c_it] = np.concatenate([inflow_annual[1][0][0][1][c][:],inflow_annual[1][1][0][1][c][:],inflow_annual[1][2][0][1][c][:],
                                      inflow_annual[1][3][0][1][c][:],inflow_annual[1][4][0][1][c][:],
                                      inflow_annual[1][1][1][1][c][:],
                                      inflow_annual[1][3][1][1][c][:],inflow_annual[1][4][1][1][c][:]])
                                      
        EOC_RCP85[c_it] = np.concatenate([inflow_annual[1][0][0][2][c][:],inflow_annual[1][1][0][2][c][:],inflow_annual[1][2][0][2][c][:],
                                      inflow_annual[1][3][0][2][c][:],inflow_annual[1][4][0][2][c][:],
                                      inflow_annual[1][0][1][2][c][:],inflow_annual[1][1][1][2][c][:],inflow_annual[1][2][1][2][c][:],
                                      inflow_annual[1][3][1][2][c][:],inflow_annual[1][4][1][2][c][:]])
    
        Delta_E_RCP26[c_it] = (EOC_RCP26[c_it] - BOC_RCP26[c_it].mean())/BOC_RCP26[c_it].mean()*100
        Delta_E_RCP45[c_it] = (EOC_RCP45[c_it] - BOC_RCP45[c_it].mean())/BOC_RCP45[c_it].mean()*100
        Delta_E_RCP85[c_it] = (EOC_RCP85[c_it] - BOC_RCP85[c_it].mean())/BOC_RCP85[c_it].mean()*100
        c_it += 1
        
    f_rcp,ax_rcp = figure_formatting('','Change in annual inflow [%]',1,1,color='k',figsiz = (14,6))
    x = np.arange(len(country_name))
    wi = 0.2
    bp1 = box_plot(Delta_E_RCP26,'k',rcp_dic['26'],ax_rcp[0],x,wi)
    bp2 = box_plot(Delta_E_RCP45,'k',rcp_dic['45'],ax_rcp[0],x+wi,wi)
    bp3 = box_plot(Delta_E_RCP85,'k',rcp_dic['85'],ax_rcp[0],x+2*wi,wi)
    
    ax_rcp[0].hlines(0,xmin=ax_rcp[0].get_xlim()[0],xmax=ax_rcp[0].get_xlim()[1],ls='--',lw=1)
    ax_rcp[0].set_xticks(x+1.5*wi)
    ax_rcp[0].set_xticklabels(country_sort_df.country.values,fontproperties=fm.FontProperties(fname='AUfonts/AUPassata_Rg.ttf',size=23))
    
    ax_rcp[0].set_ylim([-105,105])
    f_rcp.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ['RCP2.6', 'RCP4.5','RCP8.5'], loc='lower center',prop=ticks_font_3,ncol=3,frameon=False,borderaxespad=-0.25)
    f_rcp.savefig(figuredir + 'RCP_comparison.png',bbox_inches='tight') 
