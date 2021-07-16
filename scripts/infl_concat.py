# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:15:08 2021
@author: ebbek
The function "infl_concat" loads the raw modelled inflow time series from 
obtained with the ATLITE runoff conversion scheme. Each time series is of '
5 years duration, and the script concatenates the time series to the two 
eras BOC and EOC.
Input:
    country: Country 
    md_folder: Directory in which raw modelled data is located
    gd_folder: Directory in which general data is located
    gcm: General circulation model
    rcm: Regional climate model
    rcp: Representative concentration pathway
Output:
    BOC: Inflow time series at the beginning of century (2006 - 2020)
    EOC: Inflow time series at the end of century (2086 - 2100)
"""
def infl_concat(country,gd_folder,md_folder,gcm,rcm,rcp):
    import pandas as pd
    indices = pd.read_csv(gd_folder + 'index.csv',sep=';',index_col=[0,1,5])  
    init = indices.loc[gcm].loc[rcm].loc[int(rcp)].init      
    cordex_1 = pd.read_csv(md_folder + 'modelled_inflow_' + country + '_Cordex-' + gcm + '-' + rcm + '-' + 'historical-r1i1p1-1991-1995_weighted.csv',compression = 'zip',sep=';',index_col=0)
    cordex_2 = pd.read_csv(md_folder + 'modelled_inflow_' + country + '_Cordex-' + gcm + '-' + rcm + '-' + 'historical-r1i1p1-1996-2000_weighted.csv',compression = 'zip',sep=';',index_col=0)
    cordex_3 = pd.read_csv(md_folder + 'modelled_inflow_' + country + '_Cordex-' + gcm + '-' + rcm + '-' + 'historical-r1i1p1-2001-2005_weighted.csv',compression = 'zip',sep=';',index_col=0)
    cordex_4 = pd.read_csv(md_folder + 'modelled_inflow_' + country + '_Cordex-' + gcm + '-' + rcm + '-' + 'rcp' + rcp + '-' + init + '-2006-2010_weighted.csv',compression = 'zip',sep=';',index_col=0)
    cordex_5 = pd.read_csv(md_folder + 'modelled_inflow_' + country + '_Cordex-' + gcm + '-' + rcm + '-' + 'rcp' + rcp + '-' + init + '-2011-2015_weighted.csv',compression = 'zip',sep=';',index_col=0)
    cordex_6 = pd.read_csv(md_folder + 'modelled_inflow_' + country + '_Cordex-' + gcm + '-' + rcm + '-' + 'rcp' + rcp + '-' + init + '-2016-2020_weighted.csv',compression = 'zip',sep=';',index_col=0)
    cordex_7 = pd.read_csv(md_folder + 'modelled_inflow_' + country + '_Cordex-' + gcm + '-' + rcm + '-' + 'rcp' + rcp + '-' + init + '-2071-2075_weighted.csv',compression = 'zip',sep=';',index_col=0)
    cordex_8 = pd.read_csv(md_folder + 'modelled_inflow_' + country + '_Cordex-' + gcm + '-' + rcm + '-' + 'rcp' + rcp + '-' + init + '-2076-2080_weighted.csv',compression = 'zip',sep=';',index_col=0)
    cordex_9 = pd.read_csv(md_folder + 'modelled_inflow_' + country + '_Cordex-' + gcm + '-' + rcm + '-' + 'rcp' + rcp + '-' + init + '-2081-2085_weighted.csv',compression = 'zip',sep=';',index_col=0)                          
    cordex_10 = pd.read_csv(md_folder + 'modelled_inflow_' + country + '_Cordex-' + gcm + '-' + rcm + '-' + 'rcp' + rcp + '-' + init + '-2086-2090_weighted.csv',compression = 'zip',sep=';',index_col=0)
    cordex_11 = pd.read_csv(md_folder + 'modelled_inflow_' + country + '_Cordex-' + gcm + '-' + rcm + '-' + 'rcp' + rcp + '-' + init + '-2091-2095_weighted.csv',compression = 'zip',sep=';',index_col=0)
    cordex_12 = pd.read_csv(md_folder + 'modelled_inflow_' + country + '_Cordex-' + gcm + '-' + rcm + '-' + 'rcp' + rcp + '-' + init + '-2096-2100_weighted.csv',compression = 'zip',sep=';',index_col=0)                          
    BOC = pd.concat([cordex_1,cordex_2,cordex_3,cordex_4,cordex_5,cordex_6])
    EOC = pd.concat([cordex_7, cordex_8, cordex_9,cordex_10, cordex_11, cordex_12])
    return BOC,EOC

