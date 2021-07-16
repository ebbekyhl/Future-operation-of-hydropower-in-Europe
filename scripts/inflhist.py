# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:52:43 2021
@author: ebbek
The function "inflhist" loads the historical inflow data acquired from various
sources. 
Input:
    histdatadir: Directory in which historical inflow data is located
    s_year: Starting year of which historical inflow should be loaded from
    e_year: Ending year of which historical inflow should be loaded to
    country: Country code
    country_l: Country full name
Output:
    hist_month: Monthly historical inflow
"""

def inflhist(histdatadir,s_year,e_year,country,country_l):
    import pandas as pd
    from datetime import datetime
    import numpy as np
    if country == 'ES':        
        hist = pd.read_csv(histdatadir + 'Spain_1991_2019_monthly_timeseries.csv',index_col=0,names=['inflow']) 
        hist['date'] = pd.to_datetime(hist.index)
        hist.set_index('date',inplace=True)
        start_date = datetime(s_year,1,1)
        end_date = datetime(e_year+1,1,1)
        mask = (hist.index >= start_date) & (hist.index < end_date)
        hist_month = hist.loc[mask]   
        hist_month['inflow'] = hist_month.inflow.astype(float)
    elif country == 'NO':
        hist=pd.read_csv(histdatadir + 'energitilsig_Norge_1958_2017.csv',
                         engine='python',sep=';',skiprows=(1),
                         header=None, names=['date','inflow'])
        hist['date'] = pd.to_datetime(hist['date'])
        hist.set_index('date',inplace=True)
        hist_entso = pd.read_csv(histdatadir + 'Inflow_Norway_' + str(2016) + '-' + str(2019) + '.csv',names=['inflow'],skiprows=1)
        hist_entso['date'] = pd.to_datetime(hist_entso.index)
        hist_entso.set_index('date',inplace=True)
        hist_entso = hist_entso[hist_entso.index.year >= 2018]
        hist_entso_m = hist_entso.groupby(pd.Grouper(freq='MS')).sum()
        hist_m = hist.groupby(pd.Grouper(freq='MS')).sum()
        hist = pd.concat([hist_m,hist_entso_m])
        start_date = datetime(s_year,1,1)
        end_date = datetime(e_year+1,1,1)
        mask = (hist.index >= start_date) & (hist.index < end_date)
        hist_month = hist.loc[mask]   
    elif country == 'SE':
        hist_df =pd.read_csv(histdatadir + 'Inflow_SE_1980-2019.csv',
                 engine='python',sep=';')
        cnames_np = np.arange(1980,2020)
        cnames = cnames_np.astype(str)
        hist_df.columns = cnames
        hist = hist_df.stack().reset_index()
        hist.columns = ['week_no','year','inflow']
        #hist['day'] = len(hist)*['Tuesday']
        hist['week_no'] = hist['week_no'] + 1
        hist['year'] = hist['year'].astype(int)
        hist = hist.sort_values(by=['year','week_no']).reset_index(drop=True)
        hist['combined'] = hist['year'].astype(str)+'-' + 'W' + hist['week_no'].astype(str).astype(str) + '-2'
        hist = hist.drop(columns=['week_no','year'])
        r =[0]*len(hist.index)
        for i in range(len(hist.index)):
            r[i] = datetime.strptime(hist.combined.loc[i], "%Y-W%W-%w")
        hist['combined'] = r
        hist.set_index('combined',inplace=True)
        start_date = datetime(s_year,1,1)
        end_date = datetime(e_year+1,1,1)
        mask = (hist.index >= start_date) & (hist.index <= end_date)
        hist = hist.loc[mask]
    elif country == 'FI' or country == 'HR' or country == 'DE' or country == 'CZ' or country == 'HU' or country == 'PL' or country == 'SK' or country == 'BA' or country == 'MK' or country == 'SI':
        inflow = pd.read_csv(histdatadir + 'wattsight_inflow_' + country + '.csv')
        ind = inflow.date.astype(str).str.split(' ',n=1,expand=True)
        time_dt = pd.to_datetime(ind[0])
        inflow['date'] = time_dt
        inflow.set_index('date',inplace=True)
        start_date = datetime(s_year,1,1)
        end_date = datetime(e_year+1,1,1)
        mask = (inflow.index >= start_date) & (inflow.index <= end_date)
        hist = inflow.loc[mask]  
    else:
        if country == 'RS':
            hist = pd.read_csv(histdatadir + 'Inflow_' + country_l + '_' + str(2017) + '-' + str(2019) + '.csv',names=['inflow'],skiprows=1)
        else:
            hist = pd.read_csv(histdatadir + 'Inflow_' + country_l + '_' + str(2016) + '-' + str(2019) + '.csv',names=['inflow'],skiprows=1)
        idx_dt = [datetime.strptime(index, '%Y-%m-%d') for index in hist.index]     
        hist['index_dt']=idx_dt
        hist.set_index('index_dt', inplace=True) 
        hist = hist[hist > 0]
        start_date = datetime(s_year,1,1)
        end_date = datetime(e_year+1,1,1)
        mask = (hist.index >= start_date) & (hist.index <= end_date)
        hist = hist.loc[mask]
    if country != 'NO' and country != 'ES':
        hist_month = hist.groupby(pd.Grouper(freq='MS')).sum()
    return hist_month