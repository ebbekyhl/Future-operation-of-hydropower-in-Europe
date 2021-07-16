# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:30:03 2021
@author: ebbek
The function "inflforecast" first trains the modelled inflow at BOC with
historical data, and subsequently obtains a month- and country-dependent 
retain factor. The modelled inflow at EOC is then calibrated with the retain
factor. Outliers in the training data are imputed. 
Input:
    histdatadir: Directory in which historical inflow data is located
    gendatadir: Directory in which general data is located
    moddatadir: Directory in which raw modelled inflow data is located
    resdatadir: Directory in which calibrated modelled inflow data is located
    gcm: Name of general circulation model
    rcm: Name of regional climate model
    rcp: Representative concentration pathway
    countries: A list of countries (full name)
    country_codes: A list of countries (code)
Output:
    df_master_frame_hist_woa: One .csv file containing BOC inflow for all countries
    df_master_frame_eoc_woa: One .csv file containing EOC inflow for all countries
"""

def inflforecast(histdatadir,gendatadir,moddatadir,resdatadir,gcm,rcm,rcp,countries,country_codes):
    import pandas as pd
    import numpy as np
    from inflhist import inflhist
    import scipy.stats as sp
    from infl_concat import infl_concat
    rf_df_av = [0]*len(countries)
    df_master_frame_hist_woa = pd.DataFrame()
    df_master_frame_eoc_woa = pd.DataFrame()
    df_master_frame_ac = pd.DataFrame()
    df_master_frame_ac['country'] = [0]*len(countries)
    df_master_frame_ac['Hist_annual_inflow'] = [0]*len(countries)
    df_master_frame_ac['EOC_annual_inflow'] = [0]*len(countries)
    df_master_frame_ac['Percentage_change'] = [0]*len(countries)
    for c in range(len(countries)):
        hist_years = [1991,2020]
        country = country_codes[c]
        country_l = countries[c]
        
        infl_cal,infl_EOC = infl_concat(country,gendatadir,moddatadir,gcm,rcm,rcp)
        infl_cal = (infl_cal*(3.6e12)**-1) # Unit conversion from Joule to GWh 
        
        time_dt = pd.to_datetime(infl_cal.index)
        infl_cal['date'] = time_dt
        infl_cal.set_index('date',inplace=True)
        
        infl_cal_d = infl_cal.resample('d').sum()
        
        hist = inflhist(histdatadir,hist_years[0],hist_years[1],country,country_l)
        hist = hist[(hist.T != 0).any()] # Removing rows with zeros
        hist_m_train = hist #.drop(columns='month')
        
        infl_cal_monthly = infl_cal.groupby(pd.Grouper(freq='MS')).sum()
        infl_cal_m_train = infl_cal_monthly.loc[hist.index]

        if country == 'CH': # Remove outlier from historical data
            dr_index = hist_m_train[hist_m_train.index.year == 2018][hist_m_train[hist_m_train.index.year == 2018].index.month == 10].index
            hist_m_train = hist_m_train.drop(index=dr_index)
            infl_cal_m_train = infl_cal_m_train.drop(index=dr_index)
            
        if len(hist_m_train[(np.abs(sp.zscore(hist_m_train)) > 3)]) > 0: # Remove outliers (Outliers leads to a bad training)
            outlier_arr = (np.abs(sp.zscore(hist_m_train)) > 3)    
            hist_m_train[outlier_arr] = np.nan
            infl_cal_m_train[outlier_arr] = np.nan
            infl_cal_m_train = infl_cal_m_train.dropna()
            hist_m_train = hist_m_train.dropna()
    
        if len(infl_cal_m_train[(np.abs(sp.zscore(infl_cal_m_train)) > 3)]) > 0:
            outlier_arr = (np.abs(sp.zscore(infl_cal_m_train)) > 3)
            infl_cal_m_train[outlier_arr] = np.nan
            hist_m_train[outlier_arr] = np.nan
            infl_cal_m_train = infl_cal_m_train.dropna()
            hist_m_train = hist_m_train.dropna()
        
        # RETAIN FACTOR CALCULATED BASED ON MEAN SEASONAL INFLOW
        infl_cal_monthly['month'] = infl_cal_monthly.index.month
        hist_m_train['month'] = hist_m_train.index.month
        hist_m_train_seasonal = hist_m_train.groupby('month').mean()
        infl_cal_m_train_seasonal = infl_cal_monthly.groupby('month').mean()
        rf_df_train_avg1 = hist_m_train_seasonal/infl_cal_m_train_seasonal
        rf_df_train_avg1['date'] = pd.date_range('2016/01/01','2016/12/31',freq='MS')
        rf_df_train_avg1.set_index('date',inplace=True)
        hist_m_train.drop(columns='month',inplace=True)
        
        # RETAIN FACTOR CALCULATED BASED ON MONTHLY INFLOW 
        rf_df_train = hist_m_train/infl_cal_m_train
        rf_df_train.index = infl_cal_m_train.index
        rf_df_train = rf_df_train.dropna()
        rf_df_train_avg2 = pd.DataFrame()
        rf_df_train['month'] = rf_df_train.index.month
        rf_df_train_avg2['date'] = pd.date_range('2016/01/01','2016/12/31',freq='MS')
        rf_df_train_avg2['rf'] = rf_df_train.groupby('month').median().values 
        rf_df_train_avg2.set_index('date',inplace=True)
        rf_df_train.drop(columns='month',inplace=True)
        
        # Array containing 12 monthly average retain factors
        # rf_df_train_avg2.iloc[:] = rf_df_train_avg2.iloc[:].mean().item() # constant RF
        rf_df_av[c] = rf_df_train_avg1 # Retain factor based on seasonal inflow 
            
        infl_cal_d['rf'] = [0]*len(infl_cal_d)
        infl_cal['rf'] = [0]*len(infl_cal)
        for i in range(12):
            infl_cal_d.rf[infl_cal_d.index.month == i+1] = rf_df_av[c].iloc[i].item()
            infl_cal.rf[infl_cal.index.month == i+1] = rf_df_av[c].iloc[i].item()
        
        infl_cal_cor_d = infl_cal_d['inflow']*infl_cal_d['rf']
        infl_EOC = infl_EOC*((3.6e12)**-1) # convert Joule to GWh
        
        time_dt = pd.to_datetime(infl_EOC.index)
        infl_EOC['date'] = time_dt
        infl_EOC.set_index('date',inplace=True)
        infl_EOC['month'] = infl_EOC.index.month
        
        infl_EOC_h = infl_EOC.copy()
        infl_EOC_h['rf'] = [0]*len(infl_EOC_h)
        for i in range(12):
            infl_EOC_h.rf[infl_EOC_h.index.month == i+1] = rf_df_av[c].iloc[i].item()
        infl_EOC_cor_h = infl_EOC_h['inflow']*infl_EOC_h['rf']
        infl_EOC_cor_d = infl_EOC_cor_h.groupby(pd.Grouper(freq='d')).sum()
        df_master_frame_hist_woa[country] = infl_cal_cor_d
        df_master_frame_eoc_woa[country] = infl_EOC_cor_d

    df_master_frame_hist_woa.loc[pd.Timestamp('2021-01-01')] = 0
    df_master_frame_eoc_woa.loc[pd.Timestamp('2101-01-01')] = 0
    
    (df_master_frame_hist_woa.resample('h').pad()[:-1]/24*1000).round(1).to_csv(resdatadir + 'Hydro_inflow_BOC_' + gcm + '_' + rcm + '_rcp' + rcp + '.csv', sep=';', line_terminator='\n', float_format='%.1f')
    (df_master_frame_eoc_woa.resample('h').pad()[:-1]/24*1000).round(1).to_csv(resdatadir + 'Hydro_inflow_EOC_' + gcm + '_' + rcm + '_rcp' + rcp + '.csv', sep=';', line_terminator='\n', float_format='%.1f')
    
    df_master_frame_hist_woa = df_master_frame_hist_woa[:-1]
    df_master_frame_eoc_woa = df_master_frame_eoc_woa[:-1]
    
    return df_master_frame_hist_woa, df_master_frame_eoc_woa