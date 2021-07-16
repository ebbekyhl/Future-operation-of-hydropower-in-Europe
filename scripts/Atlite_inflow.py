# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 16:21:36 2020

@author: ebbek
"""
def hydropowerplants(hydrotype,hydropower_database,datapath,countries):
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import Point
    
    # Read hydropower plant database
    h = pd.read_csv(hydropower_database,index_col = [5])
    h['country_code'] = h.index
    h['index'] = np.arange(len(h))  
    h = h.loc[countries]
    h.set_index('index',inplace=True)   
    h['Coordinates'] = list(zip(h.lon, h.lat))
    h['Coordinates'] = h['Coordinates'].apply(Point)
    
    # Remove hydropower plants which causes error in the ATLITE conversion
    arrays_drop = [0]*len(countries)
    for c in range(len(countries)):
        ad = pd.read_csv(datapath + 'arr_drop_' + countries[c] + '_all.csv')
        ad_w = np.array(ad['0'])
        if countries[c] == 'ES':
            ad2 = np.array(pd.read_csv(datapath + 'Spain_debug_array.csv'))
            ad2 = ad2[ad2!=0]
            ad_w = np.concatenate([ad_w,ad2])
        elif countries[c] == 'PT':
            ad2 = np.array(pd.read_csv(datapath + 'Portugal_debug_array.csv'))
            ad2 = ad2[ad2!=0]
            ad_w = np.concatenate([ad_w,ad2])

        arrays_drop[c] = ad_w
    array_drop_big = np.sort(np.concatenate(arrays_drop).astype(int))
    h_big = h.drop(array_drop_big)
    
    # Remove hydropower plants which are not located within the country borders
    latlonrange = pd.read_csv(datapath + 'lat_lon_rang.csv',index_col = 0,sep=';') # Latitude and longitude country boundaries 
    h_big_drop = [0]*len(countries)
    for c in range(len(countries)):
        country = countries[c]
        h_big_c = h_big[h_big['country_code'] == country]
        h_big_drop[c] = h_big_c.drop(h_big_c[h_big_c.lat < latlonrange.loc[country].maxlatitude][h_big_c.lat > latlonrange.loc[country].minlatitude][h_big_c.lon < latlonrange.loc[country].maxlongitude][h_big_c.lon > latlonrange.loc[country].minlongitude].index).index
    h_big_drop_indices = np.concatenate(h_big_drop) # Outside of country border
    array_drop_big_2 = np.sort(np.concatenate([array_drop_big,h_big_drop_indices]))
    h_big = h.drop(array_drop_big_2)
    if hydrotype == 'all':
        hplants = gpd.GeoDataFrame(h_big, geometry='Coordinates')
    else:
        hplants = gpd.GeoDataFrame(h_big[h_big.type == hydrotype], geometry='Coordinates')

    return hplants

from Atlite_cutout import Atlite_cutout
import pandas as pd
import numpy as np
import atlite
from shapely.geometry import Point
#%% ============================= INPUT =======================================
# Cutout years:
cy_list = [[2016,2020]] 
  
# Driving model:
dr_model_list = ['MPI-M-MPI-ESM-LR']

# Downscaling:
rcm_list = ['RCA4']

# Type of hydropower plants included:
hydrotype = 'HDAM' # all/HDAM/HPHS/HROR

# Cutout directory:
cutout_dir = "cutouts"

# General data path:
gendatapath = 'gendata/'

# Results data path:
resdatapath = 'resdata/'

# Countries:
country_codes = ['SE','ES','NO','AT','BG','FI','HR','ME','PT','RO','CH','FR','IT','DE','CZ','HU','PL','SK','BA','MK','RS','SI']
#%% ============================= OUTPUT ======================================
# Create cutout
Atlite_cutout(cy_list,dr_model_list,rcm_list,cutout_dir,gendatapath)

hydropower_database = gendatapath + 'hydro-power-database-master/data/jrc-hydro-power-plant-database.csv'
hydrobasin_database = gendatapath + 'hydroBASINS/hybas_eu_lev08_v1c.shp'
indices = pd.read_csv(gendatapath + 'index.csv',sep=';',index_col=[0,1,5])
#array_drop_big = pd.read_csv(datapath + 'arr_drop_big.csv').values # Array containing indices of all hydro power plant which is inapplicable in the ATLITE conversion due to geometric conditions 
hplants = hydropowerplants(hydrotype,hydropower_database,gendatapath,country_codes)

for dr_model in dr_model_list:
    for rcm in rcm_list:
        for cy in cy_list:
            if cy[0] < 2006:
                rcp_list = ['85']
            else:
                rcp_list = indices.loc[dr_model].loc[rcm].index.astype(str).values.tolist()  

            for RCP in rcp_list:            
                init = indices.loc[dr_model].loc[rcm].loc[int(RCP)].init
                if cy[0] < 2006:
                    cutout_name = 'Cordex-' + dr_model + '-' + rcm + '-historical-' + init + '-' + str(cy[0]) + '-' + str(cy[1])
                else:
                    cutout_name = 'Cordex-' + dr_model + '-' + rcm + '-rcp' + RCP + '-' + init + '-' + str(cy[0]) + '-' + str(cy[1])

                # Load cutout
                cutout = atlite.Cutout(cutout_name,
                                       cutout_dir=cutout_dir)
                
                # Generate inflow time series
                inflow = cutout.hydro(hplants, hydrobasin_database)
                
                for c in range(len(country_codes)):
                    c_index = hplants[hplants['country_code'] == country_codes[c]].index
                    inflow_agg = inflow.loc[c_index].sum(axis=0)
                    data_df = pd.DataFrame(
                                index=pd.Series(
                                    data = cutout.coords["time"],
                                    name = 'utc_time'),
                                columns = pd.Series(
                                    data = ['inflow'], 
                                    name = 'names')
                                ) 
                    data_df['inflow']=inflow_agg
                    data_df.to_csv(resdatapath + 'modelled_inflow_' + country_codes[c] + '_' + cutout_name + '_weighted.csv', sep=';',compression = 'zip')