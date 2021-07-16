# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:37:37 2021

@author: ebbek
"""
import pandas as pd
import atlite
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore
from colorama import Style
countries = ['NO','FR','ES','CH','SE','DE','AT','IT','BG','HR','PT','RO',
             'CZ','HU','BA','RS','SI','FI','PL','SK','MK','ME'] 
# retrieve hydro power plants database
hydropower_database = 'data/hydro-power-database-master/data/jrc-hydro-power-plant-database.csv'
hydrobasin_database = 'data/hydroBASINS/hybas_eu_lev08_v1c.shp'
h = pd.read_csv(hydropower_database)    
h['Coordinates'] = list(zip(h.lon, h.lat))
h['Coordinates'] = h['Coordinates'].apply(Point)
g = gpd.GeoDataFrame(h, geometry='Coordinates')

cutout_name = 'Cordex-MPI-M-MPI-ESM-LR-RCA4-rcp85-r1i1p1-2006-2010'
cutout = atlite.Cutout(cutout_name, cutout_dir=cutout_dir)

for c in range(len(countries)):
    country = countries[c]
    damhydro = g[h.country_code==country] # Dataframe with all types of hydro
    i = -1
    a = np.zeros(len(damhydro))
    for k in damhydro.index:
        i = i + 1
        try:
            inflow = cutout.hydro(damhydro[damhydro.index == k], hydrobasin_database)
        except:
            a[i] = k
            pass
            
        print(f'{Fore.GREEN}Finding inapplicable hydrodam stations in {Style.RESET_ALL}' + str(country) + f'{Fore.GREEN} hydropower database:{Style.RESET_ALL}' )
        print(f'{Fore.GREEN}Station number {Style.RESET_ALL}' + str(i+1) + f'{Fore.GREEN} out of {Style.RESET_ALL}' + str(len(damhydro)) + '!')
            
    arr_drop = a[a!=0]
    np.savetxt('data/arr_drop_' + country + '.csv',arr_drop)

