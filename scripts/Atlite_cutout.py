# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:28:43 2020

This script reads runoff and altitude netcdf files from Cordex, and splits them 
up into monthly netcdf files with formats compatible with ATLITE. Thus, these 
converted files can be applied to aggregate runoff in European reservoirs using 
the ATLITE modulus in Python. This script might cause a netcdf HDF-error if disc 
space is low.
Modules version: netCDF4 1.4.2 and numpy 1.19.1

@author: ebbek
"""
def Atlite_cutout(years_list,dr_model_list,rcm_list,cutout_path,data_path):
    import netCDF4
    from netCDF4 import Dataset
    import numpy as np
    import pandas as pd
    import numpy.ma as ma
    import os

    if years_list[-1][-1] < 2006:
        exp = 'historical'
    else:
        exp = 'rcp'
        
    indices = pd.read_csv(data_path + 'index.csv',sep=';',index_col=[0,1,5])
    for dr_model in dr_model_list:
        for rcm in rcm_list:
            inst = indices.loc[dr_model].loc[rcm].loc[85].inst
            if exp == 'historical':
                rcp_list = ['85']
            else:
                rcp_list = ['26'] #indices.loc[dr_model].loc[rcm].index.astype(str).values.tolist()            
            for RCP in rcp_list:
                init = indices.loc[dr_model].loc[rcm].loc[int(RCP)].init
                downscal = indices.loc[dr_model].loc[rcm].loc[int(RCP)].downscal
                for year_it in range(len(years_list)):
                    years = years_list[year_it]
                    # Directory of cutout
                    if exp == 'historical':
                        cutout_dir = cutout_path + '/Cordex-' + dr_model + '-' + rcm + '-historical-' + init + '-' + str(years[0]) + '-' + str(years[1]) + '/'
                        if dr_model == 'MOHC-HadGEM2-ES':
                            file1 = data_path + 'mrro_EUR-11_' + dr_model + '_historical_' + init + '_' + inst + '-' + rcm + '_' + downscal + '_day_' + str(years[0]) + '0101-' + str(years[1]) + '1230.nc' # Runoff [kg/(m^2*s)]
                        else:
                            file1 = data_path + 'mrro_EUR-11_' + dr_model + '_historical_' + init + '_' + inst + '-' + rcm + '_' + downscal + '_day_' + str(years[0]) + '0101-' + str(years[1]) + '1231.nc' # Runoff [kg/(m^2*s)]
                    else:
                        cutout_dir = cutout_path + '/Cordex-' + dr_model + '-' + rcm + '-rcp' + RCP + '-' + init + '-' + str(years[0]) + '-' + str(years[1]) + '/'
                        if dr_model == 'MOHC-HadGEM2-ES':
                            if years[1] == 2100:
                                if RCP != '45':
                                    file1 = data_path + 'mrro_EUR-11_' + dr_model + '_rcp' + RCP + '_' + init + '_' + inst + '-' + rcm + '_' + downscal + '_day_' + str(years[0]) + '0101-' + '20991230.nc' # Runoff [kg/(m^2*s)]
                                else:
                                    file1 = data_path + 'mrro_EUR-11_' + dr_model + '_rcp' + RCP + '_' + init + '_' + inst + '-' + rcm + '_' + downscal + '_day_' + str(years[0]) + '0101-' + '20991130.nc' # Runoff [kg/(m^2*s)]
                            else:
                                file1 = data_path + 'mrro_EUR-11_' + dr_model + '_rcp' + RCP + '_' + init + '_' + inst + '-' + rcm + '_' + downscal + '_day_' + str(years[0]) + '0101-' + str(years[1]) + '1230.nc' # Runoff [kg/(m^2*s)]
                        else:
                            file1 = data_path + 'mrro_EUR-11_' + dr_model + '_rcp' + RCP + '_' + init + '_' + inst + '-' + rcm + '_' + downscal + '_day_' + str(years[0]) + '0101-' + str(years[1]) + '1231.nc' # Runoff [kg/(m^2*s)]
    
                    create_folder = os.path.join(cutout_dir)
                    if not os.path.exists(create_folder):
                        os.mkdir(create_folder)
                    # Cutout years
                    y_list = list(np.arange(years[0],years[1]+1))# Years that the cordex nc-file contains
                    y_write = y_list # Years to write (Must be contained in "y_list")
                    
                    # Name of cordex runoff and altitude data
                    file2 = data_path + 'altitude.nc' # Altitude [m]
                    # Read cordex nc file
                    cordex1 = netCDF4.Dataset(file1,'r') # Read runoff
                    cordex2 = netCDF4.Dataset(file2,'r') # Read altitude                  
                    runoff_cordex = cordex1.variables["mrro"][:].data*24*60*60/1000 # Array containing runoff data # Convert kg/(m^2*s) to m
                    altitude_cordex = cordex2.variables["orog"][:] # Array containing altitude
                    runoff_cordex[runoff_cordex < 0] = 0 # Removing negative runoff values
                    
                    if dr_model == 'MOHC-HadGEM2-ES' and years[1] == 2100:
                        if RCP == '45':
                            runoff_cordex = ma.append(runoff_cordex, runoff_cordex[-30:],axis=0) # Repeat last month due to data missing
                        runoff_cordex = ma.append(runoff_cordex, runoff_cordex[-360:],axis=0) # Repeat last year due to data missing
                    
                    rlats = cordex1.variables["rlat"][:] # Latitudes in reference CS
                    rlons = cordex1.variables["rlon"][:] # Longitudes in reference CS
                    rotpole = cordex1.variables['rotated_pole'] # Reference CS's pole
                    
                    lats = cordex1.variables["lat"][:] # Latitudes
                    lons = cordex1.variables["lon"][:] # Longitudes
                    
                    if dr_model != 'MOHC-HadGEM2-ES' and dr_model != 'NCC-NorESM1-M':
                        time_cordex = cordex1.variables['time'][:] # Time (Daily resolution)
                    else:
                        if exp == 'historical':
                            file3 = data_path + 'mrro_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_SMHI-RCA4_v1a_day_' + str(years[0]) + '0101-' + str(years[1]) + '1231.nc'
                        else:
                            file3 = data_path + 'mrro_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r1i1p1_SMHI-RCA4_v1a_day_' + str(years[0]) + '0101-' + str(years[1]) + '1231.nc'
                            
                        cordex3 = netCDF4.Dataset(file3,'r')
                        time_cordex = cordex3.variables['time'][:] # Time (Daily resolution)
                        
                    # Creating an array containing the monthly day indices in one year (different for leap years)
                    t_ind = [0,31,59,90,120,151,181,212,243,273,304,334,366] # Index array for a regular year
                    t_ind_leap = [0,31,60,91,121,152,182,213,244,274,305,335,367] # Index array for a leap year
                    t_ind_HadGEM = [0,30,60,90,120,150,180,210,240,270,300,330,361]
                    m_arr = ['01','02','03','04','05','06','07','08','09','10','11','12'] # Month number array
                    unout = 'hours since 1949-12-01 00:00:00' # This is how cordex present their data
                    unout1 = 'hours since ' + str(y_write[0]) + '-01-01 00:00:00' # Hours since first timestamp of the data
                    d_year = np.zeros(len(y_list)) # Array containing number of days for each year
                    d_year_sum = np.zeros(len(y_list)+1) # Cumulated days
                    A = 0
                    for yr in range(len(y_list)):
                        if ((y_list[yr]%4 == 0) and (y_list[yr]%100 != 0)) or (y_list[yr]%400 == 0): # Leap year
                            d_year[yr] = 366
                        else: # Regular year
                            d_year[yr] = 365 
                        
                        A += d_year[yr]
                        d_year_sum[yr+1] = A
                    d_year_sum = d_year_sum.astype(int)
                    d_year_sum_HadGEM = [0,360,720,1080,1440,1800]
                    d_year_sum_NorESM = [0,365,730,1095,1460,1825]
                    
                    #%%
                    # Create meta file
                    ncout = Dataset(cutout_dir + 'meta.nc','w','NETCDF4'); # using netCDF4 for output format 
                    ncout.Conventions = 'CF-1.6' # This was copied from an ATLITE template
                    ncout.history = '2020-09-18 10:53:36 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data6/adaptor.mars.internal-1600426409.4983075-19142-19-08d266a1-241c-45c3-93b5-546fe0696151.nc /cache/tmp/08d266a1-241c-45c3-93b5-546fe0696151-adaptor.mars.internal-1600426409.4989018-19142-6-tmp.grib' # This was copied from an ATLITE template
                    ncout.module = 'cordex'# Module (e.g. "era5" or "cordex")
                    ncout.createDimension('x',len(rlons));
                    ncout.createDimension('y',len(rlats));
                    ncout.createDimension('time',sum(d_year[y_list.index(y_write[0]):(y_list.index(y_write[-1])+1)].astype(int))*24);
                    ncout.createDimension('year',len(y_write));
                    ncout.createDimension('month',len(m_arr));
                    xvar = ncout.createVariable('x','float32',('x'));xvar[:] = rlons;
                    yvar = ncout.createVariable('y','float32',('y'));yvar[:] = rlats;
                    timevar = ncout.createVariable('time','int32',('time'));timevar.setncattr('units',unout1);timevar[:] = np.arange(sum(d_year[y_list.index(y_write[0]):(y_list.index(y_write[-1])+1)].astype(int))*24);
                    yearvar = ncout.createVariable('year','int32',('year'));yearvar[:] = y_write;
                    monthvar = ncout.createVariable('month','int32',('month'));monthvar[:] = np.arange(1,13);
                    lonvar = ncout.createVariable('lon','float32',('y','x'));lonvar[:] = lons;
                    latvar = ncout.createVariable('lat','float32',('y','x'));latvar[:] = lats;
                    heightvar = ncout.createVariable('height','float32',('y','x'));heightvar[:] = altitude_cordex;
                    ncout.close();
                    
                    #%%
                    # Create monthly cutout files
                    for j in range(len(y_write)):
                        for i in np.arange(12):
                            month = m_arr[i]
                            year = y_write[j]
                            t_cordex = time_cordex[d_year_sum[y_list.index(year)]:d_year_sum[y_list.index(year)+1]]
                            if len(t_cordex) == 365: # Not a leap year
                                index = t_ind
                            elif len(t_cordex) == 366: # Leap year
                                index = t_ind_leap
                                    
                            if i < 11:
                                t = np.arange((index[i+1] - index[i])*24) + t_cordex[index[i]]*24 # time hourly resolution
                            else:
                                t = np.arange((index[i+1]-1 - index[i])*24) + t_cordex[index[i]]*24 # time hourly resolution     
                                    
                            if dr_model != 'MOHC-HadGEM2-ES' and dr_model != 'NCC-NorESM1-M':# 366 calendar system
                                r_cordex = runoff_cordex[d_year_sum[y_list.index(year)]:d_year_sum[y_list.index(year)+1]]
                                runoff_rep = np.repeat(r_cordex[index[i]:index[i+1]],repeats=24,axis=0) # Converting daily to hourly resolution
                            elif dr_model == 'MOHC-HadGEM2-ES': # 360 calendar system
                                r_cordex = runoff_cordex[d_year_sum_HadGEM[y_list.index(year)]:d_year_sum_HadGEM[y_list.index(year)+1]]
                                runoff_rep = np.repeat(r_cordex[t_ind_HadGEM[i]:t_ind_HadGEM[i+1]],repeats=24,axis=0) # Converting daily to hourly resolution
                            
                                d_diff = (index[i+1] - index[i]) - (t_ind_HadGEM[i+1] - t_ind_HadGEM[i])
                                if d_diff > 0:
                                    for k in np.arange(d_diff):
                                        runoff_rep = ma.append(runoff_rep, runoff_rep[-24:],axis=0)
                                elif d_diff < 0:
                                    runoff_rep = runoff_rep[:np.abs(d_diff)*-24]
    
                            elif dr_model == 'NCC-NorESM1-M': # 365 calendar system
                                r_cordex = runoff_cordex[d_year_sum_NorESM[y_list.index(year)]:d_year_sum_NorESM[y_list.index(year)+1]]
                                runoff_rep = np.repeat(r_cordex[t_ind[i]:t_ind[i+1]],repeats=24,axis=0) # Converting daily to hourly resolution
                            
                                d_diff = (index[i+1] - index[i]) - (t_ind[i+1] - t_ind[i])
                                if d_diff > 0:
                                    for k in np.arange(d_diff):
                                        runoff_rep = ma.append(runoff_rep, runoff_rep[-24:],axis=0)
                                elif d_diff < 0:
                                    runoff_rep = runoff_rep[:np.abs(d_diff)*-24]
                            
                            runoff_hourly = np.divide(runoff_rep,24) # Converting daily to hourly mean values
                            
                            ncout = Dataset(cutout_dir + str(year) + month + '.nc','w','NETCDF4') # using netCDF4 for output format 
                            ncout.createDimension('x',len(rlons))
                            ncout.createDimension('y',len(rlats))
                            ncout.createDimension('time',len(t))
                            xvar = ncout.createVariable('x','float32',('x'));xvar[:] = rlons;
                            yvar = ncout.createVariable('y','float32',('y'));yvar[:] = rlats;
                            timevar = ncout.createVariable('time','int32',('time'));timevar.setncattr('units',unout);timevar[:] = t.round().astype(int);
                            runoffvar = ncout.createVariable('runoff','float32',('time','y','x'),zlib=True,complevel=4);runoffvar[:] = runoff_hourly;
                            lonvar = ncout.createVariable('lon','float32',('y','x'),zlib=True,complevel=4);lonvar[:] = lons;
                            latvar = ncout.createVariable('lat','float32',('y','x'),zlib=True,complevel=4);latvar[:] = lats;
                            heightvar = ncout.createVariable('height','float32',('y','x'),zlib=True,complevel=4);heightvar[:] = altitude_cordex;
                            ncout.close();
                            print(str(year) + '-' + str(month) + ' done')