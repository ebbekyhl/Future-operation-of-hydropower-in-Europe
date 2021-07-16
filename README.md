# Public_hydro

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository consists of the modelled inflow time series obtained with [atlite](https://github.com/PyPSA/atlite), and historical inflow acquired from various sources. 
The methodology is described in detail in the paper [paper_name](paper_link).
The [JRC hydropower plants database](https://github.com/energy-modelling-toolkit/hydro-power-database) and the [HydroBASINS data](https://www.hydrosheds.org/page/hydrobasins) are used to produce the inflow time series.

The main script trains the modelled inflow at the beginning of the century (BOC) using historical inflow to obtain a month and country dependent 
retain factor which is subsequently applied to calibrate modelled inflow at the end of the century (EOC). 
The scripts are compatible with the following Python packages:
> python 3.7, numpy 1.19.1, matplotlib 3.1.0, pandas 1.0.3, scipy 1.5.2, atlite 0.02, numpy 1.19.1, matplotlib 3.1.0, pandas 1.0.3, scipy 1.5.2, seaborn 0.11.1, sklearn 0.0, basemap 1.3.0, basemap-data-hires 1.2.2

Following folders are contained in the repository:
  - __cutouts__: Climate model runoff cutouts
  - __figure__: Resulting figures 
  - __gendata__: General data directory (e.g. hydroBASINS, hydropower plant database, etc.)
  - __histdata__: Historical inflow data (See .txt file __data_sources.txt for data sources)
  - __moddata__: Modelled inflow data (Compressed .csv files with raw modelled inflow data prior to calibration)
  - __resdata__: Calibrated modelled inflow data
  - __scripts__: Folder containing the python scripts described below.

The full procedure of achieving the time series is described in the following. Step 1 and 2 have already been performed, but can be repeated if new and improved Global Climate Models (GCM) or Regional Climate Models (RCM) are available.

## 1: Download climate model data from CORDEX
Climate models are acquired from [ESGF](
https://esgf-data.dkrz.de/search/cordex-dkrz/ 
) with the following search criteria:
  - **Driving Model**: General Circulation Model
  - **RCM Model**: Regional Climate Model
  - **Time Frequency**: day
  - **Variable**: mrro (runoff mass flux)
 
## 2: Convert EURO-CORDEX runoff
Run the script __Atlite_inflow.py__ by first defining the variables:
- cy_list: list of 5 years cutout intervals
- dr_model_list: General circulation model (called driving model in CORDEX terms)
- rcm_list: Regional climate model
- hydrotype: Type of hydropower plants included (options: 'all','HDAM','HPHS', or 'HROR')

It first uses the function __Atlite_cutout.py__ to create monthly .nc cutouts. Each .nc file is ~12 mb. To run data for 2 (eras) x 30 (years/era) x 12 (months/year) months, the cutout alone requires ~9 gb disc space for each climate model.)
The cutout is then converted into hourly inflow time series using the ATLITE hydro conversion scheme. The output is .csv files of 5 years intervals on a country-level. Each .csv file is ~400 kb, corresponding to ~108 mb required disc space in total for each climate model.

## 3: Calibrate modelled inflow and map the results
Run the script __Climate_change_impact_hydro_Map_Results.py__ to generate __hourly time series of inflow in units of MWh__ for 22 (11) countries including (excluding) countries for which historical data is provided by Wattsight. 
Define the following variables before running the script:
- gcm_list: List of general circulation models included
- rcm_list: List of regional climate models included
- rcp_list: List of representative concentration pathways 
- WI: Wattsight data included (1) or not (0)
- matrix: Whether inflow GCM-RCM matrix is created (1) or not (0)

The time series are found in the folder __resdata__ with file names e.g. **Hydro_inflow_BOC_ensemble_mean_rcp85** (BOC: mean of 1991-2020) and **Hydro_inflow_EOC_ensemble_mean_rcp85** (EOC: mean of 2071-2100) for RCP8.5.
The script produces a map of the mean European inflow at EOC relative to BOC, and the change in the seasonal inflow for the countries with the largest hydropower capacity.
Furthermore, figures presenting the interannual inflow distributions at the two eras are presented, containing observations from all applied climate models. Lastly, the script creates figures depicting periods of extreme events.