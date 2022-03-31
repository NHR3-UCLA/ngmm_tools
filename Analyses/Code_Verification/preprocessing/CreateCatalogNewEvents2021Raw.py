#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 09:37:23 2021

@author: glavrent
"""
# %% Required Packages
# ======================================
#load libraries
import os
import sys
import pathlib
import re
#arithmetic libraries
import numpy as np
import pandas as pd
#geographical libraries
import geopy
from geopy.distance import distance
#plotting libraries
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
#user-derfined functions
sys.path.insert(0,'../../Python_lib/catalog')
sys.path.insert(0,'../../Python_lib/plotting')
import pylib_catalog as pylib_catalog
import pylib_contour_plots as pylib_cplt

# %% Define variables
# ======================================
#input file names
fname_nrec_eq     = '../../../Raw_files/nga_w3/IRIS/fdsnws-events_CA_2011-2021.csv'
fname_nerc_sta    = '../../../Raw_files/nga_w3/IRIS/fdsn-station_USA_[HB][HN]?.csv'
fname_mag_rup_lim = '../../../Data/Verification/preprocessing/flatfiles/usable_mag_rrup/usable_Mag_Rrup_coeffs.csv'

#output directoy
dir_out = '../../../Data/Verification/preprocessing/flatfiles/CA_NV_2011-2021_raw/'
dir_fig = dir_out + 'figures/'

fname_new_cat = 'Catalog_California_2011-2021.ver02'

#station 
srate_min = 10 #minimum sample rate for accepting instrument
net_seis = np.array(['AZ','BK','CI','NC','NN','NP','PB','PG','SB','WR','US','CJ','UO']) 

# %% Load Data
# ======================================
df_nrec_eq  = pd.read_csv(fname_nrec_eq,  delimiter='|', skiprows=4)
df_nrec_sta = pd.read_csv(fname_nerc_sta, delimiter='|', skiprows=4)

#M/R limitis
df_lim_mag_rrup = pd.read_csv(fname_mag_rup_lim, index_col=0)

# %% Process Data
# ======================================
#rename earthquake and station coordinates
df_nrec_eq.rename(columns={'Latitude':'eqLat',   'Longitude':'eqLon', 'Depth':'eqDepth'}, inplace=True)
df_nrec_sta.rename(columns={'Latitude':'staLat', 'Longitude':'staLon','Depth':'staDepth'}, inplace=True)

#remove rows with nna columns
df_nrec_eq  = df_nrec_eq.loc[~df_nrec_eq.isnull().any(axis=1).values,:]
df_nrec_sta = df_nrec_sta.loc[~df_nrec_sta.isnull().any(axis=1).values,:]
df_nrec_eq  = df_nrec_eq.loc[~df_nrec_eq.isna().any(axis=1).values,:]
df_nrec_sta = df_nrec_sta.loc[~df_nrec_sta.isna().any(axis=1).values,:]

#remove invalid columns
i_nan_rec = np.array([bool(re.match('^#.*$',n)) for n in df_nrec_sta['Network']])
df_nrec_sta = df_nrec_sta.loc[~i_nan_rec,:]

#keep networks of interest
i_net = df_nrec_sta.Network.isin(net_seis)
df_nrec_sta = df_nrec_sta.loc[i_net,:]

#with only records with sufficient sampling rate
df_nrec_sta.SampleRate = df_nrec_sta.SampleRate.astype(float)
i_srate = df_nrec_sta.SampleRate > srate_min
df_nrec_sta = df_nrec_sta.loc[i_srate,:]

#create network and station IDs
_, df_nrec_sta.loc[:,'NetworkID'] = np.unique(df_nrec_sta['Network'].values.astype(str), return_inverse=True)
_, df_nrec_sta.loc[:,'StationID'] = np.unique(df_nrec_sta['Station'].values.astype(str), return_inverse=True)

#reduce to unique stations
_, i_sta_unq  = np.unique(df_nrec_sta[['NetworkID','StationID']], return_index=True, axis=0)
df_nrec_sta = df_nrec_sta.iloc[i_sta_unq,:]

#station coordinates
sta_latlon = df_nrec_sta[['staLat','staLon']].values

#initialize rec catalog
cat_new_rec = []

#number of events and stations
n_eq  = len(df_nrec_eq)
n_sta = len(df_nrec_sta)

#iterate evetns
for (k, eq) in df_nrec_eq.iterrows():
    print('Processing event %i of %i'%(k, n_eq))
    #earthquake info
    eq_latlon = eq[['eqLat','eqLon']].values
    eq_depth  = eq['eqDepth']
    eq_mag    = eq['Magnitude']

    #epicenter and hypocenter dist
    dist_epi = np.array([distance(eq_latlon, sta_ll).km for sta_ll in sta_latlon])
    dist_hyp = np.sqrt(dist_epi**2 + eq_depth**2)

    #stations that satisfy the M/R limit
    i_sta_event = pylib_catalog.UsableSta(np.full(n_sta, eq_mag), dist_hyp, df_lim_mag_rrup)
    
    #create catalog for k^th event
    df_new_r = df_nrec_sta.loc[i_sta_event,:].assign(**eq)
    #add rupture info
    df_new_r.loc[:,'HypDist'] = dist_hyp[i_sta_event]

    #combine sta with event info
    cat_new_rec.append(df_new_r)

#combine catalogs of all events into one
df_cat_new_rec = pd.concat(cat_new_rec).reset_index()

#re-roder columns
df_cat_new_rec = df_cat_new_rec[np.concatenate([df_nrec_eq.columns, df_nrec_sta.columns,['HypDist']])]

#create event and station dataframes
#indices for unique earthquakes and stations
eq_idx  = np.unique(df_cat_new_rec.EventID,  return_index=True)[1]
sta_idx = np.unique(df_cat_new_rec[['NetworkID','StationID']], return_index=True, axis=0)[1]
#event dataframe
df_cat_new_rec_eq  = df_cat_new_rec.loc[eq_idx, df_nrec_eq.columns].reset_index(drop=True)
#station dataframe
df_cat_new_rec_sta = df_cat_new_rec.loc[sta_idx, df_nrec_sta.columns].reset_index(drop=True)



# %% Output
# ======================================
# create output directories
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
if not os.path.isdir(dir_fig): pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True)

#save processed dataframes
fname_cat = '%s%s'%(dir_out, fname_new_cat )
df_cat_new_rec.to_csv(fname_cat    + '.csv',         index=False)
df_cat_new_rec_eq.to_csv(fname_cat   + '_event.csv',   index=False)
df_cat_new_rec_sta.to_csv(fname_cat + '_station.csv', index=False)

# create figures
# ----   ----   ----   ----   ----
# Mag-Dist distribution
fname_fig = 'M-R_dist'
#create figure   
fig, ax = plt.subplots(figsize = (10,9))
pl1 = ax.scatter(df_cat_new_rec.HypDist, df_cat_new_rec.Magnitude)
#edit figure properties
ax.set_xlabel(r'Distance ($km$)', fontsize=30)
ax.set_ylabel(r'Magnitude', fontsize=30)
ax.grid(which='both')
# ax.set_xscale('log')
# ax.set_xlim([0.1, 2000])
ax.set_ylim([1,   8])
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
# ax.legend(fontsize=25, loc='upper left')
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', right='on')
fig.tight_layout()
#save figure
fig.savefig( dir_fig + fname_fig + '.png' )

#log scale figure
ax.set_xscale('log')
fig.tight_layout()
#save figure
fig.savefig( dir_fig + fname_fig + '_log' + '.png' )

# Mag-Year distribution
fname_fig = 'M-date_dist'
#create figure   
fig, ax = plt.subplots(figsize = (10,9))
pl1 = ax.scatter(pd.DatetimeIndex(df_cat_new_rec['Time']).year, df_cat_new_rec['Magnitude'].values)
#edit figure properties
ax.set_xlabel(r'time ($year$)', fontsize=30)
ax.set_ylabel(r'Magnitude', fontsize=30)
ax.grid(which='both')
# ax.set_xscale('log')
ax.set_xlim([1965, 2025])
ax.set_ylim([1,   8])
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
# ax.legend(fontsize=25, loc='upper left')
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', right='on')
fig.tight_layout()
#save figure
fig.savefig( dir_fig + fname_fig + '.png' )

#eq and sta location
fname_fig = 'eq_sta_locations'
fig, ax, data_crs, gl = pylib_cplt.PlotMap(flag_grid=True)
#plot earthquake and station locations
ax.plot(df_cat_new_rec_eq['eqLon'].values,    df_cat_new_rec_eq['eqLat'].values,   '*', transform = data_crs, markersize = 10, zorder=13, label='Events')
ax.plot(df_cat_new_rec_sta['staLon'].values, df_cat_new_rec_sta['staLat'].values, 'o', transform = data_crs, markersize = 6,  zorder=13, label='Stations')
#edit figure properties
gl.xlabel_style = {'size': 25}
gl.ylabel_style = {'size': 25}
# gl.xlocator = mticker.FixedLocator([-124, -122, -120, -118, -116, -114])
# gl.ylocator = mticker.FixedLocator([32, 34, 36, 38, 40])
ax.legend(fontsize=25, loc='lower left')
# ax.set_xlim(plt_latlon_win[:,1])
# ax.set_ylim(plt_latlon_win[:,0])
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

# Print data info
# ---------------------------
print(f'New Records:')
print(f'\tnumber of rec: %.i'%len(df_cat_new_rec))
print(f'\tnumber of rec (R<200km): %.i'%np.sum(df_cat_new_rec.HypDist<=200))
print(f'\tnumber of rec (R<300km): %.i'%np.sum(df_cat_new_rec.HypDist<=300))
print(f'\tnumber of eq: %.i'%len(df_cat_new_rec_eq))
print(f'\tnumber of sta: %.i'%len(df_cat_new_rec_sta))
