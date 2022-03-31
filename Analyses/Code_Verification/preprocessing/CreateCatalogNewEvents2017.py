#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 16:12:57 2021

@author: glavrent
"""
# Required Packages
# ======================================
#load libraries
import os
import sys
import pathlib
import glob
import re           #regular expression package
#arithmetic libraries
import numpy as np
import pandas as pd
#geographic coordinates
import pyproj
#plotting libraries
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
#user-derfined functions
sys.path.insert(0,'../../Python_lib/ground_motions')
sys.path.insert(0,'../../Python_lib/plotting')
import pylib_Willis15CA_Vs30 as pylib_W15_Vs30
import pylib_contour_plots as pylib_cplt

# %% Define Input Data
# ======================================
#thresholds 
dt_thres = 0.025
rrup_thres = 300
# projection system
utm_zone = '11S'
#input flatfiles
fname_flatfile_newrec_eq  = '../../../Raw_files/nga_w3/California2011-2017/eventcatalog.csv'
fname_flatfile_newrec_sta = '../../../Raw_files/nga_w3/California2011-2017/Recorddata.csv'

#flatfile file
fname_flatfile = 'CatalogNewRecords_2011-2017_CA_NV'

#output directory
dir_out = '../../../Data/NGAWest_expansion/CA_NV_2011-2017/'
dir_out = '../../../Data/Verification/preprocessing/flatfiles/CA_NV_2011-2017/'
dir_fig = dir_out + 'figures/'



# %% Load Data
# ======================================
#merge event and station info
df_flatfile_newrec_eq  = pd.read_csv(fname_flatfile_newrec_eq)
df_flatfile_newrec_sta = pd.read_csv(fname_flatfile_newrec_sta)
df_flatfile_newrec     = pd.merge(df_flatfile_newrec_eq, df_flatfile_newrec_sta, left_on='EventIDs.i.', right_on='EventID')

# %% Cleaning files
# ======================================
#set -999 to nan
df_flatfile_newrec.replace(-999, np.nan, inplace=True)
#remove data based on timestep 
df_flatfile_newrec = df_flatfile_newrec[ df_flatfile_newrec.timeStep <= dt_thres ]
#remove data with unknown mag 
df_flatfile_newrec = df_flatfile_newrec[ ~np.isnan(df_flatfile_newrec['mag.event']) ]
#remove data with unknown coordinates
df_flatfile_newrec = df_flatfile_newrec[ ~np.isnan(df_flatfile_newrec[['latitude.event', 'longitude.event']]).any(axis=1) ]
df_flatfile_newrec = df_flatfile_newrec[ ~np.isnan(df_flatfile_newrec[['stnlat', 'stnlon']]).any(axis=1) ]
#keep single record from 
df_flatfile_newrec.loc[:,'EventID']   = df_flatfile_newrec['EventID'].values.astype(int)
df_flatfile_newrec.loc[:,'stationID'] = np.unique(df_flatfile_newrec['station'], return_inverse=True)[1]
i_unq_eq_sta                          = np.unique( np.unique(df_flatfile_newrec[['stationID','EventID']].values, return_index=True, axis=0)[1] )
df_flatfile_newrec                    = df_flatfile_newrec.iloc[i_unq_eq_sta, :]

#reset indices
df_flatfile_newrec.reset_index(inplace=True)

# %% Process Data
# ======================================
#coordinates and projection system
# projection system
utmProj = pyproj.Proj("+proj=utm +zone="+utm_zone+", +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

#earthquake and station ids
eq_id_newrec   = df_flatfile_newrec['EventIDs.i.'].values.astype(int)
sta_id_newrec    = df_flatfile_newrec['station'].values
sta_net_newrec   = df_flatfile_newrec['network'].values
sta_netid_newrec = [f'{s_net}-{s_id}' for s_net, s_id  in zip(sta_net_newrec, sta_id_newrec)]

#unique earthquake and station ids
eq_id_newrec_unq,  eq_idx_newrec  = np.unique(eq_id_newrec,  return_index=True)
# sta_id_newrec_unq, sta_inv_newrec = np.unique(sta_id_newrec, return_inverse=True)
sta_id_newrec_unq, sta_inv_newrec = np.unique(sta_netid_newrec, return_inverse=True)

#number of earthquake and stations
neq_newrec  = len(eq_id_newrec_unq)
nsta_newrec = len(sta_id_newrec_unq)

#earthquake and station coordinates
eq_latlon_newrec_all  = df_flatfile_newrec[['latitude.event', 'longitude.event']].values
sta_latlon_newrec_all = df_flatfile_newrec[['stnlat', 'stnlon']].values

#utm coordinates
eq_X_newrec_all  = np.array([utmProj(e_lon, e_lat) for e_lat, e_lon in zip(eq_latlon_newrec_all[:,0], eq_latlon_newrec_all[:,1])]) / 1000
eq_z_newrec_all  = np.minimum(-1*df_flatfile_newrec['depth.event.1000'].values, 0)
sta_X_newrec_all = np.array([utmProj(s_lon, s_lat) for s_lat, s_lon in zip(sta_latlon_newrec_all[:,0], sta_latlon_newrec_all[:,1])]) / 1000
mpt_X_newrec_all = (eq_X_newrec_all + sta_X_newrec_all) / 2

#mid point coordinates
mpt_latlon_newrec_all = np.flip( np.array([utmProj(pt_x, pt_y, inverse=True) for pt_x, pt_y in 
                                           zip(mpt_X_newrec_all[:,0], mpt_X_newrec_all[:,1]) ]), axis=1 )


#earthquake parameteres
mag_newrec = df_flatfile_newrec['mag.event'].values
rup_newrec = np.sqrt(np.linalg.norm(eq_X_newrec_all-sta_X_newrec_all, axis=1)**2+eq_z_newrec_all**2)

#year of recording
df_flatfile_newrec['year'] = pd.DatetimeIndex(df_flatfile_newrec['rec.stime']).year

#estimate station vs30
Wills15Vs30 = pylib_W15_Vs30.Willis15Vs30CA()
vs30_newrec = Wills15Vs30.lookup(np.fliplr(sta_latlon_newrec_all))[0]
vs30_newrec[vs30_newrec<=50] = np.nan

# %% Process Data to save
# ======================================
#distance threshold for data to keep
i_data2keep = rup_newrec <= rrup_thres

#records' info
rsn_array  = df_flatfile_newrec.loc[i_data2keep,'index'].values
eqid_array = eq_id_newrec[i_data2keep]
ssn_array  = sta_inv_newrec[i_data2keep]
sid_array  = sta_id_newrec[i_data2keep]
snet_array = sta_net_newrec[i_data2keep]
year_array = df_flatfile_newrec['year'].values[i_data2keep]

#records' parameters
mag_array  = mag_newrec[i_data2keep]
rrup_array = rup_newrec[i_data2keep]
vs30_array = vs30_newrec[i_data2keep]

#earthquake, station, mid-point latlon coordinates
eq_latlon  = eq_latlon_newrec_all[i_data2keep,:]
sta_latlon = sta_latlon_newrec_all[i_data2keep,:]
mpt_latlon = mpt_latlon_newrec_all[i_data2keep,:]
#earthquake, station, mid-point UTM coordinates
eq_utm  = eq_X_newrec_all[i_data2keep,:]
sta_utm = sta_X_newrec_all[i_data2keep,:]
mpt_utm = mpt_X_newrec_all[i_data2keep,:]
#earthquake source depth
eq_z = eq_z_newrec_all[i_data2keep]

#indices for unique earthquakes and stations
eq_idx = np.unique(eqid_array,  return_index=True)[1]
sta_idx = np.unique(ssn_array, return_index=True)[1]

#data to save
data_full = {'rsn':rsn_array, 'eqid':eqid_array, 'ssn':ssn_array,
             'mag':mag_array, 'Rrup':rrup_array, 'Vs30': vs30_array, 'year': year_array, 
             'eqLat':eq_latlon[:,0], 'eqLon':eq_latlon[:,1], 'staLat':sta_latlon[:,0], 'staLon':sta_latlon[:,1], 'mptLat':mpt_latlon[:,0], 'mptLon':mpt_latlon[:,1],
             'UTMzone':utm_zone,
             'eqX':eq_utm[:,0], 'eqY':eq_utm[:,1], 'eqZ':eq_z, 'staX':sta_utm[:,0], 'staY':sta_utm[:,1], 'mptX':mpt_utm[:,0], 'mptY':mpt_utm[:,1]}

#processed dataframes
df_flatfile_full = pd.DataFrame(data_full)
#event dataframe
df_flatfile_event   = df_flatfile_full.loc[eq_idx, ['eqid','mag','year','eqLat','eqLon','UTMzone','eqX','eqY','eqZ']].reset_index(drop=True)
#station dataframe
df_flatfile_station = df_flatfile_full.loc[sta_idx, ['ssn','Vs30','staLat','staLon','UTMzone','staX','staY']].reset_index(drop=True)


# %% Save data
# ======================================
# create output directories
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
if not os.path.isdir(dir_fig): pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True)

#save processed dataframes
fname_flatfile_full= '%s%s'%(dir_out, fname_flatfile)
df_flatfile_full.to_csv(fname_flatfile_full    + '.csv',         index=True)
df_flatfile_event.to_csv(fname_flatfile_full   + '_event.csv',   index=False)
df_flatfile_station.to_csv(fname_flatfile_full + '_station.csv', index=False)

# create figures
# ----   ----   ----   ----   ----
# Mag-Dist distribution
fname_fig = 'M-R_dist'
#create figure   
fig, ax = plt.subplots(figsize = (10,9))
pl1 = ax.scatter(df_flatfile_full.Rrup, df_flatfile_full.mag, label='New Records')
#edit figure properties
ax.set_xlabel(r'Distance ($km$)', fontsize=30)
ax.set_ylabel(r'Magnitude', fontsize=30)
ax.grid(which='both')
ax.set_xscale('log')
# ax.set_xlim([0.1, 2000])
ax.set_ylim([2,   8])
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.legend(fontsize=25, loc='upper left')
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', right='on')
fig.tight_layout()
#save figure
fig.savefig( dir_fig + fname_fig + '.png' )

# Mag-Year distribution
fname_fig = 'M-date_dist'
#create figure   
fig, ax = plt.subplots(figsize = (10,9))
pl1 = ax.scatter(df_flatfile_event['year'].values, df_flatfile_event['mag'].values, label='New Records')
#edit figure properties
ax.set_xlabel(r'time ($year$)', fontsize=30)
ax.set_ylabel(r'Magnitude', fontsize=30)
ax.grid(which='both')
# ax.set_xscale('log')
ax.set_xlim([1965, 2025])
ax.set_ylim([2,   8])
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.legend(fontsize=25, loc='upper left')
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
ax.plot(df_flatfile_event['eqLon'].values,    df_flatfile_event['eqLat'].values,    '*', transform = data_crs, markersize = 10, zorder=13)
ax.plot(df_flatfile_station['staLon'].values, df_flatfile_station['staLat'].values, 'o', transform = data_crs, markersize = 6,  zorder=13, label='STA')
ax.plot(df_flatfile_event['eqLon'].values,    df_flatfile_event['eqLat'].values,    '*', color='black', transform = data_crs, markersize = 14, zorder=13, label='EQ')
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
print(r'New Records:')
print(f'\tnumber of rec: %.i'%len(df_flatfile_full))
print(f'\tnumber of rec (R<200km): %.i'%np.sum(df_flatfile_full.Rrup<=200))
print(f'\tnumber of rec (R<%.1f): %.i'%(rrup_thres, np.sum(df_flatfile_full.Rrup<=rrup_thres)))
print(f'\tnumber of eq: %.i'%len(df_flatfile_event))
print(f'\tnumber of sta: %.i'%len(df_flatfile_station))
print(f'\tnumber of sta (R<300km): %.i'%len(np.unique(ssn_array[df_flatfile_full.Rrup<=300])))
print(f'\tcoverage: %.i to %i'%(df_flatfile_full.year.min(), df_flatfile_full.year.max()))


