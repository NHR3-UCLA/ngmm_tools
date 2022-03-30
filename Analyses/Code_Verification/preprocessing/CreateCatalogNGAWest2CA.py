#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 22:58:16 2021

@author: glavrent
"""
# %% Required Packages
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
sys.path.insert(0,'../../Python_lib/catalog')
sys.path.insert(0,'../../Python_lib/plotting')
import pylib_catalog as pylib_catalog
import pylib_contour_plots as pylib_cplt

# %% Define Input Data
# ======================================
#thresholds 
thres_dist = 0.01  #collocated stations
# projection system
utm_zone = '11S'
# region id
region_id = 1

#input file names
fname_flatfile_NGA2ASK14 = '../../../../Nonerg_CA_GMM/Raw_files/GMM/nga_w2_resid/ASK14/run12_min3_R300/resid_T0.200.out2.txt'
fname_flatfile_NGA2coor  = '../../../Raw_files/NGAWest2/Updated_NGA_West2_Flatfile_coordinates.csv'

#flatfile file
fname_flatfile = 'CatalogNGAWest2CA_ASK14'

#output directory
dir_out = '../../../Data/Verification/preprocessing/flatfiles/NGAWest2_CA/'
dir_fig = dir_out + 'figures/'

# %% Load Data
# ======================================
#NGAWest2
df_flatfile_NGA2ASK14 = pd.read_csv(fname_flatfile_NGA2ASK14, delim_whitespace=True)
df_flatfile_NGA2coor  = pd.read_csv(fname_flatfile_NGA2coor)
df_flatfile_NGA2 = pd.merge(df_flatfile_NGA2ASK14, df_flatfile_NGA2coor, left_on='recID', right_on='Record Sequence Number')


# %% Cleaning files
# ======================================
# NGA2
#keep only CA for NGA2
df_flatfile_NGA2 = df_flatfile_NGA2[ df_flatfile_NGA2.region == region_id ]

#reset indices
df_flatfile_NGA2.reset_index(inplace=True)

# %% Process Data
# ======================================
#coordinates and projection system
# projection system
utmProj = pyproj.Proj("+proj=utm +zone="+utm_zone+", +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

#earthquake and station ids
eq_id_NGA2  = df_flatfile_NGA2['eqid'].values.astype(int)
sta_id_NGA2 = df_flatfile_NGA2['Station Sequence Number'].values.astype(int)

#unique earthquake and station ids
eq_id_NGA2_unq,  eq_idx_NGA2  = np.unique(eq_id_NGA2,  return_index=True)
sta_id_NGA2_unq, sta_idx_NGA2 = np.unique(sta_id_NGA2, return_index=True)

#number of earthquake and stations
neq_NGA2  = len(eq_id_NGA2_unq)
nsta_NGA2 = len(sta_id_NGA2_unq)

#earthquake and station coordinates
eq_latlon_NGA2_all  = df_flatfile_NGA2[['Hypocenter Latitude (deg)','Hypocenter Longitude (deg)']].values
sta_latlon_NGA2_all = df_flatfile_NGA2[['Station Latitude','Station Longitude']].values

#utm coordinates
eq_X_NGA2_all  = np.array([utmProj(e_lon, e_lat) for e_lat, e_lon in zip(eq_latlon_NGA2_all[:,0], eq_latlon_NGA2_all[:,1])]) / 1000
eq_z_NGA2_all  = -1*df_flatfile_NGA2['Hypocenter Depth (km)'].values
sta_X_NGA2_all = np.array([utmProj(s_lon, s_lat) for s_lat, s_lon in zip(sta_latlon_NGA2_all[:,0], sta_latlon_NGA2_all[:,1])]) / 1000
mpt_X_NGA2_all = (eq_X_NGA2_all + sta_X_NGA2_all) / 2

#mid point coordinates
mpt_latlon_NGA2_all = np.flip( np.array([utmProj(pt_x, pt_y, inverse=True) for pt_x, pt_y in 
                                           zip(mpt_X_NGA2_all[:,0], mpt_X_NGA2_all[:,1]) ]), axis=1)
#ground motion parameteres
mag_NGA2 = df_flatfile_NGA2['mag'].values
rup_NGA2 = np.sqrt(np.linalg.norm(eq_X_NGA2_all-sta_X_NGA2_all, axis=1)**2+eq_z_NGA2_all**2)
vs30_NGA2 = df_flatfile_NGA2['VS30'].values

# %% Process Data to save
# ======================================
i_data2keep = np.full(len(df_flatfile_NGA2), True)

#records' info
rsn_array  = df_flatfile_NGA2['recID'].values[i_data2keep].astype(int)
eqid_array = eq_id_NGA2[i_data2keep]
ssn_array  = sta_id_NGA2[i_data2keep]
year_array = df_flatfile_NGA2['YEAR'].values[i_data2keep]

#records' parameters
mag_array  = mag_NGA2[i_data2keep]
rrup_array = rup_NGA2[i_data2keep]
vs30_array = vs30_NGA2[i_data2keep]

#earthquake, station, mid-point latlon coordinates
eq_latlon  = eq_latlon_NGA2_all[i_data2keep,:]
sta_latlon = sta_latlon_NGA2_all[i_data2keep,:]
mpt_latlon = mpt_latlon_NGA2_all[i_data2keep,:]
#earthquake, station, mid-point UTM coordinates
eq_utm  = eq_X_NGA2_all[i_data2keep,:]
sta_utm = sta_X_NGA2_all[i_data2keep,:]
mpt_utm = mpt_X_NGA2_all[i_data2keep,:]
#earthquake source depth
eq_z = eq_z_NGA2_all[i_data2keep]

#indices for unique earthquakes and stations
_, eq_idx, eq_inv   = np.unique(eqid_array, return_index=True, return_inverse=True)
_, sta_idx, sta_inv = np.unique(ssn_array,  return_index=True, return_inverse=True)
n_eq_orig  = len(eq_idx)
n_sta_orig = len(sta_idx)

# NGAWest2 dataframe
# ----   ----   ----   ----   ----
data_full = {'rsn':rsn_array, 'eqid':eqid_array, 'ssn':ssn_array,
             'mag':mag_array, 'Rrup':rrup_array, 'Vs30': vs30_array, 'year': year_array,
             'eqLat':eq_latlon[:,0], 'eqLon':eq_latlon[:,1], 'staLat':sta_latlon[:,0], 'staLon':sta_latlon[:,1], 'mptLat':mpt_latlon[:,0], 'mptLon':mpt_latlon[:,1],
             'UTMzone':utm_zone,
             'eqX':eq_utm[:,0], 'eqY':eq_utm[:,1], 'eqZ':eq_z, 'staX':sta_utm[:,0], 'staY':sta_utm[:,1], 'mptX':mpt_utm[:,0], 'mptY':mpt_utm[:,1]}

df_flatfile_full = pd.DataFrame(data_full)

# colocate stations
# ----   ----   ----   ----   ----
#update ssn for colocated stations
df_flatfile_full = pylib_catalog.ColocatePt(df_flatfile_full, 'ssn', ['staX','staY'], thres_dist=thres_dist)

#keep single record from each event after collocation
# ----   ----   ----   ----   ----
i_unq_eq_sta       = np.unique(df_flatfile_full[['eqid','ssn']].values, return_index=True, axis=0)[1]
df_flatfile_full   = df_flatfile_full.iloc[i_unq_eq_sta, :].sort_index()

_, eq_idx, eq_inv   = np.unique(df_flatfile_full.loc[:,'eqid'], axis=0, return_index=True, return_inverse=True)
_, sta_idx, sta_inv = np.unique(df_flatfile_full.loc[:,'ssn'],  axis=0, return_index=True, return_inverse=True)
n_eq  = len(eq_idx)
n_sta = len(sta_idx)

# average gm parameters
# ----   ----   ----   ----   ----
df_flatfile_full = pylib_catalog.IndexAvgColumns(df_flatfile_full, 'eqid', ['mag','eqLat','eqLon','eqX','eqY','eqZ'])
df_flatfile_full = pylib_catalog.IndexAvgColumns(df_flatfile_full, 'ssn',  ['Vs30','staLat','staLon','staX','staY'])

# create event and station dataframes
# ----   ----   ----   ----   ----
#event dataframe
df_flatfile_event   = df_flatfile_full.iloc[eq_idx,:][['eqid','mag','year','eqLat','eqLon','UTMzone','eqX','eqY','eqZ']].reset_index(drop=True)
#station dataframe
df_flatfile_station = df_flatfile_full.iloc[sta_idx,:][['ssn','Vs30','staLat','staLon','UTMzone','staX','staY']].reset_index(drop=True)



# %% Save data
# ======================================
# create output directories
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
if not os.path.isdir(dir_fig): pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True)

#save processed dataframes
fname_flatfile_full= '%s%s'%(dir_out, fname_flatfile)
df_flatfile_full.to_csv(fname_flatfile_full    + '.csv',         index=False)
df_flatfile_event.to_csv(fname_flatfile_full   + '_event.csv',   index=False)
df_flatfile_station.to_csv(fname_flatfile_full + '_station.csv', index=False)

# create figures
# ----   ----   ----   ----   ----
# Mag-Dist distribution
fname_fig = 'M-R_dist'
#create figure   
fig, ax = plt.subplots(figsize = (10,9))
pl1 = ax.scatter(df_flatfile_full.Rrup, df_flatfile_full.mag, label='NGAWest2 CA')
#edit figure properties
ax.set_xlabel(r'Distance ($km$)', fontsize=30)
ax.set_ylabel(r'Magnitude', fontsize=30)
ax.grid(which='both')
ax.set_xscale('log')
# ax.set_xlim([0.1, 2000])
ax.set_ylim([2,   8])
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

# Mag-Year distribution
fname_fig = 'M-date_dist'
#create figure   
fig, ax = plt.subplots(figsize = (10,9))
pl1 = ax.scatter(df_flatfile_event['year'].values, df_flatfile_event['mag'].values, label='NGAWest2 CA')
#edit figure properties
ax.set_xlabel(r'time ($year$)', fontsize=30)
ax.set_ylabel(r'Magnitude', fontsize=30)
ax.grid(which='both')
# ax.set_xscale('log')
ax.set_xlim([1965, 2025])
ax.set_ylim([2,   8])
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
ax.plot(df_flatfile_event['eqLon'].values,    df_flatfile_event['eqLat'].values,    '*', transform = data_crs, markersize = 10, zorder=13, label='Events')
ax.plot(df_flatfile_station['staLon'].values, df_flatfile_station['staLat'].values, 'o', transform = data_crs, markersize = 6,  zorder=12, label='Stations')
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
print(f'NGAWest2:')
print(f'\tnumber of rec: %.i'%len(df_flatfile_full))
print(f'\tnumber of rec (R<200km): %.i'%np.sum(df_flatfile_full.Rrup<=200))
print(f'\tnumber of rec (R<300km): %.i'%np.sum(df_flatfile_full.Rrup<=300))
print(f'\tnumber of eq: %.i'%len(df_flatfile_event))
print(f'\tnumber of sta: %.i'%len(df_flatfile_station))
print(f'\tcoverage: %.i to %i'%(df_flatfile_full.year.min(), df_flatfile_full.year.max()))

#write out summary
# ----   ----   ----   ----   ----
f = open(dir_out + 'summary_data' + '.txt', 'w')
f.write(f'NGAWest2:\n')
f.write(f'\tnumber of rec: %.i\n'%len(df_flatfile_full))
f.write(f'\tnumber of rec (R<200km): %.i\n'%np.sum(df_flatfile_full.Rrup<=200))
f.write(f'\tnumber of rec (R<300km): %.i\n'%np.sum(df_flatfile_full.Rrup<=300))
f.write(f'\tnumber of eq: %.i\n'%len(df_flatfile_event))
f.write(f'\tnumber of sta: %.i\n'%len(df_flatfile_station))
f.write(f'\tcoverage: %.i to %i\n'%(df_flatfile_full.year.min(), df_flatfile_full.year.max()))
f.close()

