#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:58:20 2021

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
import warnings
#arithmetic libraries
import numpy as np
import numpy.linalg
import scipy as sp
import scipy.linalg
import pandas as pd
#geometric libraries
from shapely.geometry import Point as shp_pt, Polygon as shp_poly
#geographic libraries
import pyproj
#plottign libraries
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
#user-derfined functions
sys.path.insert(0,'../../Python_lib/catalog')
sys.path.insert(0,'../../Python_lib/plotting')
import pylib_catalog as pylib_catalog
import pylib_contour_plots as pylib_cplt

# %% Define Input Data
# ======================================
# threshold distances
# distance for collocated stations
thres_dist = 0.01

#flatfiles
dir_flatfiles   = '../../../Data/Verification/preprocessing/flatfiles/'
# fname_flatfiles = ['NGAWest2_CA/CatalogNGAWest2CA_ASK14.csv',
#                    'CA_NV_2011-2020/CatalogNewRecords_2011-2021_CA_NV.csv']
fname_flatfiles = ['NGAWest2_CA/CatalogNGAWest2CA_ASK14.csv',
                   'CA_NV_2011-2021Lite/CatalogNewRecordsLite_2011-2021_CA_NV.csv']

#flatfile file
# fname_flatfile_out = 'CatalogNGAWest3CA'
fname_flatfile_out = 'CatalogNGAWest3CALite'

#output directory
dir_out = '../../../Data/Verification/preprocessing/flatfiles/merged/'
dir_fig = dir_out + 'figures/'
flag_col = True

#North and South CA polygons
sreg_NCA_latlon = np.array([[42,-124.5],[35,-124.5],[35,-120],[36,-120],[36,-119.5],[37,-119.5],[37,-119],[38,-119],[38,-114],[42,-114],[42,-124.5]])
sreg_SCA_latlon = np.array([[31,-124.5],[35,-124.5],[35,-120],[36,-120],[36,-119.5],[37,-119.5],[37,-119],[38,-119],[38,-114],[31,-114],[31,-124.5]])

#earthquake station info columns for averaging
eq_col_idx   = 'eqid'
sta_col_idx  = 'ssn'
eq_col_info  = ['mag', 'eqX', 'eqY', 'eqZ']
sta_col_info = ['Vs30', 'staX', 'staY']

# %% Load Data
# ======================================
#load individual faltfiles
df_flatfile = [pd.read_csv(dir_flatfiles + fn_fltfile) for fn_fltfile in fname_flatfiles]

# %% Process Data
# ======================================
#compute number of events and stations per dataframe
n_eq_df_orig  = [len(np.unique(df_fltf.eqid)) for df_fltf in df_flatfile]
n_sta_df_orig = [len(np.unique(df_fltf.ssn))  for df_fltf in df_flatfile]

# Merge Data-sets
#----   ----   ----   ----   ----
#define data-set id, copy original rsn, eqid, and ssn 
for ds_id in range(len(df_flatfile)):
    df_flatfile[ds_id].loc[:,'dsid']     =  ds_id
    #copy original columns
    df_flatfile[ds_id].loc[:,'rsn_orig']  = df_flatfile[ds_id].loc[:,'rsn']
    df_flatfile[ds_id].loc[:,'eqid_orig'] = df_flatfile[ds_id].loc[:,'eqid']
    df_flatfile[ds_id].loc[:,'ssn_orig']  = df_flatfile[ds_id].loc[:,'ssn']

#merge datasets
df_flatfile = pd.concat(df_flatfile).reset_index(drop=True)

#define projection system 
assert(len(np.unique(df_flatfile.UTMzone))==1),'Error. Multiple UTM zones defined.'
utm_zone = df_flatfile.UTMzone[0]
utmProj = pyproj.Proj("+proj=utm +zone="+utm_zone+", +ellps=WGS84 +datum=WGS84 +units=km +no_defs")

#reset rsn
df_flatfile.rsn = np.arange(len(df_flatfile))+1

# Original info data-frame
#----   ----   ----   ----   ----
df_flatfile_orig = df_flatfile.copy()

# New Earthquake IDs
#----   ----   ----   ----   ----
#define new earthquake id
_, eq_inv = np.unique( df_flatfile[['dsid','eqid']], axis=0, return_inverse=True ) 
df_flatfile.eqid = eq_inv + 1

#unique eqids
eqid_array = np.unique(df_flatfile.eqid)

#total number of events
n_eq_orig = len(eqid_array)
assert(n_eq_orig == np.sum(n_eq_df_orig)),'Error. Total number of events is not equal to sum of number of events from individual data-sets'

# New Station IDs
#----   ----   ----   ----   ----
#define new earthquake id, initially different stations for separate data-sets
_, sta_idx, sta_inv = np.unique(df_flatfile[['dsid','ssn']], axis=0, return_index=True, return_inverse=True )
df_flatfile.ssn = sta_inv + 1

#total number of statios before collocation
n_sta_orig = len(np.unique(df_flatfile.ssn))
assert(n_sta_orig == np.sum(n_sta_df_orig)),'Error. Total number of stations, before collocation, is not equal to sum of number of station from individual data-sets'

# Collocate Stations
#----   ----   ----   ----   ----
#update ssn for colocated stations
df_flatfile = pylib_catalog.ColocatePt(df_flatfile, 'ssn', ['staX','staY'], thres_dist=thres_dist)

#keep single record from each event
i_unq_eq_sta = np.unique(df_flatfile[['eqid','ssn']].values, return_index=True, axis=0)[1]
df_flatfile  = df_flatfile.iloc[i_unq_eq_sta, :].sort_index()

# Average GM Parameters
# ----   ----   ----   ----   ----
df_flatfile = pylib_catalog.IndexAvgColumns(df_flatfile, 'eqid', ['mag','eqX','eqY','eqZ'])
df_flatfile = pylib_catalog.IndexAvgColumns(df_flatfile, 'ssn',  ['Vs30','staX','staY','staElev'])

#verify no station has multiple records at the same event
for eqid in eqid_array:
    sta_eq = df_flatfile.loc[df_flatfile.eqid == eqid,'ssn'].values
    assert(len(sta_eq) == len(np.unique(sta_eq))),'Error. Event %i has multiple collocated stations'%eqid

#recalculated lat/lon coordinates
_, eq_idx,  eq_inv  = np.unique(df_flatfile.loc[:,'eqid'], axis=0, return_index=True, return_inverse=True)
_, sta_idx, sta_inv = np.unique(df_flatfile.loc[:,'ssn'],  axis=0, return_index=True, return_inverse=True)
n_eq  = len(eq_idx)
n_sta = len(sta_idx)

eq_latlon  = np.flip([utmProj(e.eqX, e.eqY, inverse=True)   for _, e in df_flatfile.iloc[eq_idx,:].iterrows()],  axis=1)  
sta_latlon = np.flip([utmProj(s.staX, s.staY, inverse=True) for _, s in df_flatfile.iloc[sta_idx,:].iterrows()], axis=1)
df_flatfile.loc[:,['eqLat','eqLon']]   = eq_latlon[eq_inv,:]
df_flatfile.loc[:,['staLat','staLon']] = sta_latlon[sta_inv,:]

# Midpoint Coordinates
# ----   ----   ----   ----   ----
df_flatfile.loc[:,['mptX','mptY']]     = (df_flatfile.loc[:,['eqX','eqY']].values + df_flatfile.loc[:,['staX','staY']].values) / 2
df_flatfile.loc[:,['mptLat','mptLon']] = np.flip( np.array([utmProj(pt.mptX, pt.mptY, inverse=True) for _, pt in  df_flatfile.iterrows()]), axis=1 )

#recompute rupture distance
rrup_array = np.sqrt( np.linalg.norm(df_flatfile[['eqX','eqY']].values-df_flatfile[['staX','staY']].values, axis=1)**2 + 
                     df_flatfile['eqZ'].values**2 )
df_flatfile.Rrup = rrup_array

# Difference between original and process catalog
#----   ----   ----   ----   ----
eq_corr_diff  = np.linalg.norm(df_flatfile[['eqLat','eqLon']].values   - df_flatfile_orig[['eqLat','eqLon']].values,   axis=1)
sta_corr_diff = np.linalg.norm(df_flatfile[['staLat','staLon']].values - df_flatfile_orig[['staLat','staLon']].values, axis=1)
eq_loc_diff   = np.linalg.norm(df_flatfile[['eqX','eqY']].values       - df_flatfile_orig[['eqX','eqY']].values,       axis=1)
sta_loc_diff  = np.linalg.norm(df_flatfile[['staX','staY']].values     - df_flatfile_orig[['staX','staY']].values,     axis=1)
mag_diff      = np.abs(df_flatfile['mag'].values  - df_flatfile_orig['mag'].values)
rrup_diff     = np.abs(df_flatfile['Rrup'].values - df_flatfile_orig['Rrup'].values)
vs30_diff     = np.abs(df_flatfile['Vs30'].values - df_flatfile_orig['Vs30'].values)

#North South CA regions 
#----   ----   ----   ----   ----
#shapely polygons for Northern and Southern CA
sreg_NCA_X = np.array([utmProj(pt_lon, pt_lat) for pt_lat, pt_lon in zip(sreg_NCA_latlon[:,0], sreg_NCA_latlon[:,1])])
sreg_SCA_X = np.array([utmProj(pt_lon, pt_lat) for pt_lat, pt_lon in zip(sreg_SCA_latlon[:,0], sreg_SCA_latlon[:,1])])

#shapely polygons for Northern and Southern CA
sreg_NCA_poly = shp_poly(sreg_NCA_X)
sreg_SCA_poly = shp_poly(sreg_SCA_X)

#indices for earthquakes belonging to Northern and Southern CA
i_sregNCA = np.array([ shp_pt(eq_x).within(sreg_NCA_poly) for _, eq_x in df_flatfile[['eqX','eqY']].iterrows() ])
i_sregSCA = np.array([ shp_pt(eq_x).within(sreg_SCA_poly) for _, eq_x in df_flatfile[['eqX','eqY']].iterrows() ])
assert( (i_sregNCA + i_sregSCA <= 1).all() ),'Error. Overlapping sub-regions'
       
#add region info to catalog
df_flatfile.loc[:,'sreg'] = 0
df_flatfile.loc[i_sregNCA,'sreg'] = 1
df_flatfile.loc[i_sregSCA,'sreg'] = 2

# Clean Records
#----   ----   ----   ----   ----
#remove records with unknown earthquake and source parameters
i_val_rec = ~np.isnan(df_flatfile[eq_col_info+sta_col_info]).any(axis=1)
df_flatfile = df_flatfile.loc[i_val_rec,:]
     
# %% Save data
# ======================================
#create output directories
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
if not os.path.isdir(dir_fig): pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True)

#rearange columns
df_flatfile = df_flatfile[['rsn', 'eqid', 'ssn', 'dsid', 'rsn_orig', 'eqid_orig', 'ssn_orig',
                           'mag', 'Rrup', 'Vs30', 'year',
                           'eqLat', 'eqLon', 'staLat', 'staLon', 'mptLat', 'mptLon', 
                           'UTMzone', 'eqX', 'eqY', 'eqZ', 'staX', 'staY', 'mptX', 'mptY', 'sreg']]

# #create individual North and South CA catalogs
# df_flatfileN = df_flatfile.loc[df_flatfile.sreg==1]
# df_flatfileS = df_flatfile.loc[df_flatfile.sreg==2]

#save processed dataframe
fullname_flatfile_out = '%s%s.csv'%(dir_out, fname_flatfile_out)
df_flatfile.to_csv(fullname_flatfile_out, index=False)
# fullname_flatfile_out = '%s%sNCA.csv'%(dir_out, fname_flatfile_out)
# df_flatfileS.to_csv(fullname_flatfile_out, index=False)
# fullname_flatfile_out = '%s%sSCA.csv'%(dir_out, fname_flatfile_out)
# df_flatfileN.to_csv(fullname_flatfile_out, index=False)

# Print data info
# ---------------------------
print(f'NGAWest3:')
print(f'\tGeneral Info:')
print(f'\t\tnumber of rec: %.i'%len(df_flatfile))
print(f'\t\tnumber of rec (R<200km): %.i'%np.sum(df_flatfile.Rrup<=200))
print(f'\t\tnumber of rec (R<300km): %.i'%np.sum(df_flatfile.Rrup<=300))
print(f'\t\tnumber of eq: %.i'%len(df_flatfile.eqid.unique()))
print(f'\t\tnumber of sta: %.i'%len(df_flatfile.ssn.unique()))
print(f'\t\tcoverage: %.i to %i'%(df_flatfile.year.min(), df_flatfile.year.max()))
print(f'\tMerging Info:')
print(f'\t\tnumber of merged stations: %.i'%(n_sta_orig-n_sta))
print(f'\t\tmax EQ latlon difference: %.2f'%eq_corr_diff[i_val_rec].max())
print(f'\t\tmax EQ UTM difference: %.2f'%eq_loc_diff[i_val_rec].max())
print(f'\t\tmax Sta latlon difference: %.2f'%sta_corr_diff[i_val_rec].max())
print(f'\t\tmax Sta UTM difference: %.2f'%sta_loc_diff[i_val_rec].max())
print(f'\t\tmax M difference: %.2f'%mag_diff[i_val_rec].max())
print(f'\t\tmax Rrup difference: %.2fkm'%rrup_diff[i_val_rec].max())
print(f'\t\tmax Vs30 difference: %.2fm/sec'%vs30_diff[i_val_rec].max())
print(f'\t\tnumber of invalid records: %.i'%np.sum(~i_val_rec))

#write out summary
# ----   ----   ----   ----   ----
f = open(dir_out + 'summary_data' + '.txt', 'w')
f.write(f'NGAWest3:\n')
f.write(f'\tGeneral Info:\n')
f.write(f'\t\tnumber of rec: %.i\n'%len(df_flatfile))
f.write(f'\t\tnumber of rec (R<200km): %.i\n'%np.sum(df_flatfile.Rrup<=200))
f.write(f'\t\tnumber of rec (R<300km): %.i\n'%np.sum(df_flatfile.Rrup<=300))
f.write(f'\t\tnumber of eq: %.i\n'%len(df_flatfile.eqid.unique()))
f.write(f'\t\tnumber of sta: %.i\n'%len(df_flatfile.ssn.unique()))
f.write(f'\t\tcoverage: %.i to %i\n'%(df_flatfile.year.min(), df_flatfile.year.max()))
f.write(f'\tMerging Info:\n')
f.write(f'\t\tnumber of merged stations: %.i\n'%(n_sta_orig-n_sta))
f.write(f'\t\tmax EQ latlon difference: %.2f\n'%eq_corr_diff[i_val_rec].max())
f.write(f'\t\tmax EQ UTM difference: %.2f\n'%eq_loc_diff[i_val_rec].max())
f.write(f'\t\tmax Sta latlon difference: %.2f\n'%sta_corr_diff[i_val_rec].max())
f.write(f'\t\tmax Sta UTM difference: %.2f\n'%sta_loc_diff[i_val_rec].max())
f.write(f'\t\tmax M difference: %.2f\n'%mag_diff[i_val_rec].max())
f.write(f'\t\tmax Rrup difference: %.2fkm\n'%rrup_diff[i_val_rec].max())
f.write(f'\t\tmax Vs30 difference: %.2fm/sec\n'%vs30_diff[i_val_rec].max())
f.write(f'\t\tnumber of invalid records: %.i\n'%np.sum(~i_val_rec))

# %% Plotting
# ======================================

df_flt = df_flatfile.copy().reset_index(drop=True)

# Mag-Dist distribution
fname_fig = 'M-R_dist_log'
#create figure   
fig, ax = plt.subplots(figsize = (10,9))
pl1 = ax.scatter(df_flt.Rrup, df_flt.mag)
#edit figure properties
ax.set_xlabel(r'Distance ($km$)', fontsize=30)
ax.set_ylabel(r'Magnitude', fontsize=30)
ax.grid(which='both')
ax.set_xscale('log')
ax.set_xlim([0.1, 2000])
ax.set_ylim([2,   8])
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', right='on')
fig.tight_layout()
#save figure
fig.savefig( dir_fig + fname_fig + '.png' )

# Mag-Dist distribution
fname_fig = 'M-R_dist_linear'
#create figure   
fig, ax = plt.subplots(figsize = (10,9))
pl1 = ax.scatter(df_flt.Rrup, df_flt.mag)
#edit figure properties
ax.set_xlabel(r'Distance ($km$)', fontsize=30)
ax.set_ylabel(r'Magnitude', fontsize=30)
ax.grid(which='both')
ax.set_xlim([0.1, 500])
ax.set_ylim([2,   8])
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', right='on')
fig.tight_layout()
#save figure
fig.savefig( dir_fig + fname_fig + '.png' )

# Source depht distribution
fname_fig = 'eqZ_dist'
#create figure   
fig, ax = plt.subplots(figsize = (10,9))
pl1 = ax.hist(-df_flt.eqZ)
#edit figure properties
ax.set_xlabel(r'Source depth (km)', fontsize=30)
ax.set_ylabel(r'Count', fontsize=30)
ax.grid(which='both')
# ax.set_xscale('log')
# ax.set_xlim([0.1, 2000])
# ax.set_ylim([2,   8])
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', right='on')
fig.tight_layout()
#save figure
fig.savefig( dir_fig + fname_fig + '.png' )

# eq and sta location
#get unique earthquake indices
_, eq_idx  = np.unique(df_flt['eqid'], axis=0, return_index=True )
#get unique station indices
_, sta_idx = np.unique(df_flt['ssn'], axis=0, return_index=True)

# eq and sta location
fname_fig = 'eq_sta_locations'
fig, ax, data_crs, gl = pylib_cplt.PlotMap(flag_grid=True)
#plot earthquake and station locations
ax.plot(df_flt.loc[eq_idx,'eqLon'].values,   df_flt.loc[eq_idx,'eqLat'].values,   
        '*', transform = data_crs, markersize = 10, zorder=13, label='Events')
ax.plot(df_flt.loc[sta_idx,'staLon'].values, df_flt.loc[sta_idx,'staLat'].values, 
        'o', transform = data_crs, markersize = 6,  zorder=13, label='Stations')
#edit figure properties
gl.xlabel_style = {'size': 25}
gl.ylabel_style = {'size': 25}
# gl.xlocator = mticker.FixedLocator([-124, -122, -120, -118, -116, -114])
# gl.ylocator = mticker.FixedLocator([32, 34, 36, 38, 40])
ax.legend(fontsize=25, loc='lower left')
# ax.set_xlim([-125, -113.5])
# ax.set_ylim([30.5, 42.5])
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )
