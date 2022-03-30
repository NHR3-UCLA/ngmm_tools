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
sys.path.insert(0,'../../Python_lib/catalog')
sys.path.insert(0,'../../Python_lib/plotting')
import pylib_catalog as pylib_catalog
import pylib_contour_plots as pylib_cplt

# %% Define Input Data
# ======================================
#thresholds 
#min sampling rate
thres_dt   = 0.025 
#collocated stations
thres_dist = 0.01
#maximum depth
eq_mag_depth = 25
#distance range
# rrup_thres = 300 
rrup_thres = 400
#year range
year_min = 2011
year_max = 2021
# year_max = 2013
# projection system
utm_zone = '11S'
#input flatfiles
fname_flatfile_newrec = '../../../Data/Verification/preprocessing/flatfiles/CA_NV_2011-2021_raw/Catalog_California_2011-2021.ver02.csv'
fname_sta_vs30        = '../../../Raw_files/NGAWest3/IRIS/Catalog_California_2011-2021.ver02_station_Vs30PW.csv'

#flatfile file
fname_flatfile = 'CatalogNewRecords_%.i-%.i_CA_NV'%(year_min, year_max )

#output directory
dir_out = '../../../Data/Verification/preprocessing/flatfiles/CA_NV_%.i-%.i/'%(year_min, year_max)

dir_fig = dir_out + 'figures/'

#latlon window
win_latlon = np.array([[30, 43],[-125, -110]])
#win_latlon = np.array([[32, 43],[-125, -114]])

# %% Load Data
# ======================================
#read event and station info
df_flatfile_newrec  = pd.read_csv(fname_flatfile_newrec)

#read vs30 info
if fname_sta_vs30:
    df_sta_vs30 = pd.read_csv(fname_sta_vs30)
else:
    df_sta_vs30 = None
    
# %% Process Data
# ======================================
# projection system
# ----   ----   ----   ----   ----
utmProj = pyproj.Proj("+proj=utm +zone="+utm_zone+", +ellps=WGS84 +datum=WGS84 +units=km +no_defs")

# rename columns
# ----   ----   ----   ----   ----
df_flatfile_newrec.rename(columns={'EventID':'eventid', 'Time':'time', 'Latitude':'eqLat', 'Longitude':'eqLon', 
                                   'Author':'author', 'Catalog':'cat', 'Contributor':'contributor', 'ContributorID':'contributor_id',
                                   'Magnitude':'mag', 'MagType':'mag_type', 'MagAuthor':'mag_author',
                                   'EventLocationName':'eq_loc', 'ESN':'esn', 'Double.event':'double_event',
                                   'event.ESN':'esn_y', 'event.EventID':'eventid', 'Network':'network', 'Station':'station', 
                                   'Latitude':'staLat', 'Longitude':'staLon', 'Elevation':'staElev'},
                          inplace=True)
#year of event
df_flatfile_newrec['year'] = pd.DatetimeIndex(df_flatfile_newrec['time']).year

#station vs30
if fname_sta_vs30:
    df_sta_vs30.rename(columns={'Network':'network', 'Station':'station', 'Vs30_lnSD': 'Vs30sig'}, inplace=True)
    _, df_sta_vs30.loc[:,'sta_id'] = np.unique(df_sta_vs30.station, return_inverse=True)
    _, df_sta_vs30.loc[:,'net_id'] = np.unique(df_sta_vs30.network, return_inverse=True)
    assert(len(df_sta_vs30) == len(np.unique(df_sta_vs30[['net_id','sta_id']], axis=1))),'Error. Non-unique network stations'

# cleaning files
# ----   ----   ----   ----   ----
#set -999 to nan
df_flatfile_newrec.replace(-999, np.nan, inplace=True)
#remove data with unknown mag 
df_flatfile_newrec = df_flatfile_newrec[ ~np.isnan(df_flatfile_newrec['mag']) ]
#remove data with unknown coordinates
df_flatfile_newrec = df_flatfile_newrec[ ~np.isnan(df_flatfile_newrec[['eqLat',  'eqLon']]).any(axis=1) ]
df_flatfile_newrec = df_flatfile_newrec[ ~np.isnan(df_flatfile_newrec[['staLat', 'staLon']]).any(axis=1) ]

# keep only data in spatio-temporal window
# ----   ----   ----   ----   ----
#earthquakes
i_space_win_eq = np.all(np.array([df_flatfile_newrec.eqLat >= win_latlon[0,0], 
                                  df_flatfile_newrec.eqLat < win_latlon[0,1], 
                                  df_flatfile_newrec.eqLon >= win_latlon[1,0], 
                                  df_flatfile_newrec.eqLon < win_latlon[1,1]]),axis=0)
#stations
i_space_win_sta = np.all(np.array([df_flatfile_newrec.staLat >= win_latlon[0,0], 
                                   df_flatfile_newrec.staLat < win_latlon[0,1], 
                                   df_flatfile_newrec.staLon >= win_latlon[1,0], 
                                   df_flatfile_newrec.staLon < win_latlon[1,1]]),axis=0)
#depth limit
i_eq_depth = df_flatfile_newrec.eqDepth <= eq_mag_depth
#time
i_time_win = np.logical_and(df_flatfile_newrec.year >= year_min, df_flatfile_newrec.year <= year_max)

#records to keep
i_win  = np.all(np.array([i_space_win_eq, i_space_win_sta, i_eq_depth, i_time_win]),axis=0)
df_flatfile_newrec = df_flatfile_newrec[i_win] 

# Vs30
# ----   ----   ----   ----   ----
#Wills 2015 model for CA
# Wills15Vs30 = pylib_W15_Vs30.Willis15Vs30CA()
# df_flatfile_newrec.loc[:,'Vs30'] = Wills15Vs30.lookup(np.fliplr(df_flatfile_newrec[['staLat', 'staLon']]))[0]
# df_flatfile_newrec.loc[df_flatfile_newrec.loc[:,'Vs30'] < 50,'Vs30'] = np.nan
#geology based Vs30
if not df_sta_vs30 is None:
    #Vs30 estimates from geology and topography (Pengfei, personal communication)
    df_flatfile_newrec = pd.merge(df_flatfile_newrec, df_sta_vs30[['station','network','Vs30','Vs30sig']], on=['station','network'] )
else:
    #Unavailable Vs30
    df_flatfile_newrec.loc[:,['Vs30','Vs30sig']] = np.nan

# define earthquake, station and network ids
# ----   ----   ----   ----   ----
#original earthquake and station ids
df_flatfile_newrec.loc[:,'eventid'] = df_flatfile_newrec['eventid'].values.astype(int)
df_flatfile_newrec.loc[:,'staid']   = np.unique(df_flatfile_newrec['station'], return_inverse=True)[1] + 1
df_flatfile_newrec.loc[:,'netid']   = np.unique(df_flatfile_newrec['network'], return_inverse=True)[1] + 1
#keep single record from each event
i_unq_eq_sta       = np.unique(df_flatfile_newrec[['eventid','staid','netid']].values, return_index=True, axis=0)[1]
df_flatfile_newrec = df_flatfile_newrec.iloc[i_unq_eq_sta, :]

# define rsn eqid and ssn
# ----   ----   ----   ----   ----
#reset indices
df_flatfile_newrec.reset_index(drop=True, inplace=True)
df_flatfile_newrec.rename_axis('rsn',     inplace=True)
#updated earthquake and station ids
_, eq_idx,  eq_inv  = np.unique(df_flatfile_newrec.loc[:,'eventid'],         axis=0, return_index=True, return_inverse=True)
_, sta_idx, sta_inv = np.unique(df_flatfile_newrec.loc[:,['netid','staid']], axis=0, return_index=True, return_inverse=True)
df_flatfile_newrec.loc[:,'eqid'] = eq_inv+1
df_flatfile_newrec.loc[:,'ssn']  = sta_inv+1
n_eq_orig  = len(eq_idx)
n_sta_orig = len(sta_idx)

# cartesian coordinates
# ----   ----   ----   ----   ----
#utm coordinates
eq_X  = np.array([utmProj(e.eqLon, e.eqLat)   for _, e in df_flatfile_newrec.iloc[eq_idx,:].iterrows()])  
sta_X = np.array([utmProj(s.staLon, s.staLat) for _, s in df_flatfile_newrec.iloc[sta_idx,:].iterrows()])
df_flatfile_newrec.loc[:,['eqX','eqY']]   = eq_X[eq_inv,:]
df_flatfile_newrec.loc[:,['staX','staY']] = sta_X[sta_inv,:]
#eq depth
df_flatfile_newrec.loc[:,'eqZ'] = np.minimum(-1*df_flatfile_newrec['eqDepth'].values, 0)
#utm zone
df_flatfile_newrec.loc[:,'UTMzone'] = utm_zone

# rupture distance
# ----   ----   ----   ----   ----
df_flatfile_newrec.loc[:,'Rrup'] = np.sqrt(np.linalg.norm(df_flatfile_newrec[['eqX','eqY']].values-df_flatfile_newrec[['staX','staY']].values, axis=1)**2 +
                                           df_flatfile_newrec['eqZ']**2)
#remove records based on rupture distance
i_rrup = df_flatfile_newrec['Rrup'] < rrup_thres
df_flatfile_newrec = df_flatfile_newrec.loc[i_rrup,:]

# colocate stations
# ----   ----   ----   ----   ----
#update ssn for colocated stations
df_flatfile_newrec = pylib_catalog.ColocatePt(df_flatfile_newrec, 'ssn', ['staX','staY'], thres_dist=thres_dist)

#keep single record from each event
i_unq_eq_sta       = np.unique(df_flatfile_newrec[['eqid','ssn']].values, return_index=True, axis=0)[1]
df_flatfile_newrec = df_flatfile_newrec.iloc[i_unq_eq_sta, :].sort_index()

# average gm parameters
# ----   ----   ----   ----   ----
df_flatfile_newrec = pylib_catalog.IndexAvgColumns(df_flatfile_newrec, 'eqid', ['mag','eqX','eqY','eqZ'])
df_flatfile_newrec = pylib_catalog.IndexAvgColumns(df_flatfile_newrec, 'ssn',  ['Vs30','staX','staY','staElev'])

#recalculated lat/lon coordinates
_, eq_idx,  eq_inv  = np.unique(df_flatfile_newrec.loc[:,'eqid'], axis=0, return_index=True, return_inverse=True)
_, sta_idx, sta_inv = np.unique(df_flatfile_newrec.loc[:,'ssn'],  axis=0, return_index=True, return_inverse=True)
n_eq  = len(eq_idx)
n_sta = len(sta_idx)

eq_latlon  = np.flip([utmProj(e.eqX, e.eqY, inverse=True)   for _, e in df_flatfile_newrec.iloc[eq_idx,:].iterrows()],  axis=1)  
sta_latlon = np.flip([utmProj(s.staX, s.staY, inverse=True) for _, s in df_flatfile_newrec.iloc[sta_idx,:].iterrows()], axis=1)
df_flatfile_newrec.loc[:,['eqLat','eqLon']]   = eq_latlon[eq_inv,:]
df_flatfile_newrec.loc[:,['staLat','staLon']] = sta_latlon[sta_inv,:]

# midpoint coordinates
# ----   ----   ----   ----   ----
df_flatfile_newrec.loc[:,['mptX','mptY']]     = (df_flatfile_newrec.loc[:,['eqX','eqY']].values + df_flatfile_newrec.loc[:,['staX','staY']].values) / 2
df_flatfile_newrec.loc[:,['mptLat','mptLon']] = np.flip( np.array([utmProj(pt.mptX, pt.mptY, inverse=True) for _, pt in  df_flatfile_newrec.iterrows()]), axis=1 )


#recalculate rupture distance after averaging
df_flatfile_newrec.loc[:,'Rrup'] = np.sqrt(np.linalg.norm(df_flatfile_newrec[['eqX','eqY']].values-df_flatfile_newrec[['staX','staY']].values, axis=1)**2 +
                                           df_flatfile_newrec['eqZ']**2)

# %% Save Data
# ======================================
# create output directories
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
if not os.path.isdir(dir_fig): pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True)

#full dataframe
df_flatfile_full = df_flatfile_newrec[['eqid','ssn','eventid','staid','netid','station','network',
                                       'mag','mag_type','mag_author','Rrup','Vs30','time','year',
                                       'eqLat','eqLon','staLat','staLon','mptLat','mptLon',
                                       'UTMzone','eqX','eqY','eqZ','staX','staY','staElev','mptX','mptY',
                                       'author','cat','contributor','contributor_id','eq_loc']]
#event dataframe
df_flatfile_event = df_flatfile_newrec.iloc[eq_idx,:][['eqid','eventid','mag','mag_type','mag_author','year',
                                                       'eqLat','eqLon','UTMzone','eqX','eqY','eqZ',
                                                       'author','cat','contributor','contributor_id','eq_loc']].reset_index(drop=True)

#station dataframe
df_flatfile_station = df_flatfile_newrec.iloc[sta_idx,:][['ssn','Vs30',
                                                          'staLat','staLon','UTMzone','staX','staY','staElev']].reset_index(drop=True)

# save dataframe
# ----   ----   ----   ----   ----
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
pl1 = ax.scatter(df_flatfile_full.Rrup, df_flatfile_full.mag, label='new records')
#edit figure properties
ax.set_xlabel(r'Distance ($km$)', fontsize=30)
ax.set_ylabel(r'Magnitude', fontsize=30)
ax.grid(which='both')
ax.set_xscale('log')
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

# Mag-Year distribution
fname_fig = 'M-date_dist'
#create figure   
fig, ax = plt.subplots(figsize = (10,9))
pl1 = ax.scatter(df_flatfile_event['year'].values, df_flatfile_event['mag'].values, label='new records')
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
ax.plot(df_flatfile_event['eqLon'].values,    df_flatfile_event['eqLat'].values,    '*', transform = data_crs, markersize = 10, zorder=12, label='Events')
ax.plot(df_flatfile_station['staLon'].values, df_flatfile_station['staLat'].values, 'o', transform = data_crs, markersize = 6,  zorder=13, label='Stations')
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

# print data info
# ----   ----   ----   ----   ----
print(r'New Records:')
print(f'\tnumber of rec: %.i'%len(df_flatfile_newrec))
print(f'\tnumber of rec (R<200km): %.i'%np.sum(df_flatfile_newrec.Rrup<=200))
print(f'\tnumber of rec (R<%.1f): %.i'%(rrup_thres, np.sum(df_flatfile_newrec.Rrup<=rrup_thres)))
print(f'\tnumber of eq: %.i'%n_eq)
print(f'\tnumber of sta: %.i'%n_sta)
print(f'\tmin magnitude: %.1f'%df_flatfile_newrec.mag.min())
print(f'\tmax magnitude: %.1f'%df_flatfile_newrec.mag.max())
print(f'\tcoverage: %.i to %i'%(df_flatfile_newrec.year.min(), df_flatfile_newrec.year.max()))

#write out summary
# ----   ----   ----   ----   ----
f = open(dir_out + 'summary_data' + '.txt', 'w')
f.write(f'New Records:\n')
f.write(f'\tnumber of rec: %.i\n'%len(df_flatfile_newrec))
f.write(f'\tnumber of rec (R<200km): %.i\n'%np.sum(df_flatfile_newrec.Rrup<=200))
f.write(f'\tnumber of rec (R<%.1f): %.i\n'%(rrup_thres, np.sum(df_flatfile_newrec.Rrup<=rrup_thres)))
f.write(f'\tnumber of eq: %.i\n'%n_eq)
f.write(f'\tnumber of sta: %.i\n'%n_sta)
f.write(f'\tmin magnitude: %.1f\n'%df_flatfile_newrec.mag.min())
f.write(f'\tmax magnitude: %.1f\n'%df_flatfile_newrec.mag.max())
f.write(f'\tcoverage: %.i to %i\n'%(df_flatfile_newrec.year.min(), df_flatfile_newrec.year.max()))
f.close()

