#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:00:32 2020

@author: glavrent
"""
# %% Required Packages
# ======================================
#load variables
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
#plottign libraries
from matplotlib import pyplot as plt
#user-derfined functions
sys.path.insert(0,'../../Python_lib/plotting')
import pylib_contour_plots as pycplt

# Define Input Data
# ---------------------------
fname_flatfile      = '../../../Data/Verification/preprocessing/flatfiles/CA_NV_2011-2021Lite/CatalogNewRecordsLite_2011-2021_CA_NV.csv'
fname_cellinfo      = '../../../Data/Verification/preprocessing/cell_distances/CatalogNGAWest3CALite_cellinfo.csv'
fname_celldistfile  = '../../../Data/Verification/preprocessing/cell_distances/CatalogNGAWest3CALite_distancematrix.csv'

#grid limits and size
coeff_latlon_win = np.array([[32, -125],[42.5, -114]])

#log scale for number of paths
flag_logscl = True

#output directory
dir_out = '../../../Data/Verification/preprocessing/cell_distances/figures/'

# Load Data
# ---------------------------
df_flatfile = pd.read_csv(fname_flatfile)
df_cellinfo = pd.read_csv(fname_cellinfo)
#cell distance file
df_celldata = pd.read_csv(fname_celldistfile, index_col=0).reindex(df_flatfile.rsn)

# Process Data
# ---------------------------
#coordinates and projection system
# projection system
utm_zone = np.unique(df_flatfile.UTMzone)[0] #utm zone
utmProj = pyproj.Proj("+proj=utm +zone="+utm_zone+", +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

#cell edge coordinates
cell_edge_latlon = []
for cell_edge in [['q5X','q5Y'], ['q6X','q6Y'], ['q8X','q8Y'], 
                  ['q7X','q7Y'], ['q5X','q5Y']]:
    
    cell_edge_latlon.append( np.fliplr(np.array([utmProj(c_xy[0]*1000, c_xy[1]*1000, inverse=True) for c_xy in 
                                                 df_cellinfo.loc[:,cell_edge].values])) )                       
cell_edge_latlon = np.hstack(cell_edge_latlon)

#cell mid-coordinates
cell_latlon = np.fliplr(np.array([utmProj(c_xy[0]*1000, c_xy[1]*1000, inverse=True) for c_xy in 
                                  df_cellinfo.loc[:,['mptX','mptY']].values])) 

#earthquake and station ids
eq_id_train  = df_flatfile['eqid'].values.astype(int)
sta_id_train = df_flatfile['ssn'].values.astype(int)
eq_id, eq_idx_inv   = np.unique(eq_id_train,  return_index=True)
sta_id, sta_idx_inv = np.unique(sta_id_train, return_index=True)

#earthquake and station coordinates
eq_latlon_train     = df_flatfile[['eqLat', 'eqLon']].values
stat_latlon_train   = df_flatfile[['staLat', 'staLon']].values

#unique earthquake and station coordinates
eq_latlon   = eq_latlon_train[eq_idx_inv,:]
stat_latlon = stat_latlon_train[sta_idx_inv,:]

#cell names
cell_i      = [bool(re.match('^c\\..*$',c_n)) for c_n in df_celldata.columns.values] #indices for cell columns
cell_names  = df_celldata.columns.values[cell_i]

#cell-distance matrix with all cells
cell_dist    = df_celldata[cell_names]
cell_n_paths = (cell_dist > 0).sum()

# Create cell figures
# ---------------------------
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)

# if flag_pub:
#     # mpl.rcParams['font.family'] = 'Avenir'
#     plt.rcParams['axes.linewidth'] = 2


# Plot cell paths
fname_fig = 'cA_paths'
fig, ax, data_crs, gl = pycplt.PlotMap()
#plot earthquake and station locations
ax.plot(eq_latlon[:,1],   eq_latlon[:,0],   '*', transform = data_crs, markersize = 10, zorder=13,  label='Events')
ax.plot(stat_latlon[:,1], stat_latlon[:,0], 'o', transform = data_crs, markersize = 6,  zorder=12, label='Stations')
# ax.plot(eq_latlon[:,1],   eq_latlon[:,0],    '^', transform = data_crs, color = 'black', markersize = 10, zorder=13, label='Earthquake')
# ax.plot(stat_latlon[:,1], stat_latlon[:,0],  'o', transform = data_crs, color = 'black', markersize = 3,  zorder=12, label='Station')
#plot earthquake-station paths
for rec in df_flatfile[['eqLat','eqLon','staLat','staLon']].iterrows():
    ax.plot(rec[1][['eqLon','staLon']], rec[1][['eqLat','staLat']], transform = data_crs, color = 'gray', linewidth=0.05, zorder=10, alpha=0.2)
#plot cells
for ce_xy in cell_edge_latlon:
    ax.plot(ce_xy[[1,3,5,7,9]],ce_xy[[0,2,4,6,8]],color='gray', transform = data_crs)
#figure limits
ax.set_xlim( coeff_latlon_win[:,1] )
ax.set_ylim( coeff_latlon_win[:,0] )
#edit figure properties
#grid lines
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 25}
gl.ylabel_style = {'size': 25}
#add legend
ax.legend(fontsize=25, loc='lower left')
#apply tight layout
fig.show()
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png')

# Plot cell paths
fname_fig = 'cA_num_paths'
cbar_label = 'Number of paths'
data2plot = np.vstack([cell_latlon.T, cell_n_paths.values]).T
#color limits
cmin = 0
cmax = 2000
#log scale options
if flag_logscl:
    # data2plot[:,2] = np.maximum(data2plot[:,2], 1)
    cmin = np.log(1)
    cmax = np.log(cmax)
#create figure
fig, ax, cbar, data_crs, gl = pycplt.PlotCellsCAMap(data2plot, cmin=cmin,  cmax=cmax, log_cbar = flag_logscl,
                                                    frmt_clb = '%.0f',  cmap='OrRd')
#plot cells
for ce_xy in cell_edge_latlon:
    ax.plot(ce_xy[[1,3,5,7]],ce_xy[[0,2,4,6]],color='gray', transform = data_crs)
#figure limits
ax.set_xlim( coeff_latlon_win[:,1] )
ax.set_ylim( coeff_latlon_win[:,0] )
#edit figure properties
#grid lines
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 25}
gl.ylabel_style = {'size': 25}
#update colorbar 
cbar.set_label(cbar_label, size=30)
cbar.ax.tick_params(labelsize=25)
#apply tight layout
fig.show()
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png')

