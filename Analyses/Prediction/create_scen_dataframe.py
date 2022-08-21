#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 17:26:17 2022

@author: glavrent
"""

#load variables
import os
import sys
import pathlib
#arithmetic libraries
import numpy as np
#statistics libraries
import pandas as pd
#geographic libraries
import pyproj
import geopy.distance

#user libraries
sys.path.insert(0,'../Python_lib/ground_motions')
from pylib_gmm_eas import BA18
ba18 = BA18()

# Define Problem
# ---------------------------
#structural period
freq = 5.0119

#earthquake scenario
mag   = 7.0
vs30  = 400
sof   = 'SS'
dip   = 90
z_tor = 0
#color bar limits
cbar_lim = [np.log(1e-8),np.log(.06)]

#earthquake coordinates
scen_eq_latlon  = [34.2,    -116.9]
#utm zone
utm_zone = '11S'

#grid
grid_X_dxdy = [10, 10]

#scenario filename
fname_scen_predict = '../../Data/Prediction/scen_predict.csv'

# UTM projection
# ---------------------------
# projection system
utmProj = pyproj.Proj("+proj=utm +zone="+utm_zone+", +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

#grid limits in UTM
grid_X_win = np.array([[-140, 3500], [780, 4700]])

#create coordinate grid
grid_x_edge = np.arange(grid_X_win[0,0],grid_X_win[1,0],grid_X_dxdy[0])
grid_y_edge = np.arange(grid_X_win[0,1],grid_X_win[1,1],grid_X_dxdy[0])
grid_x, grid_y = np.meshgrid(grid_x_edge, grid_y_edge)
#create coordinate array with all grid nodes
grid_X = np.vstack([grid_x.T.flatten(), grid_y.T.flatten()]).T
#compute lat/lon coordinate array
grid_latlon = np.fliplr(np.array([utmProj(g_x*1000, g_y*1000, inverse=True) for g_x, g_y in 
                                  zip(grid_X[:,0], grid_X[:,1])]))
n_gpt = len(grid_X)

#earthquake UTM coordinates
scen_eq_X = np.array(utmProj(scen_eq_latlon[1], scen_eq_latlon[0])) / 1000

#create earthquake and site ids
eqid_array = np.full(n_gpt, -1)
site_array = -1*(1+np.arange(n_gpt))

# Compute Ergodic Base Scaling
# ---------------------------
#compute distances
scen_dist_array = np.linalg.norm(grid_X - scen_eq_X, axis=1)
scen_dist_array = np.sqrt(scen_dist_array**2 + z_tor**2)

#scenarios of interest
scen_eas_nerg_scl   = np.full(n_gpt, np.nan)
scen_eas_nerg_aleat = np.full(n_gpt, np.nan)
for k, d in enumerate(scen_dist_array):
    fnorm = 1 if sof == 'SS' else 0
    #median and aleatory    
    scen_eas_nerg_scl[k], _, scen_eas_nerg_aleat[k] = ba18.EasF(freq, mag, rrup=d, vs30=vs30, ztor=z_tor, fnorm=fnorm, flag_keep_b7 = False)
    
    
# Summarize Scenario Dataframe
# ---------------------------
df_scen_prdct = pd.DataFrame({'eqid':eqid_array, 'ssn':site_array,
                              'eqLat':np.full(n_gpt,scen_eq_latlon[0]), 'eqLon':np.full(n_gpt,scen_eq_latlon[0]),
                              'staLat':grid_latlon[:,0], 'staLon':grid_latlon[:,1],
                              'eqX':np.full(n_gpt,scen_eq_X[0]), 'eqY':np.full(n_gpt,scen_eq_X[1]), 'eqZ':np.full(n_gpt,-z_tor),
                              'staX':grid_X[:,0], 'staY':grid_X[:,1],
                              'erg_base':scen_eas_nerg_scl})

#save prediction scenarios
df_scen_prdct.to_csv(fname_scen_predict )
