#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:01:54 2022

@author: glavrent
"""
# Working directory and Packages
# ---------------------------
#load packages
import os
import sys
import pathlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import linalg as scipylinalg 
#geographic libraries
import pyproj
import geopy.distance
#ground-motion models
import pygmm
#plottign libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
#user libraries
sys.path.insert(0,'../../Analyses/Python_lib/ground_motions')
import pylib_NGMM_prediction as pyNGMM

# Define Problem
# ---------------------------
#hyper-parameters
ell   = 25
omega = 0.4
sig   = 0.6

#grid
grid_win = np.array([[ 0, 0], [100, 100]])
grid_dxdy = [1, 1]

#number of samples
n_samp = 150
n_rep  = 10

#output directory
dir_out = 'data/'

# Grid
# ---------------------------
#create coordinate grid
grid_x_edge = np.arange(grid_win[0,0],grid_win[1,0]+1e-9,grid_dxdy[0])
grid_y_edge = np.arange(grid_win[0,1],grid_win[1,1]+1e-9,grid_dxdy[0])
grid_x, grid_y = np.meshgrid(grid_x_edge, grid_y_edge, indexing='ij')
#create coordinate array with all grid nodes
grid_X = np.vstack([grid_x.T.flatten(), grid_y.T.flatten()]).T
#number of grid points
n_pt_g = grid_X.shape[0]
n_pt_x = len(grid_x_edge)
n_pt_y = len(grid_y_edge)
#grid point ids 
grid_ids = np.arange(n_pt_g)

del grid_x, grid_y

# Create Dataset
# ---------------------------
# Underling process
# ---   ---   ---   ---   ---
#grid covariance matrix
grid_cov = pyNGMM.KernelNegExp(grid_X, grid_X, hyp_ell=ell, hyp_omega=omega, delta=1e-9)
#grid GP 
grid_gp = np.linalg.cholesky(grid_cov) @ np.random.normal(size=n_pt_g)
#constant offset
c0 = np.random.normal(0, 0.1)
#GP dataframe
df_gp = pd.DataFrame({'g_id':grid_ids , 'X':grid_X[:,0], 'Y':grid_X[:,1], 'c0':c0, 'gp':grid_gp}).set_index('g_id')
#total effect
df_gp.loc[:,'tot'] = df_gp[['c0','gp']].sum(axis=1)


# Samples
# ---   ---   ---   ---   ---
#random samples
samp_ids_orig = np.random.randint(n_pt_g, size=n_samp)
samp_ids = np.hstack([np.full(np.random.randint(low=1, high=n_rep, size=1), s) for s in samp_ids_orig])

#samples data frame
df_samp = df_gp.loc[samp_ids,:].reset_index()
df_samp.index.name = 'samp_id'
#noise term
df_samp.loc[:,'eps'] = np.random.normal(0, sig, len(df_samp))
#response variable
df_samp.loc[:,'y'] = df_samp[['tot','eps']].sum(axis=1)

# Save Dataset
# ---------------------------
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 

df_gp.to_csv( dir_out + 'examp_grid_gp.csv' )
df_samp.to_csv( dir_out + 'examp_obs.csv' )

# Summary Figures
# ---------------------------
#color bar
cbar_levs  = np.linspace(-2, 2, 101).tolist()    
cbar_ticks = np.arange(-2, 2.01, 0.8).tolist()    

#figure title
fname_fig = 'examp_data_gp_field'
#create figure
fig, ax = plt.subplots(figsize = (10,11))
#contour plot
cs = ax.contourf(grid_x_edge, grid_y_edge, df_gp.tot.values.reshape(n_pt_x,n_pt_y), vmin=-2, vmax=2, levels = cbar_levs)
#obsevations
hl = ax.plot(df_samp.X, df_samp.Y, 'o', color='black',markersize=12, markerfacecolor='none', markeredgewidth=2)
#figure properties
ax.grid(which='both')
#color bar
cbar = fig.colorbar(cs, orientation="horizontal", pad=0.15, boundaries=cbar_levs, ticks=cbar_ticks)
#tick size
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
#labels
ax.set_xlabel(r'$t_1$', fontsize=35)
ax.set_ylabel(r'$t_2$', fontsize=35)
#update colorbar 
cbar.ax.tick_params(tick1On=1, labelsize=30)
cbar.set_label(r'$c_0 + c_1(\vec{t})$', size=35)
#save figure
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png' )

#figure title
fname_fig = 'examp_obs_gp_field'
#create figure
fig, ax = plt.subplots(figsize = (10,11))
#obsevations
hl = ax.scatter(df_samp.X, df_samp.Y, c=df_samp.tot, vmin=-2, vmax=2, s=100)
#figure properties
ax.grid(which='both')
#color bar
cbar = fig.colorbar(hl, orientation="horizontal", pad=0.15, boundaries=cbar_levs, ticks=cbar_ticks)
#tick size
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
#labels
ax.set_xlabel(r'$t_1$', fontsize=35)
ax.set_ylabel(r'$t_2$', fontsize=35)
#figure limits
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
#update colorbar 
cbar.ax.tick_params(tick1On=1, labelsize=30)
cbar.set_label(r'$c_0 + c_1(\vec{t})$', size=35)
#save figure
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png' )

#figure title
fname_fig = 'examp_obs_noise'
#create figure
fig, ax = plt.subplots(figsize = (10,11))
#obsevations
hl = ax.scatter(df_samp.X, df_samp.Y, c=df_samp.y, vmin=-2, vmax=2, s=100)
#figure properties
ax.grid(which='both')
#color bar
cbar = fig.colorbar(hl, orientation="horizontal", pad=0.15, boundaries=cbar_levs, ticks=cbar_ticks)
#tick size
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
#labels
ax.set_xlabel(r'$t_1$', fontsize=35)
ax.set_ylabel(r'$t_2$', fontsize=35)
#figure limits
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
#update colorbar 
cbar.ax.tick_params(tick1On=1, labelsize=30)
cbar.set_label(r'$y=c_0 + c_1(\vec{t}) + \epsilon$', size=35)
#save figure
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png' )
