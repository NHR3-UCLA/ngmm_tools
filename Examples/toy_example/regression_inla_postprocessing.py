#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 12:20:36 2022

@author: glavrent
"""
# Working directory and Packages
# ---------------------------

#load packages
import sys
import pathlib
import glob
import re           #regular expression package
import pickle
from joblib import cpu_count
#arithmetic libraries
import numpy as np
#statistics libraries
import pandas as pd
#plot libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import  AutoLocator as plt_autotick

# Define Problem
# ---------------------------
#data filename
fname_data = 'data/examp_obs.csv'
#inla regression filename
fname_inla_reg = 'data/inla_regression/inla_regression.csv'

#output directory
dir_out = 'data/inla_regression/'

# Read Data
# ---------------------------
#observation data
df_data = pd.read_csv(fname_data, index_col=0)
#inla regression results
df_reg_summary = pd.read_csv(fname_inla_reg, index_col=0)

# Summary figures
# ---------------------------
#color bar (mean)
cbar_levs_mean  = np.linspace(-2, 2, 101).tolist()    
cbar_ticks_mean = np.arange(-2, 2.01, 0.8).tolist()    
#color bar (sigma)
cbar_levs_sig  = np.linspace(0.0, 0.5, 101).tolist()    
cbar_ticks_sig = np.arange(0, 0.501, 0.1).tolist()    

# scatter comparison 
fname_fig = 'inla_gp_scatter'
#create figure
fig, ax = plt.subplots(figsize = (10,10))
#obsevations scatter
hl = ax.plot(df_data.tot, df_reg_summary.tot_mean, 'o')
ax.axline((0,0), slope=1, color="black", linestyle="--")
#figure properties
ax.grid(which='both')
#tick size
ax.tick_params(axis='x', labelsize=32)
ax.tick_params(axis='y', labelsize=32)
#figure limits
ax.set_xticks([-2,-1,0,1,2])
ax.set_yticks([-2,-1,0,1,2])
ax.set_xlim([-2.0, 2.0])
ax.set_ylim([-2.0, 2.0])
#labels
ax.set_xlabel('Data',      fontsize=35)
ax.set_ylabel('Estimated', fontsize=35)
#save figure
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png' )

#field mean
fname_fig = 'inla_gp_field_mean'
#create figure
fig, ax = plt.subplots(figsize = (10,11))
#obsevations map
hl = ax.scatter(df_reg_summary.X, df_reg_summary.Y, c=df_reg_summary.tot_mean, marker='D', vmin=-2, vmax=2, s=100)
#figure properties
ax.grid(which='both')
#color bar
cbar = fig.colorbar(hl, orientation="horizontal", pad=0.15, boundaries=cbar_levs_mean, ticks=cbar_ticks_mean)
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
cbar.set_label(r'$\mu(c_0 + c_1(\vec{t}))$', size=35)
#save figure
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png' )

#field std
fname_fig = 'inla_gp_field_std'
#create figure
fig, ax = plt.subplots(figsize = (10,11))
#obsevations map
hl = ax.scatter(df_reg_summary.X, df_reg_summary.Y, c=df_reg_summary.tot_sig, marker='D', vmin=0, vmax=0.5, s=100, cmap='Oranges')
#figure properties
ax.grid(which='both')
#color bar
cbar = fig.colorbar(hl, orientation="horizontal", pad=0.15, boundaries=cbar_levs_sig, ticks=cbar_ticks_sig)
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
cbar.set_label(r'$\psi(c_0 + c_1(\vec{t}))$', size=35)
#save figure
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png' )
