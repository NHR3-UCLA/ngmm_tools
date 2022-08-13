#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:38:50 2022

@author: glavrent
"""
# Working directory and Packages
# ---------------------------
#load variables
import os
import sys
import pathlib
#arithmetic libraries
import numpy as np
#statistics libraries
import pandas as pd
#plot libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import  AutoLocator as plt_autotick

# Define variables
# ---------------------------
#mesh info
mesh_info = ['coarse', 'medium', 'fine']

#dataset name
dataset_name = ['NGAWest2CANorth', 'NGAWest2CA', 'NGAWest3CA']

#correlation info
# 1: Small Correlation Lengths
# 2: Large Correlation Lenghts
corr_id = 1

#correlation name
if corr_id == 1:
    synds_name   = 'small corr len'
    synds_suffix = '_small_corr_len' 
elif corr_id == 2:
    synds_name   = 'large corr len'
    synds_suffix = '_large_corr_len'

#directories regressions
dir_reg = '../../../../Data/Verification/regression/ds1/'

#directory output
dir_out = '../../../../Data/Verification/regression/ds1/comparisons/'

# Load Data
# ---------------------------           
#initialize dataframe
df_runinfo_all = {};

#iterate over different analyses
for j1, m_i in enumerate(mesh_info):
    for j2, d_n in enumerate(dataset_name):
        key_runinfo   = '%s_%s'%(m_i, d_n)
        fname_runinfo = '%s/INLA_%s_%s%s/run_info.csv'%(dir_reg, d_n, m_i, synds_suffix)
        #store calc time
        df_runinfo_all[key_runinfo] = pd.read_csv(fname_runinfo)
        


# Comparison Figures
# ---------------------------         
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)

#line style (iterate with mesh info)
line_style = [':','--','-']
#color map (iterate with dataset)
c_map = plt.get_cmap('Dark2')

#run time figure
fig_fname = 'run_time_inla'
#create figure axes
fig, ax = plt.subplots(figsize = (20,10))
#iterate over different analyses
for j2, d_n in enumerate(dataset_name):
    for j1, (m_i, l_s) in enumerate(zip(mesh_info, line_style)):
        key_runinfo   = '%s_%s'%(m_i, d_n)
        #
        ds_id   = df_runinfo_all[key_runinfo].ds_id
        ds_name = ['Y%i'%d_i for d_i in ds_id]
        #
        run_time = df_runinfo_all[key_runinfo].run_time
        
        ax.plot(ds_id, run_time, linestyle=l_s, marker='o', linewidth=2, markersize=10, color=c_map(j2), label='%s - %s'%(d_n, m_i))

#figure properties
ax.set_ylim([0, max(0.50, max(ax.get_ylim()))])
ax.set_xlabel('synthetic dataset', fontsize=30)
ax.set_ylabel('Run Time (min)',    fontsize=30)
ax.grid(which='both')
ax.set_xticks(ds_id)
ax.set_xticklabels(labels=ds_name)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
#legend
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),  fontsize=25)
#save figure
fig.tight_layout()
fig.savefig( dir_out + fig_fname + '.png' )



