#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:50:27 2022

@author: glavrent
"""
# Working directory and Packages
# ---------------------------
#change working directory
import os
os.chdir('/mnt/halcloud_nfs/glavrent/Research/Nonerg_GMM_methodology/Analyses/Code_Verification/regressions/ds1')

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
#user functions
def PlotRSMCmp(df_KL_all, c_name, fig_fname):
    
    #create figure axes
    fig, ax = plt.subplots(figsize = (10,10))
    
    for m_i in df_KL_all:
        df_KL = df_KL_all[m_i]
        ds_id = np.array(range(len(df_KL)))
        ax.plot(ds_id, df_KL.loc[:,c_name], linestyle='-', marker='o', linewidth=2, markersize=10, label=m_i)
    #figure properties
    ax.set_ylim([0, max(0.50, max(ax.get_ylim()))])
    ax.set_xlabel('synthetic dataset', fontsize=30)
    ax.set_ylabel('RMSE divergence', fontsize=30)
    ax.grid(which='both')
    ax.set_xticks(ds_id)
    ax.set_xticklabels(labels=df_KL.index)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    #legend
    ax.legend(loc='upper left', fontsize=30)
    #save figure
    fig.tight_layout()
    fig.savefig( fig_fname + '.png' )

    return fig, ax

def PlotKLCmp(df_KL_all, c_name, fig_fname):
    
    #create figure axes
    fig, ax = plt.subplots(figsize = (10,10))
    
    for m_i in df_KL_all:
        df_KL = df_KL_all[m_i]
        ds_id = np.array(range(len(df_KL)))
        ax.plot(ds_id, df_KL.loc[:,c_name], linestyle='-', marker='o', linewidth=2, markersize=10, label=m_i)
    #figure properties
    ax.set_ylim([0, max(0.50, max(ax.get_ylim()))])
    ax.set_xlabel('synthetic dataset', fontsize=30)
    ax.set_ylabel('KL divergence', fontsize=30)
    ax.grid(which='both')
    ax.set_xticks(ds_id)
    ax.set_xticklabels(labels=df_KL.index)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    #legend
    ax.legend(loc='upper left', fontsize=25)
    #save figure
    fig.tight_layout()
    fig.savefig( fig_fname + '.png' )

    return fig, ax


# Define variables
# ---------------------------

#comparisons
name_reg = ['PYSTAN_NGAWest2CANorth_chol_eff_small_corr_len','PYSTAN3_NGAWest2CANorth_chol_eff_small_corr_len',]

#dataset info
ds_id = 1

#correlation info
# 1: Small Correlation Lengths
# 2: Large Correlation Lenghts
corr_id = 1

#packages comparison
packg_info = ['PYSTAN', 'PYSTAN3', 'fine']

#correlation name
if corr_id == 1:   synds_suffix = '_small_corr_len' 
elif corr_id == 2: synds_suffix = '_large_corr_len'

#dataset name
if ds_id == 1:   name_dataset = 'NGAWest2CANorth'
elif ds_id == 2: name_dataset = 'NGAWest2CA'
elif ds_id == 3: name_dataset = 'NGAWest3CA'

#directories regressions
dir_reg = [f'../../../../Data/Verification/regression/ds1/%s_%s_%s%s/'%(name_dataset, m_info, synds_suffix) for m_info in mesh_info]

#directory output
dir_out = '../../../../Data/Verification/regression/ds1/comparisons/'

# Load Data
# ---------------------------           
#initialize dataframe
df_KL_all = {};

#read KL scores
for k, (d_r, m_i) in enumerate(zip(dir_reg, mesh_info)):
    #filename KL score
    fname_KL = d_r + 'summary/coeffs_KL_divergence.csv'
    #read KL score for coefficients
    df_KL_all[m_i] = pd.read_csv(fname_KL, index_col=0)

# Comparison Figures
# ---------------------------         
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
  
#coefficient name
c_name = 'nerg_tot'
#figure name
fig_fname = '%s/%s%s_KLdiv_%s'%(dir_out, name_dataset, synds_suffix, c_name)
#plotting
PlotKLCmp(df_KL_all, c_name, fig_fname);

#coefficient name
c_name = 'dc_1e'
#figure name
fig_fname = '%s/%s%s_KLdiv_%s'%(dir_out, name_dataset, synds_suffix, c_name)
#plotting
PlotKLCmp(df_KL_all, c_name, fig_fname);

#coefficient name
c_name = 'dc_1as'
#figure name
fig_fname = '%s/%s%s_KLdiv_%s'%(dir_out, name_dataset, synds_suffix, c_name)
#plotting
PlotKLCmp(df_KL_all, c_name, fig_fname);

#coefficient name
c_name = 'dc_1bs'
#figure name
fig_fname = '%s/%s%s_KLdiv_%s'%(dir_out, name_dataset, synds_suffix, c_name)
#plotting
PlotKLCmp(df_KL_all, c_name, fig_fname);

