#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:50:27 2022

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
#user functions
def PlotRSMCmp(df_rms_all, c_name, fig_fname):
    
    #create figure axes
    fig, ax = plt.subplots(figsize = (10,10))
    
    for j, k in enumerate(df_rms_all):
        df_rms = df_rms_all[k]
        ds_id = np.array(range(len(df_rms)))
        lcol  = mpl.cm.get_cmap('tab10')(0) if j in [0,2] else mpl.cm.get_cmap('tab10')(1)
        ltype = '-' if j in [0,1] else '--'
        ax.plot(ds_id, df_rms.loc[:,c_name+'_rms'], marker='o', linewidth=2, markersize=10, label=k,  linestyle=ltype, color=lcol)
    #figure properties
    ax.set_ylim([0, max(0.50, max(ax.get_ylim()))])
    ax.set_xlabel('synthetic dataset', fontsize=35)
    ax.set_ylabel('RMSE',   fontsize=35)
    ax.grid(which='both')
    ax.set_xticks(ds_id)
    ax.set_xticklabels(labels=df_rms.index)
    ax.tick_params(axis='x', labelsize=32)
    ax.tick_params(axis='y', labelsize=32)
    #legend
    ax.legend(loc='upper left', fontsize=32)
    #save figure
    fig.tight_layout()
    fig.savefig( fig_fname + '.png' )

    return fig, ax

def PlotKLCmp(df_KL_all, c_name, fig_fname):
    
    #create figure axes
    fig, ax = plt.subplots(figsize = (10,10))
    
    for k in df_KL_all:
        df_KL = df_KL_all[k]
        ds_id = np.array(range(len(df_KL)))
        ax.plot(ds_id, df_KL.loc[:,c_name+'_KL'], linestyle='-', marker='o', linewidth=2, markersize=10, label=k)
    #figure properties
    ax.set_ylim([0, max(0.50, max(ax.get_ylim()))])
    ax.set_xlabel('synthetic dataset', fontsize=35)
    ax.set_ylabel('KL divergence',     fontsize=35)
    ax.grid(which='both')
    ax.set_xticks(ds_id)
    ax.set_xticklabels(labels=df_KL.index)
    ax.tick_params(axis='x', labelsize=32)
    ax.tick_params(axis='y', labelsize=32)
    #legend
    ax.legend(loc='upper left', fontsize=32)
    #save figure
    fig.tight_layout()
    fig.savefig( fig_fname + '.png' )

    return fig, ax


# Define variables
# ---------------------------
# # Sparse Distance Matrix
# # ---   ---   ---   ---   ---
# # NGAWest 2 CA North
# cmp_name  = 'STAN_sparse_cmp_NGAWest2CA'
# reg_title = ['STAN','STAN w/ sp dist matrix']
# reg_fname = ['CMDSTAN_NGAWest2CANorth_corr_cells_chol_eff_small_corr_len','CMDSTAN_NGAWest2CANorth_corr_cells_chol_eff_sp_small_corr_len']
# ylim_time = [0, 800]
# NGAWest 2 CA
cmp_name  = 'STAN_sparse_cmp_NGAWest2CA'
reg_title = ['STAN','STAN w/ sp dist matrix']
reg_fname = ['CMDSTAN_NGAWest2CA_corr_cells_chol_eff_small_corr_len','CMDSTAN_NGAWest2CA_corr_cells_chol_eff_sp_small_corr_len']
ylim_time = [0, 7000]
# NGAWest 2 CA & NGAWest 2 CA North
cmp_name  = 'STAN_sparse_cmp_NGAWest2CA_'
reg_title = ['STAN - NGAW2 CA','STAN - NGAW2 CA North',
             'STAN - NGAW2 CA\nw/ sp dist matrix',f'STAN NGAW2 CA North\nw/ sp dist matrix, ']
reg_fname = ['CMDSTAN_NGAWest2CA_corr_cells_chol_eff_small_corr_len',    'CMDSTAN_NGAWest2CANorth_corr_cells_chol_eff_small_corr_len',
             'CMDSTAN_NGAWest2CA_corr_cells_chol_eff_sp_small_corr_len', 'CMDSTAN_NGAWest2CANorth_corr_cells_chol_eff_sp_small_corr_len']
ylim_time = [0, 7000]

# # Different Software
# # ---   ---   ---   ---   ---
# cmp_name  = 'STAN_vs_INLA_cmp_NGAWest2CANorth'
# reg_title = ['STAN corr. cells','STAN uncorr. cells','INLA uncorr. cells']
# reg_fname = ['CMDSTAN_NGAWest2CANorth_corr_cells_chol_eff_small_corr_len','CMDSTAN_NGAWest2CANorth_corr_cells_chol_eff_small_corr_len',
#              'INLA_NGAWest2CANorth_uncorr_cells_coarse_small_corr_len']
# reg_fname = ['PYSTAN_NGAWest2CANorth_corr_cells_chol_eff_small_corr_len','PYSTAN_NGAWest2CANorth_uncorr_cells_chol_eff_small_corr_len',
#              'INLA_NGAWest2CANorth_uncorr_cells_coarse_small_corr_len']
# ylim_time = [0, 800]


#directories regressions
reg_dir = [f'../../../../Data/Verification/regression/ds2/%s/'%r_f for r_f in reg_fname]

#directory output
dir_out = '../../../../Data/Verification/regression/ds2/comparisons/'

# Load Data
# ---------------------------           
#initialize misfit dataframe
df_sum_misfit_all = {};
#read misfit info
for k, (r_t, r_d) in enumerate(zip(reg_title, reg_dir)):
    #filename misfit info
    fname_sum = r_d + 'summary/misfit_summary.csv'
    #read KL score for coefficients
    df_sum_misfit_all[r_t] = pd.read_csv(fname_sum, index_col=0)

#initialize run time dataframe
df_runinfo_all = {};
#read run time info
for k, (r_t, r_d) in enumerate(zip(reg_title, reg_dir)):
    #filename run time
    fname_runinfo = r_d + '/run_info.csv'
    #store calc time
    df_runinfo_all[r_t] = pd.read_csv(fname_runinfo)


# Comparison Figures
# ---------------------------         
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)

# RMSE divergence
# ---   ---   ---   ---   ---
#coefficient name
c_name = 'nerg_tot'
#figure name
fig_fname = '%s/%s_%s_RMSE'%(dir_out, cmp_name, c_name)
#plotting
PlotRSMCmp(df_sum_misfit_all , c_name, fig_fname);

#coefficient name
c_name = 'dc_1e'
#figure name
fig_fname = '%s/%s_%s_RMSE'%(dir_out, cmp_name, c_name)
#plotting
PlotRSMCmp(df_sum_misfit_all , c_name, fig_fname);

#coefficient name
c_name = 'dc_1as'
#figure name
fig_fname = '%s/%s_%s_RMSE'%(dir_out, cmp_name, c_name)
#plotting
PlotRSMCmp(df_sum_misfit_all , c_name, fig_fname);

#coefficient name
c_name = 'dc_1bs'
#figure name
fig_fname = '%s/%s_%s_RMSE'%(dir_out, cmp_name, c_name)
#plotting
PlotRSMCmp(df_sum_misfit_all , c_name, fig_fname);

# KL divergence
# ---   ---   ---   ---   ---
#coefficient name
c_name = 'nerg_tot'
#figure name
fig_fname = '%s/%s_%s_KLdiv'%(dir_out, cmp_name, c_name)
#plotting
PlotKLCmp(df_sum_misfit_all , c_name, fig_fname);

#coefficient name
c_name = 'dc_1e'
#figure name
fig_fname = '%s/%s_%s_KLdiv'%(dir_out, cmp_name, c_name)
#plotting
PlotKLCmp(df_sum_misfit_all , c_name, fig_fname);

#coefficient name
c_name = 'dc_1as'
#figure name
fig_fname = '%s/%s_%s_KLdiv'%(dir_out, cmp_name, c_name)
#plotting
PlotKLCmp(df_sum_misfit_all , c_name, fig_fname);

#coefficient name
c_name = 'dc_1bs'
#figure name
fig_fname = '%s/%s_%s_KLdiv'%(dir_out, cmp_name, c_name)
#plotting
PlotKLCmp(df_sum_misfit_all , c_name, fig_fname);


# Run Time
# ---   ---   ---   ---   ---
#run time figure
fig_fname = '%s/%s_run_time'%(dir_out, cmp_name)
#create figure axes
fig, ax = plt.subplots(figsize = (10,10))
#iterate over different analyses
for j, k in enumerate(df_runinfo_all):
    ds_id   = df_runinfo_all[k].ds_id
    ds_name = ['Y%i'%d_i for d_i in ds_id]
    run_time = df_runinfo_all[k].run_time
    #
    lcol  = mpl.cm.get_cmap('tab10')(0) if j in [0,2] else mpl.cm.get_cmap('tab10')(1)
    ltype = '-' if j in [0,1] else '--'
    ax.plot(ds_id, run_time, marker='o', linewidth=2, markersize=10, label=k, linestyle=ltype, color=lcol)
#figure properties
ax.set_ylim(ylim_time)
ax.set_xlabel('synthetic dataset', fontsize=35)
ax.set_ylabel('Run Time (min)',    fontsize=35)
ax.grid(which='both')
ax.set_xticks(ds_id)
ax.set_xticklabels(labels=ds_name)
ax.tick_params(axis='x', labelsize=32)
ax.tick_params(axis='y', labelsize=32)
#legend
# ax.legend(loc='lower left', fontsize=32)
# ax.legend(loc='upper left', fontsize=32)
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),  fontsize=25)
#save figure
fig.tight_layout()
fig.savefig( fig_fname + '.png' )
