#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:40:29 2022

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
from matplotlib.ticker import FormatStrFormatter
#user functions
def PlotRSMCmp(df_list, names_list, c_name, width, fig_fname):
    
    #create figure axes
    fig, ax = plt.subplots(figsize = (10,10))

    #
    x_offset = 

    #plot rms value
    for nm, df_l in zip(names_list, df_list):
        ax.bar(np.arange(df_l)-x_offset, df_sum_reg_stan.nerg_tot_rms.values, width=width, label=nm)

    #figure properties
    ax.set_ylim([0, 0.2])
    ax.set_xticks(labels=df_l.ds_name)
    ax.set_xlabel('Dataset', fontsize=35)
    ax.set_ylabel('RMSE',    fontsize=35)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=32)
    ax.tick_params(axis='y', labelsize=32)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #legend
    ax.legend(loc='upper left', fontsize=32)
    #save figure
    fig.tight_layout()
    fig.savefig( fig_fname + '.png' )
    
    


# Define variables
# ---------------------------
# COMPARISONS
# Different Mesh sizes 
# ---   ---   ---   ---   ---
cmp_name   = 'STAN_INLA_medium'
reg_title  = [f'STAN - NGAW2 CA, North', f'STAN - NGAW2 CA', f'STAN - NGAW3* CA',
              f'INLA - NGAW2 CA, North', f'INLA - NGAW2 CA', f'INLA - NGAW3* CA' ]
reg_fname  = ['CMDSTAN_NGAWest2CANorth_chol_eff_small_corr_len', 'CMDSTAN_NGAWest2CA_chol_eff_small_corr_len', 'CMDSTAN_NGAWest3CA_chol_eff_small_corr_len',  
              'INLA_NGAWest2CANorth_medium_small_corr_len',      'INLA_NGAWest2CA_medium_small_corr_len',      'INLA_NGAWest3CA_medium_small_corr_len']
ds_name    = ['NGAW2\nCA, North','NGAW2\nCA', 'NGAW3\nCA', 
              'NGAW2\nCA, North','NGAW2\nCA', 'NGAW3*\nCA']
ds_id      = np.array([1,2,3,1,2,3])
sftwr_name = 3*['STAN'] + 3*['INLA']
sftwr_id   = np.array(3*[1]+3*[2])

#directories regressions
reg_dir = [f'../../../../Data/Verification/regression/ds1/%s/'%r_f for r_f in reg_fname]

#directory output
dir_out = '../../../../Data/Verification/regression/ds1/comparisons/'

# Load Data
# ---------------------------
#intialize main regression summary dataframe
df_sum_reg = pd.DataFrame({'ds_id':ds_id, 'ds_name':ds_name, 'sftwr_id':sftwr_id, 'sftwr_name':sftwr_name})

#initialize misfit dataframe
df_sum_misfit_all = {};
#read misfit info
for k, (r_t, r_d) in enumerate(zip(reg_title, reg_dir)):
    #filename misfit info
    fname_sum = r_d + 'summary/misfit_summary.csv'
    #read KL score for coefficients
    df_sum_misfit_all[r_t] = pd.read_csv(fname_sum, index_col=0)
    #summarize regression rms
    df_sum_reg.loc[k,'nerg_tot_rms'] = df_sum_misfit_all[r_t].nerg_tot_rms.mean()

#initialize run time dataframe
df_runinfo_all = {};
#read run time info
for k, (r_t, r_d) in enumerate(zip(reg_title, reg_dir)):
    #filename run time
    fname_runinfo = r_d + '/run_info.csv'
    #store calc time
    df_runinfo_all[r_t] = pd.read_csv(fname_runinfo)
    #summarize regression rms
    df_sum_reg.loc[k,'run_time'] = df_runinfo_all[r_t].run_time.mean()
    #print mean run time
    print(f'%s: %.1f min'%( r_t, df_runinfo_all[r_t].run_time.mean() ))

#separate STNA and INLA runs
df_sum_reg_stan = df_sum_reg.loc[df_sum_reg.sftwr_id==1,:]
df_sum_reg_inla = df_sum_reg.loc[df_sum_reg.sftwr_id==2,:]

# Comparison Figures
# ---------------------------         
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)

# RMSE 
# ---   ---   ---   ---   ---
#coefficient name
c_name = 'nerg_tot'
#figure name
fig_fname = '%s/%s_%s_RMSE'%(dir_out, cmp_name, c_name)

# #create figure axes
# fig, ax = plt.subplots(figsize = (10,10))
# #plot rms value
# ax.bar(np.array([1,3,5])-0.3, df_sum_reg_stan.nerg_tot_rms.values, width=0.6, label='STAN')
# ax.bar(np.array([1,3,5])+0.3, df_sum_reg_inla.nerg_tot_rms.values, width=0.6, label='INLA')
# #figure properties
# ax.set_ylim([0, 0.2])
# ax.set_xticks([1,3,5], df_sum_reg_stan.ds_name)
# ax.set_xlabel('Dataset', fontsize=35)
# ax.set_ylabel('RMSE',    fontsize=35)
# ax.grid(which='both')
# ax.tick_params(axis='x', labelsize=32)
# ax.tick_params(axis='y', labelsize=32)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# #legend
# ax.legend(loc='upper left', fontsize=32)
# #save figure
# fig.tight_layout()
# fig.savefig( fig_fname + '.png' )


# Run Time
# ---   ---   ---   ---   ---
#run time figure
fig_fname = '%s/%s_run_time'%(dir_out, cmp_name)
#create figure axes
fig, ax = plt.subplots(figsize = (10,10))
#plot rms value
ax.bar(np.array([1,3,5])-0.3, df_sum_reg_stan.run_time.values, width=0.6, label='STAN')
ax.bar(np.array([1,3,5])+0.3, df_sum_reg_inla.run_time.values, width=0.6, label='INLA')
#figure properties
# ax.set_ylim([0, 0.2])
ax.set_xticks([1,3,5], df_sum_reg_stan.ds_name)
ax.set_xlabel('Dataset',           fontsize=35)
ax.set_ylabel('Run Time (min)',    fontsize=35)
ax.grid(which='both')
ax.set_yscale('log')
ax.tick_params(axis='x', labelsize=32)
ax.tick_params(axis='y', labelsize=32)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#legend
ax.legend(loc='upper left', fontsize=32)
#save figure
fig.tight_layout()
fig.savefig( fig_fname + '.png' )
