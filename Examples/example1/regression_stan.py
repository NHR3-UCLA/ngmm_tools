#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 12:20:36 2022

@author: glavrent
"""
# Working directory and Packages
# ---------------------------

#load packages
import os
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
import arviz as az

# Define Problem
# ---------------------------
#data filename
fname_data = 'data/examp_obs.csv'

#stan parameters
pystan_ver = 2
n_iter = 10000
n_chains = 4
adapt_delta   = 0.8
max_treedepth = 10
fname_stan_model = 'regression_stan_model.stan'

#output directory
dir_out = 'data/stan_regression/'

# Read Data
# ---------------------------
df_data = pd.read_csv(fname_data, index_col=0)

#read stan model
with open(fname_stan_model, "r") as f:
    stan_model_code = f.read()

# Preprocess Data
# ---------------------------
n_data = len(df_data)

#grid data
data_grid_all = df_data[['g_id','X','Y']].values
_, g_idx, g_inv = np.unique(df_data[['g_id']].values, axis=0, return_inverse=True, return_index=True)
data_grid = data_grid_all[g_idx,:]
X_g = data_grid[:,[1,2]] #grid coordinates
#create grid ids for all data (1 to n_g)
g_id = g_inv + 1
n_g = len(data_grid)

#observations  
y_data = df_data['y'].to_numpy().copy()

#stan data
stan_data = {'N':   n_data, 
             'NG':  n_g,
             'gid': g_id,   #grid id
             'X_g': X_g,    #grid coordinates
             'Y':   y_data
            }    

# Run Stan
# ---------------------------
if pystan_ver == 2:
    import pystan
    #control paramters
    control_stan = {'adapt_delta':adapt_delta, 'max_treedepth':max_treedepth}
    #compile 
    stan_model = pystan.StanModel(model_code=stan_model_code)
    #full Bayesian statistics
    stan_fit = stan_model.sampling(data=stan_data, iter=n_iter, chains = n_chains, refresh=10, control = control_stan)
elif pystan_ver == 3:
    import nest_asyncio
    import stan
    #compile stan
    nest_asyncio.apply()
    stan_model = stan.build(stan_model_code, data=stan_data, random_seed=1)
    #run stan
    stan_fit = stan_model.sample(num_chains=n_chains, num_samples=n_iter, max_depth=max_treedepth, delta=adapt_delta)


# Post-processing
# ---------------------------
#hyper-parameters
col_names_hyp = ['c_0','ell', 'omega', 'sigma']
#spatially varying term
col_names_c1 = ['c_1.%i'%(k)  for k in range(n_g)]
col_names_all = col_names_hyp + col_names_c1

#stan posterior
stan_posterior = np.stack([stan_fit[c_n].flatten() for c_n in col_names_hyp], axis=1)
if pystan_ver == 2:
    stan_posterior = np.concatenate((stan_posterior, stan_fit['c_1']),   axis=1)
elif pystan_ver == 3:
    stan_posterior = np.concatenate((stan_posterior, stan_fit['c_1'].T), axis=1)

#save raw-posterior distribution
df_stan_posterior_raw = pd.DataFrame(stan_posterior, columns = col_names_all)

#summarize posterior distributions of hyper-parameters
perc_array = np.array([0.05,0.25,0.5,0.75,0.95])
df_stan_hyp = df_stan_posterior_raw[col_names_hyp].quantile(perc_array)
df_stan_hyp = df_stan_hyp.append(df_stan_posterior_raw[col_names_hyp].mean(axis = 0), ignore_index=True)
df_stan_hyp.index = ['prc_%.2f'%(prc) for prc in perc_array]+['mean'] 
    
# model coefficients
#---  ---  ---  ---  ---  ---  ---  ---
#constant shift coefficient
coeff_0_mu  = df_stan_posterior_raw.loc[:,'c_0'].mean()   * np.ones(n_data)
coeff_0_med = df_stan_posterior_raw.loc[:,'c_0'].median() * np.ones(n_data)
coeff_0_sig = df_stan_posterior_raw.loc[:,'c_0'].std()    * np.ones(n_data)

#spatially varying earthquake constant coefficient
coeff_1_mu  = np.array([df_stan_posterior_raw.loc[:,f'c_1.{k}'].mean()    for k in range(n_g)])[g_inv]
coeff_1_med = np.array([df_stan_posterior_raw.loc[:,f'c_1.{k}'].median() for k in range(n_g)])[g_inv]
coeff_1_sig = np.array([df_stan_posterior_raw.loc[:,f'c_1.{k}'].std()    for k in range(n_g)])[g_inv]

# model prediction and residuals
#---  ---  ---  ---  ---  ---  ---  ---
#mean prediction
y_mu  = (coeff_0_mu + coeff_1_mu)
#std of prediction
y_sig = np.sqrt(coeff_0_sig**2 + coeff_1_sig**2)
# residuals
res   = y_data - y_mu

# summarize regression results
#---  ---  ---  ---  ---  ---  ---  ---
#initialize flat-file for summary of coefficients and residuals
df_info = df_data[['g_id','X','Y']]
#summarize coeff and predictions
reg_summary = np.vstack((coeff_0_mu, coeff_0_sig,
                         coeff_1_mu, coeff_1_sig,
                         y_mu, y_sig, res)).T
columns_names = ['c_0_mean', 'c_0_sig',
                 'c_1_mean', 'c_1_sig',
                 'tot_mean',   'tot_sig', 'res']
df_reg_summary = pd.DataFrame(reg_summary, columns = columns_names, index=df_data.index)
df_reg_summary = pd.merge(df_info, df_reg_summary, how='right', left_index=True, right_index=True)
  
# Output directory
# ---------------------------
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
#regression results
df_reg_summary.to_csv( dir_out + 'stan_regression.csv' )

# Summary figures
# ---------------------------
#color bar (mean)
cbar_levs_mean  = np.linspace(-2, 2, 101).tolist()    
cbar_ticks_mean = np.arange(-2, 2.01, 0.8).tolist()    
#color bar (sigma)
cbar_levs_sig  = np.linspace(0.0, 0.5, 101).tolist()    
cbar_ticks_sig = np.arange(0, 0.501, 0.1).tolist()    

# scatter comparison 
fname_fig = 'stan_gp_scatter'
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
fname_fig = 'stan_gp_field_mean'
#create figure
fig, ax = plt.subplots(figsize = (10,11))
#obsevations map
hl = ax.scatter(df_reg_summary.X, df_reg_summary.Y, c=df_reg_summary.tot_mean, marker='s', vmin=-2, vmax=2, s=100)
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
fname_fig = 'stan_gp_field_std'
#create figure
fig, ax = plt.subplots(figsize = (10,11))
#obsevations map
hl = ax.scatter(df_reg_summary.X, df_reg_summary.Y, c=df_reg_summary.tot_sig, marker='s', vmin=0, vmax=0.5, s=100, cmap='Oranges')
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

#create stan trace plots
chain_cmap = mpl.cm.get_cmap('tab10')
for c_name in col_names_hyp:
    #create trace plot with arviz
    ax = az.plot_trace(stan_fit,  var_names=c_name, figsize=(20,10)).ravel()
    #change colors
    for a in ax:
        for c_i in range(n_chains):
            a.get_lines()[c_i].set_color(chain_cmap(c_i))
            a.get_lines()[c_i].set_linestyle('-')
            a.get_lines()[c_i].set_alpha(1)
    #edit figure
    ax[0].yaxis.set_major_locator(plt_autotick())
    # ax[0].set_xlabel('sample value')
    # ax[0].set_ylabel('frequency')
    ax[0].grid(axis='both')
    ax[0].tick_params(axis='x', labelsize=30)
    ax[0].tick_params(axis='y', labelsize=30)
    ax[0].set_xlabel(c_name, fontsize=35)
    ax[0].set_ylabel('posterior(%s)'%c_name, fontsize=35)
    ax[0].set_title('')
    # ax[1].set_xlabel('iteration')
    # ax[1].set_ylabel('sample value')
    ax[1].grid(axis='both')
    ax[1].legend(['chain %i'%(c_i+1) for c_i in range(n_chains)], loc='upper right', fontsize=32)
    ax[1].tick_params(axis='x', labelsize=30)
    ax[1].tick_params(axis='y', labelsize=30)
    ax[1].set_xlabel('iteration', fontsize=35)
    ax[1].set_ylabel(c_name,      fontsize=35)
    ax[1].set_title('')
    if c_name == 'omega':
        ax[0].set_xlim([0.2,1.2])
        ax[0].set_ylim([0,10])
        ax[1].set_ylim([0.2,1.2])
    fig = ax[0].figure
    fig.suptitle(c_name, fontsize=35)
    fig.savefig(dir_out + 'stan_traceplot_' + c_name + '_arviz' + '.png')

