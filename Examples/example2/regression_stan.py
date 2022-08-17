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
fname_data = 'data/regression_dataset.csv'

#stan parameters
pystan_ver = 2
# n_iter = 50
n_iter = 100
# n_iter = 200
# n_iter = 500
# n_iter = 1000
# n_iter = 10000
# n_iter = 100000
n_chains = 4
adapt_delta   = 0.8
max_treedepth = 10
fname_stan_model = 'regression_stan_model.stan'

#output directory
dir_out = f'data/stan_regression_%iiter/'%n_iter

# Read Data
# ---------------------------
df_data = pd.read_csv(fname_data)

#read stan model
with open(fname_stan_model, "r") as f:
    stan_model_code = f.read()

# Preprocess Data
# ---------------------------
n_data = len(df_data)

#scaling
x1_data =  df_data['x1'].to_numpy().copy()

#observations  
y_data = df_data['y'].to_numpy().copy()

#stan data
stan_data = {'N':   n_data, 
             'X_1': x1_data,
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
#hyper-parameters and model coeffs
col_names = ['c_0','c_1', 'sigma']

#stan posterior
stan_posterior = np.stack([stan_fit[c_n].flatten() for c_n in col_names], axis=1)

#save raw-posterior distribution
df_stan_posterior_raw = pd.DataFrame(stan_posterior, columns = col_names)

#summarize posterior distributions of hyper-parameters
perc_array = np.array([0.05,0.25,0.5,0.75,0.95])
df_stan_hyp = df_stan_posterior_raw[col_names].quantile(perc_array)
df_stan_hyp = df_stan_hyp.append(df_stan_posterior_raw[col_names].mean(axis = 0), ignore_index=True)
df_stan_hyp.index = ['prc_%.2f'%(prc) for prc in perc_array]+['mean'] 
    
# model prediction and residuals
#---  ---  ---  ---  ---  ---  ---  ---
c0_mu = df_stan_hyp.loc['mean','c_0']
c1_mu = df_stan_hyp.loc['mean','c_1']
#mean prediction
y_mu  = c0_mu + c1_mu * x1_data
# residuals
res   = y_data - y_mu

# prediction
#---  ---  ---  ---  ---  ---  ---  ---
x1_array = np.linspace(-4,4)
y_array  =  c0_mu + c1_mu * x1_array

# summarize regression results
#---  ---  ---  ---  ---  ---  ---  ---
#initialize flat-file for summary of coefficients and residuals
df_info = df_data[['x1']]
#summarize coeff and predictions
reg_summary = np.vstack((y_mu, res)).T
columns_names = ['y_mu', 'res_mean']
df_reg_summary = pd.DataFrame(reg_summary, columns = columns_names, index=df_data.index)
df_reg_summary = pd.merge(df_info, df_reg_summary, how='right', left_index=True, right_index=True)
  
# Output directory
# ---------------------------
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
#MCMC samples
df_stan_posterior_raw.to_csv(dir_out + 'stan_posterior_raw.csv')
#regression results
df_reg_summary.to_csv( dir_out + 'stan_regression.csv' )

# Summary figures
# ---------------------------
# prediction
fname_fig = 'stan_prediction'
#create figure
fig, ax = plt.subplots(figsize = (10,10))
#obsevations scatter
hl = ax.plot(df_data.x1, df_data.y, 'o')
hl = ax.plot(x1_array, y_array, color="black", )
#figure properties
ax.grid(which='both')
#tick size
ax.tick_params(axis='x', labelsize=32)
ax.tick_params(axis='y', labelsize=32)
#figure limits
ax.set_xlim([-4, 4])
ax.set_ylim([-3, 3])
#labels
ax.set_xlabel(f'$x_1$',      fontsize=35)
ax.set_ylabel(f'$y$', fontsize=35)
#save figure
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png' )


#stan residuals
fname_fig = 'stan_residuals'
#create figure
fig, ax = plt.subplots(figsize = (10,10))
#obsevations scatter
hl = ax.plot(df_reg_summary.x1, df_reg_summary.res_mean, 'o')
# ax.axline((0,0), slope=1, color="black", linestyle="--")
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
ax.set_xlabel(f'$x_1$',      fontsize=35)
ax.set_ylabel(f'$\epsilon$', fontsize=35)
#save figure
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png' )


#create stan trace plots
chain_cmap = mpl.cm.get_cmap('tab10')
for c_name in col_names:
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
    if c_name == 'c_0':
        ax[0].set_xlim([-0.25,-0.1])
        ax[0].set_ylim([0,30])
        ax[1].set_ylim([-0.4,0.0])
    elif c_name == 'c_1':
        ax[0].set_xlim([0.5,0.8])
        ax[0].set_ylim([0,30])
        ax[1].set_ylim([0.5,0.8])
    elif c_name == 'sigma':
        ax[0].set_xlim([0.6,0.8])
        ax[0].set_ylim([0,30])
        ax[1].set_ylim([0.6,0.8])
    fig = ax[0].figure
    fig.suptitle(c_name, fontsize=35)
    fig.savefig(dir_out + 'stan_traceplot_' + c_name + '_arviz' + '.png')

