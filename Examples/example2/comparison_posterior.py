#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 07:00:55 2022

@author: glavrent
"""

# Load Packages
# ---------------------------
#arithmetic libraries
import numpy as np
from scipy import stats
#statistics libraries
import pandas as pd
#plottign libraries
import matplotlib as mpl
from matplotlib import pyplot as plt

# Load Data
# ---------------------------
# posterior distributions 
# ---   ---   ---
#stan regression
# fname_post_stan = ['data/stan_regression_100iter/stan_posterior_raw.csv',
#                    'data/stan_regression_200iter/stan_posterior_raw.csv',
#                    'data/stan_regression_500iter/stan_posterior_raw.csv',
#                    'data/stan_regression_100000iter/stan_posterior_raw.csv']
# n_iter_stan = [100, 200, 500, 100000]
fname_post_stan = ['data/stan_regression_100iter/stan_posterior_raw.csv',
                   'data/stan_regression_1000iter/stan_posterior_raw.csv',
                   'data/stan_regression_100000iter/stan_posterior_raw.csv']
n_iter_stan = [100, 1000, 100000]
# fname_post_stan = ['data/stan_regression_200iter/stan_posterior_raw.csv',
#                    'data/stan_regression_1000iter/stan_posterior_raw.csv',
#                    'data/stan_regression_100000iter/stan_posterior_raw.csv']
# n_iter_stan = [200, 1000, 100000]

#inla regression
fname_post_inla_c0    = 'data/inla_regression/inla_c0_posterior.csv'
fname_post_inla_c1    = 'data/inla_regression/inla_c1_posterior.csv'
fname_post_inla_sigma = 'data/inla_regression/inla_sigma_posterior.csv'

#load posterior distributions
df_post_stan_raw = [pd.read_csv(fn) for fn in fname_post_stan]
#inla
df_post_inla_c0    = pd.read_csv(fname_post_inla_c0)
df_post_inla_c1    = pd.read_csv(fname_post_inla_c1)
df_post_inla_sigma = pd.read_csv(fname_post_inla_sigma)

#process stan posteriors
#c0
c0_array = np.linspace(-.4, 0.0, 1000)
post_stan_c0_kde = [stats.gaussian_kde(df['c_0']) for df in df_post_stan_raw]
df_post_stan_c0 = [pd.DataFrame({'x':c0_array, 'y':p_kde(c0_array)}) 
                      for p_kde in post_stan_c0_kde]
#c1
c1_array = np.linspace(0.5, 0.8, 1000)
post_stan_c1_kde = [stats.gaussian_kde(df['c_1']) for df in df_post_stan_raw]
df_post_stan_c1 = [pd.DataFrame({'x':c1_array, 'y':p_kde(c1_array)}) 
                      for p_kde in post_stan_c1_kde]
#sigma
sigma_array = np.linspace(0.6, 0.8, 1000)
post_stan_sigma_kde = [stats.gaussian_kde(df['sigma']) for df in df_post_stan_raw]
df_post_stan_sigma = [pd.DataFrame({'x':sigma_array, 'y':p_kde(sigma_array)}) 
                      for p_kde in post_stan_sigma_kde]


# Create Figures
# ---------------------------
#figure title
fname_fig = 'post_c0'
#create figure
fig, ax = plt.subplots(figsize = (10,10))
#plot examples
for df, n_iter in zip(df_post_stan_c0, n_iter_stan):
    ax.plot(df.x, df.y, linewidth=4, label=r'STAN, $n_{iter}=%i$'%n_iter)
ax.plot(df_post_inla_c0.x, df_post_inla_c0.y, linewidth=4,  linestyle='--', color='black', label=r'INLA')
#figure properties
ax.grid(which='both')
#tick size
ax.tick_params(axis='x', labelsize=32)
ax.tick_params(axis='y', labelsize=32)
#figure limits
ax.set_xlim([-.25, -0.1])
ax.set_ylim([0.0, 20.0])
#legend
pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width * 0.6, pos.height])
ax.legend(loc='center right', bbox_to_anchor=(2.0, 0.5), fontsize=32)
#labels
ax.set_xlabel(r'$c_0$',       fontsize=35)
ax.set_ylabel('Probability Density', fontsize=35)
#save figure
fig.tight_layout()
fig.savefig( fname_fig + '.png' )

#figure title
fname_fig = 'post_c1'
#create figure
fig, ax = plt.subplots(figsize = (10,10))
#plot examples
for df, n_iter in zip(df_post_stan_c1, n_iter_stan):
    ax.plot(df.x, df.y, linewidth=3, label=r'STAN, $n_{iter}=%i$'%n_iter)
ax.plot(df_post_inla_c1.x, df_post_inla_c1.y, linewidth=4,  linestyle='--', color='black', label=r'INLA')
#figure properties
ax.grid(which='both')
#tick size
ax.tick_params(axis='x', labelsize=32)
ax.tick_params(axis='y', labelsize=32)
#figure limits
ax.set_xlim([0.55, 0.75])
ax.set_ylim([0.0, 20.0])
#legend
pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width * 0.6, pos.height])
ax.legend(loc='center right', bbox_to_anchor=(2.0, 0.5), fontsize=32)
#labels
ax.set_xlabel(r'$c_1$',       fontsize=35)
ax.set_ylabel('Probability Density', fontsize=35)
#save figure
fig.tight_layout()
fig.savefig( fname_fig + '.png' )

#figure title
fname_fig = 'post_sigma'
#create figure
fig, ax = plt.subplots(figsize = (10,10))
#plot examples
for df, n_iter in zip(df_post_stan_sigma, n_iter_stan):
    ax.plot(df.x, df.y, linewidth=3, label=r'STAN, $n_{iter}=%i$'%n_iter)
ax.plot(df_post_inla_sigma.x, df_post_inla_sigma.y, linewidth=4,  linestyle='--', color='black', label=r'INLA')
#figure properties
ax.grid(which='both')
#tick size
ax.tick_params(axis='x', labelsize=32)
ax.tick_params(axis='y', labelsize=32)
#figure limits
ax.set_xlim([0.65, 0.75])
ax.set_ylim([0.0, 30.0])
#legend
pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width * 0.6, pos.height])
ax.legend(loc='center right', bbox_to_anchor=(2.0, 0.5), fontsize=32)
#labels
ax.set_xlabel(r'$\sigma$',       fontsize=35)
ax.set_ylabel('Probability Density', fontsize=35)
#save figure
fig.tight_layout()
fig.savefig( fname_fig + '.png' )
