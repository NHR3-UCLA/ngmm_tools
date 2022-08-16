#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 21:25:34 2021

@author: glavrent
"""
# load libraries
import os
import pathlib
# arithmetic libraries
import numpy as np
# statistics libraries
import pandas as pd
# python interface to Stan for Bayesian inference
# for installation check https://pystan.readthedocs.io/en/latest/
import pystan
# set working directories
os.chdir(os.getcwd()) # change directory to current directory

# %% Define Input Data
# ======================================
# USER SETS THE INPUT FLATFILE NAMES AND PATH
# ++++++++++++++++++++++++++++++++++++++++
#input flatfile
# fname_flatfile = 'CatalogNGAWest3CA'
# fname_flatfile = 'CatalogNGAWest3CA_2013'
# fname_flatfile = 'CatalogNGAWest3NCA'
# fname_flatfile = 'CatalogNGAWest3SCA'
fname_flatfile = 'CatalogNGAWest3CALite'
dir_flatfile   = '../../../Data/Validation/preprocessing/flatfiles/merged/'
# ++++++++++++++++++++++++++++++++++++++++

# USER SETS THE INPUT FLATFILE NAMES AND PATH
# ++++++++++++++++++++++++++++++++++++++++
fname_stan_model = 'create_synthetic_ds1.stan'
# ++++++++++++++++++++++++++++++++++++++++

# USER SETS THE OUTPUT FILE PATH AND NAME
# ++++++++++++++++++++++++++++++++++++++++
# output filename sufix
# synds_suffix = '_small_corr_len' 
# synds_suffix = '_large_corr_len'
# output directories
dir_out = f'../../../Data/Validation/synthetic_datasets/ds1{synds_suffix}/' 
# ++++++++++++++++++++++++++++++++++++++++

# user defines hyper parameters
# --------------------------------------
# number of synthetic data-sets
n_ds = 5

# number of chains and seed number in stan model
n_chains = 1
n_seed   = 1

# define hyper-parameters
# omega_0:    standard deviation for constant offset
# omega_1e:   standard deviation for spatially varying earthquake constant
# omega_1as:  standard deviation for spatially vayring site constant
# omega_1bs:  standard deviation for independent site constant
# ell_1e:     correlation lenght for spatially varying earthquake constant
# ell_1as:    correlation lenght for spatially vayring site constant
# phi_0:      within-event standard deviation
# tau_0:      between-event standard deviation

# USER NEEDS TO SPECIFY HYPERPARAMETERS
# ++++++++++++++++++++++++++++++++++++++++
# # small correlation lengths
# hyp = {'omega_0': 0.1, 'omega_1e':0.1, 'omega_1as': 0.35, 'omega_1bs': 0.25,
#        'ell_1e':60, 'ell_1as':30, 'phi_0':0.4, 'tau_0':0.3 }
#
# #large correlation lengths
# hyp = {'omega_0': 0.1, 'omega_1e':0.2, 'omega_1as': 0.4, 'omega_1bs': 0.3,
#         'ell_1e':100, 'ell_1as':70, 'phi_0':0.4, 'tau_0':0.3 }
# ++++++++++++++++++++++++++++++++++++++++


# %% Load Data
# ======================================
#load flatfile
fullname_flatfile = dir_flatfile + fname_flatfile + '.csv'
df_flatfile = pd.read_csv(fullname_flatfile)


# %% Processing
# ======================================
# read earthquake and station data from the flatfile
n_rec = len(df_flatfile)

# read earthquake data
# earthquake IDs (eqid), magnitudes (mag), and coordinates (eqX,eqY)
# user may change these IDs based on the headers of the flatfile
data_eq_all = df_flatfile[['eqid','mag','eqX', 'eqY']].values
_, eq_idx, eq_inv   = np.unique(df_flatfile[['eqid']], axis=0, return_index=True, return_inverse=True)
data_eq = data_eq_all[eq_idx,:]
X_eq = data_eq[:,[2,3]] #earthquake coordinates
# create earthquake ids for all recordings
eq_id = eq_inv + 1
n_eq = len(data_eq)

# read station data
# station IDs (ssn), Vs30, and coordinates (staX,staY)
# user may change these IDs based on the headers of the flatfile
data_stat_all = df_flatfile[['ssn','Vs30','staX','staY']].values
_, sta_idx, sta_inv = np.unique(df_flatfile[['ssn']].values, axis = 0, return_index=True, return_inverse=True)
data_stat = data_stat_all[sta_idx,:]
X_stat = data_stat[:,[2,3]] #station coordinates
# create station ids for all recordings
sta_id = sta_inv + 1
n_stat = len(data_stat)

# %% Stan 
# ======================================

## Stan Data
# ---------------------------
stan_data = {'N':       n_rec,
             'NEQ':     n_eq,
             'NSTAT':   n_stat,
             'X_e':     X_eq,           #earthquake coordinates
             'X_s':     X_stat,         #station coordinates
             'eq':      eq_id,          #earthquake index
             'stat':    sta_id,         #station index
             'mu_gmm':  np.zeros(n_rec),
             #hyper-parameters of generated data-set
             'omega_0':   hyp['omega_0'],
             'omega_1e':  hyp['omega_1e'],
             'omega_1as': hyp['omega_1as'],
             'omega_1bs': hyp['omega_1bs'],
             'ell_1e':    hyp['ell_1e'],
             'ell_1as':   hyp['ell_1as'],
             #aleatory terms
             'phi_0':     hyp['phi_0'],
             'tau_0':     hyp['tau_0']
            }


## Compile and Run Stan model
# ---------------------------
# compile model
sm = pystan.StanModel(file=fname_stan_model)

# generate samples
fit = sm.sampling(data=stan_data, algorithm="Fixed_param", iter=n_ds, chains=n_chains, seed=n_seed)

# keep valid datasets
Y_nerg_med = fit['Y_nerg_med']
Y_aleat    = fit['Y_aleat']
Y_tot      = fit['Y_tot']

# %% Output
# ======================================
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 

# save generated data-sets
for k, (Y_nm, Y_t) in enumerate(zip(Y_nerg_med, Y_tot)):
    #copy catalog info to synthetic data-set
    df_synthetic_data = df_flatfile.copy()
    #add residuals columns
    df_synthetic_data.loc[:,'nerg_gm']  = Y_nm
    df_synthetic_data.loc[:,'tot']      = Y_t
    #add columns with sampled coefficients
    df_synthetic_data.loc[:,'dc_0']   = fit['dc_0'][k]
    df_synthetic_data.loc[:,'dc_1e']  = fit['dc_1e'][k][eq_inv]
    df_synthetic_data.loc[:,'dc_1as'] = fit['dc_1as'][k][sta_inv]
    df_synthetic_data.loc[:,'dc_1bs'] = fit['dc_1bs'][k][sta_inv]
    #add columns aleatory terms
    df_synthetic_data.loc[:,'dW'] = fit['dW'][k]
    df_synthetic_data.loc[:,'dB'] = fit['dB'][k][eq_inv]
    #create data-frame with synthetic dataset
    fname_synthetic_data = dir_out + f'{fname_flatfile}_synthetic_data{synds_suffix}_Y{k+1}.csv'
    df_synthetic_data.to_csv(fname_synthetic_data, index=False)


