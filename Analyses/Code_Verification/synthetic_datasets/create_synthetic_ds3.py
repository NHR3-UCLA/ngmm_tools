#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 15:47:17 2021

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
# %% set working directories
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
# cell data
# fname_cellinfo = 'CatalogNGAWest3CA_cellinfo.csv'
# fname_celldist = 'CatalogNGAWest3CA_distancematrix.csv'
# fname_cellinfo = 'CatalogNGAWest3CA_2013_cellinfo.csv'
# fname_celldist = 'CatalogNGAWest3CA_2013_distancematrix.csv'
fname_cellinfo    = 'CatalogNGAWest3CALite_cellinfo.csv'
fname_celldist    = 'CatalogNGAWest3CALite_distancematrix.csv'
fname_celldist_sp = 'CatalogNGAWest3CALite_distancematrix_sparce.csv'
dir_celldata = '../../../Data/Validation/preprocessing/cell_distances/'
# ++++++++++++++++++++++++++++++++++++++++

# USER SETS THE INPUT FLATFILE NAMES AND PATH
# ++++++++++++++++++++++++++++++++++++++++
fname_stan_model = 'create_synthetic_ds3.stan'
# ++++++++++++++++++++++++++++++++++++++++

# USER SETS THE OUTPUT FILE PATH AND NAME
# ++++++++++++++++++++++++++++++++++++++++
# output filename sufix
# synds_suffix = '_small_corr_len' 
# synds_suffix = '_large_corr_len'
# output directories
dir_out = f'../../../Data/Validation/synthetic_datasets/ds3{synds_suffix}/' 
# ++++++++++++++++++++++++++++++++++++++++

# number of synthetic data-sets
n_dataset  = 5
n_attempts = 500

# number of chains and seed number in stan model
n_chains = 1
n_seed   = 1

# define hyper-parameters
# omega_0:        standard deviation for constant offset
# omega_1e:       standard deviation for spatially varying earthquake constant
# omega_1as:      standard deviation for spatially varying site constant
# omega_1bs:      standard deviation for independent site constant
# ell_1e:         correlation length for spatially varying earthquake constant
# ell_1as:        correlation length for spatially varying site constant
# c_2_erg:        ergodic geometrical-spreading coefficient
# omega_2:        standard deviation for shift in average geometrical-spreading
# omega_2p:       standard deviation for spatially varying geometrical-spreading coefficient
# ell_2p:         correlation length for spatially varying geometrical-spreading coefficient
# c_3_erg:        ergodic Vs30 scaling coefficient
# omega_3:        standard deviation for shift in average Vs30 scaling 
# omega_3s:       standard deviation for spatially varying Vs30 scaling 
# ell_3s:         correlation length for spatially varying Vs30 scaling 
# c_cap_erg:      erogodic cell-specific anelastic attenuation
# omega_cap_mu:   standard deviation for constant offset of cell-specific anelastic attenuation
# omega_ca1p:     standard deviation for spatially varying component of cell-specific anelastic attenuation
# omega_ca2p:     standard deviation for spatially independent component of cell-specific anelastic attenuation
# ell_ca1p:       correlation length for spatially varying component of cell-specific anelastic attenuation
# phi_0:          within-event standard deviation
# tau_0:          between-event standard deviation

# USER NEEDS TO SPECIFY HYPERPARAMETERS
# ++++++++++++++++++++++++++++++++++++++++
# # small correlation lengths
# hyp = {'omega_0': 0.1, 'omega_1e':0.1, 'omega_1as': 0.35, 'omega_1bs': 0.25,
#         'ell_1e':60, 'ell_1as':30, 
#         'c_2_erg': -2.0, 
#         'omega_2': 0.2,
#         'omega_2p': 0.15, 'ell_2p': 80,
#         'c_3_erg':-0.6, 
#         'omega_3': 0.15,
#         'omega_3s': 0.15, 'ell_3s': 130,
#         'c_cap_erg': -0.011,
#         'omega_cap_mu': 0.005, 'omega_ca1p':0.004, 'omega_ca2p':0.002,
#         'ell_ca1p': 75,
#         'phi_0':0.3, 'tau_0':0.25 }
# # large correlation lengths
# hyp = {'omega_0': 0.1, 'omega_1e':0.2, 'omega_1as': 0.4, 'omega_1bs': 0.3,
#         'ell_1e':100, 'ell_1as':70, 
#         'c_2_erg': -2.0, 
#         'omega_2': 0.2,
#         'omega_2p': 0.15, 'ell_2p': 140,
#         'c_3_erg':-0.6, 
#         'omega_3': 0.15,
#         'omega_3s': 0.15, 'ell_3s': 180,
#         'c_cap_erg': -0.02,
#         'omega_cap_mu': 0.008, 'omega_ca1p':0.005, 'omega_ca2p':0.003,
#         'ell_ca1p': 120,
#         'phi_0':0.3, 'tau_0':0.25}
# ++++++++++++++++++++++++++++++++++++++++
#psuedo depth term for mag saturation
h_M = 4

# %% Load Data
# ======================================
# load flatfile
fullname_flatfile = dir_flatfile + fname_flatfile + '.csv'
df_flatfile = pd.read_csv(fullname_flatfile)

# load celldata
df_cell_dist    = pd.read_csv(dir_celldata + fname_celldist,    index_col=0)
df_cell_dist_sp = pd.read_csv(dir_celldata + fname_celldist_sp)
df_cell_info    = pd.read_csv(dir_celldata + fname_cellinfo)

# %% Processing
# ======================================
# read earthquake and station data from the flatfile
n_rec = len(df_flatfile)

# read earthquake data
# earthquake IDs (rqid), magnitudes (mag), and coordinates (eqX,eqY)
# user may change these IDs based on the headers of the flatfile
data_eq_all = df_flatfile[['eqid','mag','eqX', 'eqY']].values
_, eq_idx, eq_inv   = np.unique(df_flatfile[['eqid']], axis=0, return_index=True, return_inverse=True)
data_eq = data_eq_all[eq_idx,:]
X_eq = data_eq[:,[2,3]] # earthquake coordinates
# create earthquake ids for all recordings
eq_id = eq_inv + 1
n_eq = len(data_eq)

# read station data
# station IDs (ssn), Vs30, and coordinates (staX,staY)
# user may change these IDs based on the headers of the flatfile
data_sta_all = df_flatfile[['ssn','Vs30','staX','staY']].values
_, sta_idx, sta_inv = np.unique(df_flatfile[['ssn']].values, axis = 0, return_index=True, return_inverse=True)
data_sta = data_sta_all[sta_idx,:]
X_sta = data_sta[:,[2,3]] # station coordinates
# create station ids for all recordings
sta_id = sta_inv + 1
n_sta = len(data_sta)

# geometrical spreading covariate
x_2 = np.log(np.sqrt(df_flatfile.Rrup.values**2 + h_M**2))
#vs30 covariate
x_3 = np.log(np.minimum(data_sta[:,1], 1000)/1000)
assert(~np.isnan(x_3).all()),'Error. Invalid Vs30 values'

# read cell data
n_cell  = len(df_cell_info)
df_cell_dist = df_cell_dist.reindex(df_flatfile.rsn) #cell distance matrix for records in the synthetic data-set

# cell names
cells_names = df_cell_info.cellname.values
cells_id    = df_cell_info.cellid.values
# cell distance matrix
cell_dmatrix = df_cell_dist.loc[:,cells_names].values
# cell coordinates
X_cell = df_cell_info[['mptX','mptY']].values

# valid cells 
i_val_cells = cell_dmatrix.sum(axis=0) > 0

# %% Stan 
# ======================================

## Stan Data
# ---------------------------
stan_data = {'N':       n_rec,
             'NEQ':     n_eq,
             'NSTAT':   n_sta,
             'NCELL':   n_cell,
             'eq':      eq_id,          #earthquake index
             'stat':    sta_id,         #station index
             'X_e':     X_eq,           #earthquake coordinates
             'X_s':     X_sta,          #station coordinates
             'X_c':     X_cell,         #cell coordinates
             'RC':      cell_dmatrix,   #cell distances
             'mu_gmm':  np.zeros(n_rec),
             #covariates
             'x_2':     x_2,            #geometrical spreading
             'x_3':     x_3,            #Vs30 scaling
             #hyper-parameters of generated data-set
             'omega_0':   hyp['omega_0'],
             'omega_1e':  hyp['omega_1e'],
             'omega_1as': hyp['omega_1as'],
             'omega_1bs': hyp['omega_1bs'],
             'ell_1e':    hyp['ell_1e'],
             'ell_1as':   hyp['ell_1as'],
             'c_2_erg':   hyp['c_2_erg'],
             'omega_2':   hyp['omega_2'],
             'omega_2p':  hyp['omega_2p'],
             'ell_2p':    hyp['ell_2p'],
             'c_3_erg':   hyp['c_3_erg'],
             'omega_3':   hyp['omega_3'],
             'omega_3s':  hyp['omega_3s'],
             'ell_3s':    hyp['ell_3s'],
             #anelastic attenuation
             'c_cap_erg':  hyp['c_cap_erg'],
             'omega_cap_mu':  hyp['omega_cap_mu'],
             'omega_ca1p': hyp['omega_ca1p'],
             'omega_ca2p': hyp['omega_ca2p'],
             'ell_ca1p':   hyp['ell_ca1p'],
             #aleatory terms
             'phi_0':     hyp['phi_0'],
             'tau_0':     hyp['tau_0']
            }
  
## Compile and Run Stan model
# ---------------------------
# compile model
sm = pystan.StanModel(file=fname_stan_model)

# generate samples
fit = sm.sampling(data=stan_data, algorithm="Fixed_param", iter=n_attempts, chains=n_chains, seed=n_seed)

# select only data-sets with negative anelastic attenuation coefficients
valid_dataset = np.array( n_attempts * [False] )
for k, (c_2p, c_cap) in enumerate(zip(fit['c_2p'], fit['c_cap'])):
    valid_dataset[k] = np.all(c_2p <= 0 ) & np.all(c_cap <= 0 ) 
valid_dataset = np.where(valid_dataset)[0] #valid data-set ids
valid_dataset = valid_dataset[:min(n_dataset,len(valid_dataset))]

# keep valid datasets
Y_nerg_med  = fit['Y_nerg_med'][valid_dataset]
Y_var_coeff = fit['Y_var_ceoff'][valid_dataset]
Y_inattent  = fit['Y_inattent'][valid_dataset]
Y_aleat     = fit['Y_aleat'][valid_dataset]
Y_tot       = fit['Y_tot'][valid_dataset]
c_cap       = fit['c_cap'][valid_dataset]


# %% Output
# ======================================
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 

# save generated data-sets
for k, (k_vds, Y_nm, Y_vc, Y_iatt, Y_t) in enumerate(zip(valid_dataset, Y_nerg_med, Y_var_coeff, Y_inattent, Y_tot)):
    #copy catalog info to synthetic data-set
    df_synthetic_data = df_flatfile.copy()
    #add covariates
    df_synthetic_data.loc[:,'x_2']      = x_2
    df_synthetic_data.loc[:,'x_3']      = x_3[sta_inv]
    #add residuals columns
    df_synthetic_data.loc[:,'nerg_gm']  = Y_nm
    df_synthetic_data.loc[:,'vcoeff']   = Y_vc
    df_synthetic_data.loc[:,'inatten']  = Y_iatt
    df_synthetic_data.loc[:,'tot']      = Y_t
    #add columns with sampled coefficients
    df_synthetic_data.loc[:,'dc_0']   = fit['dc_0'][k_vds]
    df_synthetic_data.loc[:,'dc_1e']  = fit['dc_1e'][k_vds][eq_inv]
    df_synthetic_data.loc[:,'dc_1as'] = fit['dc_1as'][k_vds][sta_inv]
    df_synthetic_data.loc[:,'dc_1bs'] = fit['dc_1bs'][k_vds][sta_inv]
    df_synthetic_data.loc[:,'c_2']    = fit['c_2_mu'][k_vds]
    df_synthetic_data.loc[:,'c_2p']   = fit['c_2p'][k_vds][eq_inv]
    df_synthetic_data.loc[:,'c_3']    = fit['c_3_mu'][k_vds]
    df_synthetic_data.loc[:,'c_3s']   = fit['c_3s'][k_vds][sta_inv]
    #add columns aleatory terms
    df_synthetic_data.loc[:,'dW'] = fit['dW'][k_vds]
    df_synthetic_data.loc[:,'dB'] = fit['dB'][k_vds][eq_inv]
    #create data-frame with synthetic dataset
    fname_synthetic_data = dir_out + f'{fname_flatfile}_synthetic_data{synds_suffix}_Y{k+1}.csv'
    df_synthetic_data.to_csv(fname_synthetic_data, index=False)

# save coeffiicients
for k, (k_vds, c_ca) in enumerate(zip(valid_dataset, c_cap)):
    #create synthetic cell dataset
    df_synthetic_cell = df_cell_info.copy()
    #cell specific anelastic attenuation
    df_synthetic_cell.loc[:,'c_cap_mu'] = fit['c_cap_mu'][k_vds]
    df_synthetic_cell.loc[:,'c_cap']    = c_ca
    #create data-frame with cell specific dataset
    fname_synthetic_atten = dir_out + f'{fname_flatfile}_synthetic_atten{synds_suffix}_Y{k+1}.csv'
    df_synthetic_cell.to_csv(fname_synthetic_atten, index=False)   

# save cell data
fname_cell_info    = dir_out + f'{fname_flatfile}_cellinfo.csv'
fname_cell_dist    = dir_out + f'{fname_flatfile}_distancematrix.csv'
fname_cell_dist_sp = dir_out + f'{fname_flatfile}_distancematrix_sparse.csv'
df_cell_info.to_csv(fname_cell_info, index=False)
df_cell_dist.to_csv(fname_cell_dist)
df_cell_dist_sp.to_csv(fname_cell_dist_sp, index=False)


