#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:17:52 2021

@author: glavrent
"""
# Working directory and Packages
# ---------------------------
#load libraries
import os
import sys
import numpy as np
import pandas as pd
import time
#user functions
sys.path.insert(0,'../../../Python_lib/regression/pystan/')
from regression_pystan_model1_unbounded_hyp import RunStan

# Define variables
# ---------------------------
#filename suffix
# synds_suffix = '_small_corr_len'
# synds_suffix = '_large_corr_len'

#synthetic datasets directory
ds_dir = '../../../../Data/Verification/synthetic_datasets/ds1'
ds_dir = r'%s%s/'%(ds_dir, synds_suffix)

# dataset info 
# ds_fname_main = 'CatalogNGAWest3CA_synthetic_data'
ds_fname_main = 'CatalogNGAWest3CALite_synthetic_data'
ds_id = np.arange(1,6)

#stan model 
# sm_fname = '../../../Stan_lib/regression_stan_model1_unbounded_hyp.stan'
# sm_fname = '../../../Stan_lib/regression_stan_model1_unbounded_hyp_chol.stan'
# sm_fname = '../../../Stan_lib/regression_stan_model1_unbounded_hyp_chol_efficient.stan'
# sm_fname = '../../../Stan_lib/regression_stan_model1_unbounded_hyp_chol_efficient2.stan'

#output info
#main output filename
out_fname_main = 'NGAWest2CA_syndata'
#main output directory
out_dir_main   = '../../../../Data/Verification/regression/ds1/'
#output sub-directory
#python 2
# out_dir_sub    = 'PYSTAN_NGAWest2CA'
# out_dir_sub    = 'PYSTAN_NGAWest2CA_chol'
# out_dir_sub    = 'PYSTAN_NGAWest2CA_chol_eff'
# out_dir_sub    = 'PYSTAN_NGAWest2CA_chol_eff2'
#python 3
# out_dir_sub    = 'PYSTAN3_NGAWest2CA'
# out_dir_sub    = 'PYSTAN3_NGAWest2CA_chol'
# out_dir_sub    = 'PYSTAN3_NGAWest2CA_chol_eff'
# out_dir_sub    = 'PYSTAN3_NGAWest2CA_chol_eff2'

#stan parameters
runstan_flag = True
# pystan_ver = 2
# pystan_ver = 3
res_name = 'tot'
n_iter   = 1000
n_chains = 4
adapt_delta   = 0.8
max_treedepth = 10
#parallel options
# flag_parallel = True
flag_parallel = False

#output sub-dir with corr with suffix info
out_dir_sub = f'%s%s'%(out_dir_sub, synds_suffix)

# Run stan regression
# ---------------------------
#create datafame with computation time
df_run_info = list()

#iterate over all synthetic datasets
for d_id in ds_id:
    print('Synthetic dataset %i fo %i'%(d_id, len(ds_id)))
    #run time start
    run_t_strt = time.time()    
    #input flatfile
    ds_fname = '%s%s%s_Y%i.csv'%(ds_dir, ds_fname_main, synds_suffix, d_id)
    #load flatfile
    df_flatfile = pd.read_csv(ds_fname)
    #keep only NGAWest2 records
    df_flatfile = df_flatfile.loc[df_flatfile.dsid==0,:]
    
    #output file name and directory
    out_fname = '%s%s_Y%i'%(out_fname_main, synds_suffix, d_id)
    out_dir   = '%s/%s/Y%i/'%(out_dir_main, out_dir_sub, d_id)

    #run stan model
    RunStan(df_flatfile, sm_fname, out_fname, out_dir, res_name, 
            runstan_flag=runstan_flag, n_iter=n_iter, n_chains=n_chains,
            adapt_delta=adapt_delta, max_treedepth=max_treedepth,
            pystan_ver=pystan_ver, pystan_parallel=flag_parallel)
    
    #run time end
    run_t_end = time.time()

    #compute run time
    run_tm = (run_t_end - run_t_strt)/60
  
    #log run time
    df_run_info.append(pd.DataFrame({'computer_name':os.uname()[1],'out_name':out_dir_sub,
                                     'ds_id':d_id,'run_time':run_tm}, index=[d_id]))
                           
    #write out run info
    out_fname   = '%s%s/run_info.csv'%(out_dir_main, out_dir_sub)
    pd.concat(df_run_info).reset_index(drop=True).to_csv(out_fname, index=False)

