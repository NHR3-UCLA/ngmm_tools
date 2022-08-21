#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 14:54:54 2022

@author: glavrent
"""

# Packages
# ---------------------------
#arithmetic libraries
import numpy as np
from scipy import linalg as scipylinalg 
from sklearn.gaussian_process.kernels import Matern
#user functions
import pylib_kernels as pylib_kern
import pylib_cell_dist as pylib_cells

# Non-ergodic GMM effects prediction
# ---------------------------
def PredictNErgEffects(n_samp, nerg_coeff_info, df_scen_predict,  df_nerg_coeffs, 
                               nerg_catten_info=None, df_cell_info=None, df_nerg_cellatten=None):
    '''
    Predict non-egodic ground motion effects 

    Parameters
    ----------
    n_samp : int
        Number of samples.
    nerg_coeff_info : dict
        Non-ergodic coefficient information dictionary.
    df_scen_predict : pd.dataframe
        Prediction scenarios.
    df_nerg_coeffs : pd.dataframe
        Regressed non-ergodic coefficients .
    nerg_catten_info : dict, optional
        cell-specific anelastic attenuation information dictionary. The default is None.
    df_cell_info : pd.dataframe, optional
        Cell info dataframe. The default is None.
    df_nerg_cellatten : pd.dataframe, optional
        Regressed anelastic attenuation coefficients. The default is None.

    Returns
    -------
    nerg_effects_prdct_samp : np.array
        Samples of total non-ergodic effects.
    nerg_vcm_prdct_samp : TYPE
        Samples of spatially varying component of non-ergodic effects.
    nerg_atten_prdct_samp : TYPE
        Samples of anelastic attenuation component of non-ergodic effects.
    nerg_effects_prdct_mu : TYPE
        Mean of total non-ergodic effects.
    nerg_effects_prdct_sig : TYPE
        Standard deviation of total non-ergodic effects.
    nerg_vcm_cmp : list
        List with individual components of spatially varying non-ergodic effects.
    nerg_atten_cmp : list
        List with individual components of anelast attenuation.
    '''
    
    #number of prediction scenarios
    n_predict = len(df_scen_predict)
    
    # VCM component
    # ---   ---   ---   ---
    #initialize vcm samples
    nerg_vcm_prdct_samp = np.zeros(shape=(n_predict,n_samp))
    nerg_vcm_prdct_mu   = np.zeros(shape=n_predict)
    nerg_vcm_prdct_var  = np.zeros(shape=n_predict)
    nerg_vcm_cmp = {}

    #iterate over non-ergodic coefficients
    for nerg_c in nerg_coeff_info:
        #kernel type
        k_type = nerg_coeff_info[nerg_c]['kernel_type']

        #hyper-parameters
        if 'hyp' in nerg_coeff_info[nerg_c]:
            hyp_param =  nerg_coeff_info[nerg_c]['hyp']
            hyp_mean_c = hyp_param['mean_c']  if (('mean_c' in hyp_param) and (not hyp_param['mean_c'] is None)) else 0 
            hyp_ell    = hyp_param['ell']     if (('ell'    in hyp_param) and (not hyp_param['ell']    is None)) else 0 
            hyp_omega  = hyp_param['omega']   if (('omega'  in hyp_param) and (not hyp_param['omega']  is None)) else 0 
            hyp_pi     = hyp_param['pi']      if (('pi'     in hyp_param) and (not hyp_param['pi']     is None)) else 0    
            hyp_nu     = hyp_param['nu']      if (('nu'     in hyp_param) and (not hyp_param['nu']     is None)) else 0    
     
        #mean and std of non-ergodic coefficients at known locations
        c_mean_train = df_nerg_coeffs.loc[:,nerg_coeff_info[nerg_c]['coeff'][0]].values
        c_sig_train  = df_nerg_coeffs.loc[:,nerg_coeff_info[nerg_c]['coeff'][1]].values
        #non-ergodic coefficient scaling
        c_scl  = np.ones(n_predict) if nerg_coeff_info[nerg_c]['scaling'] is None else df_nerg_coeffs.loc[:,nerg_coeff_info[nerg_c]['scaling']].values

        if k_type == 0: #constan
            assert(len(np.unique(c_mean_train))==1)
            #mean and std of non-ergodic coefficient
            c_mean_train = c_mean_train[0]
            c_sig_train  = c_sig_train[0]
            #draw random samples
            c_prdct_samp = np.random.normal(loc=c_mean_train, scale=c_sig_train, size=n_samp)
            #sample non-ergodic coefficient for prediction scenarios
            c_prdct_samp  = np.full((n_predict,n_samp), c_prdct_samp)
            #mean and sigma
            c_prdct_mu  = np.full(n_predict, c_mean_train)
            c_prdct_sig = np.full(n_predict, c_sig_train)
            
        if k_type == 1: #group
            #group ids in training data
            id_train = df_nerg_coeffs.loc[:,nerg_coeff_info[nerg_c]['cor_info']].values
            id_train, idx_train = np.unique(id_train, axis=0, return_index=True)
            #group ids in prediction data
            id_prdct = df_scen_predict.loc[:,nerg_coeff_info[nerg_c]['cor_info']].values
            id_prdct, inv_prdct = np.unique(id_prdct, axis=0, return_inverse=True)
            #mean and std of non-ergodic coefficient
            c_mean_train = c_mean_train[idx_train]
            c_sig_train  = c_sig_train[idx_train]
            #compute mean and cov of non-erg coeffs for prediction scenarios
            c_prdct_mu, _, c_prdct_cov = pylib_kern.PredictGroupKern(id_prdct, id_train, 
                                                          c_train_mu=c_mean_train, c_train_sig=c_sig_train,  
                                                          hyp_mean_c=hyp_mean_c, hyp_omega=hyp_omega)
            #sample non-ergodic coefficient for prediction scenarios
            c_prdct_samp = MVNRnd(mean=c_prdct_mu, cov=c_prdct_cov, n_samp=n_samp)
            c_prdct_samp = c_prdct_samp[inv_prdct,:]
            #mean and sigma
            c_prdct_mu  = c_prdct_mu[inv_prdct]
            c_prdct_sig = np.sqrt( np.diag(c_prdct_cov) )[inv_prdct]

        if k_type == 2: #exponetial
            #coordinates of training data
            t_train = df_nerg_coeffs.loc[:,nerg_coeff_info[nerg_c]['cor_info']].values
            t_train, idx_train = np.unique(t_train, axis=0, return_index=True)
            #coordinates of prediction data
            t_prdct = df_scen_predict.loc[:,nerg_coeff_info[nerg_c]['cor_info']].values
            t_prdct, inv_prdct = np.unique(t_prdct, axis=0, return_inverse=True)
            #mean and std of non-ergodic coefficient
            c_mean_train = c_mean_train[idx_train]
            c_sig_train  = c_sig_train[idx_train]
            #compute mean and cov of non-erg coeffs for prediction scenarios
            c_prdct_mu, _, c_prdct_cov = pylib_kern.PredictExpKern(t_prdct, t_train,
                                                                   c_train_mu=c_mean_train, c_train_sig=c_sig_train,  
                                                                   hyp_mean_c=hyp_mean_c, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi)
            #sample non-ergodic coefficient for prediction scenarios
            c_prdct_samp = MVNRnd(mean=c_prdct_mu, cov=c_prdct_cov, n_samp=n_samp)
            c_prdct_samp = c_prdct_samp[inv_prdct,:]
            #mean and sigma
            c_prdct_mu  = c_prdct_mu[inv_prdct]
            c_prdct_sig = np.sqrt( np.diag(c_prdct_cov) )[inv_prdct]

        if k_type == 3: #squared exponetial
            #coordinates of training data
            t_train = df_nerg_coeffs.loc[:,nerg_coeff_info[nerg_c]['cor_info']].values
            t_train, idx_train = np.unique(t_train, axis=0, return_index=True)
            #coordinates of prediction data
            t_prdct = df_scen_predict.loc[:,nerg_coeff_info[nerg_c]['cor_info']].values
            t_prdct, inv_prdct = np.unique(t_prdct, axis=0, return_inverse=True)
            #mean and std of non-ergodic coefficient
            c_mean_train = c_mean_train[idx_train]
            c_sig_train  = c_sig_train[idx_train]
            #compute mean and cov of non-erg coeffs for prediction scenarios
            c_prdct_mu, _, c_prdct_cov = pylib_kern.PredictSqExpKern(t_prdct, t_train, 
                                                                     c_train_mu=c_mean_train, c_train_sig=c_sig_train,  
                                                                     hyp_mean_c=hyp_mean_c, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi)
            #sample non-ergodic coefficient for prediction scenarios
            c_prdct_samp = MVNRnd(mean=c_prdct_mu, cov=c_prdct_cov, n_samp=n_samp)
            c_prdct_samp = c_prdct_samp[inv_prdct,:]
            #mean and sigma
            c_prdct_mu  = c_prdct_mu[inv_prdct]
            c_prdct_sig = np.sqrt( np.diag(c_prdct_cov) )[inv_prdct]
            
        if k_type == 4: #Matern kernel function
            #coordinates of training data
            t_train = df_nerg_coeffs.loc[:,nerg_coeff_info[nerg_c]['cor_info']].values
            t_train, idx_train = np.unique(t_train, axis=0, return_index=True)
            #coordinates of prediction data
            t_prdct = df_scen_predict.loc[:,nerg_coeff_info[nerg_c]['cor_info']].values
            t_prdct, inv_prdct = np.unique(t_prdct, axis=0, return_inverse=True)
            #mean and std of non-ergodic coefficient
            c_mean_train = c_mean_train[idx_train]
            c_sig_train  = c_sig_train[idx_train]
            #compute mean and cov of non-erg coeffs for prediction scenarios
            c_prdct_mu, _, c_prdct_cov = pylib_kern.PredictMaternKern(t_prdct, t_train, 
                                                                      c_train_mu=c_mean_train, c_train_sig=c_sig_train,  
                                                                      hyp_mean_c=hyp_mean_c, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi,
                                                                      hyp_nu=hyp_nu)
            #sample non-ergodic coefficient for prediction scenarios
            c_prdct_samp = MVNRnd(mean=c_prdct_mu, cov=c_prdct_cov, n_samp=n_samp)
            c_prdct_samp = c_prdct_samp[inv_prdct,:]
            #mean and sigma
            c_prdct_mu  = c_prdct_mu[inv_prdct]
            c_prdct_sig = np.sqrt( np.diag(c_prdct_cov) )[inv_prdct]

        #add contribution of non-ergodic effect
        nerg_vcm_prdct_samp += c_scl[:,np.newaxis] * c_prdct_samp
        #mean and std contribution of non-ergodic effect 
        nerg_vcm_prdct_mu  += c_scl * c_prdct_mu
        nerg_vcm_prdct_var += c_scl**2 * c_prdct_sig**2
        #summarize individual components
        nerg_vcm_cmp[nerg_c] = [c_scl * c_prdct_mu, c_scl * c_prdct_sig, c_scl[:,np.newaxis] * c_prdct_samp]
        
    # Anelastic attenuation
    # ---   ---   ---   ---
    #initialize anelastic attenuation
    nerg_atten_prdct_samp = np.zeros(shape=(n_predict,n_samp))
    nerg_atten_prdct_mu   = np.zeros(shape=n_predict)
    nerg_atten_prdct_var  = np.zeros(shape=n_predict)
    nerg_atten_cmp   = {}
    
    if not nerg_catten_info is None:
        #cell edge coordinates for path seg calculation
        ct4dist = df_cell_info.loc[:,['q1X', 'q1Y', 'q1Z', 'q8X', 'q8Y', 'q8Z']].values
        #cell limts
        c_lmax = ct4dist[:,[3,4,5]].max(axis=0)
        c_lmin = ct4dist[:,[0,1,2]].min(axis=0)
        #compute cell-path
        cell_path  = np.zeros([n_predict, len(df_cell_info)])
        for j, (rsn, scn_p) in enumerate(df_scen_predict.iterrows()):
            pt1 = scn_p[['eqX','eqY','eqZ']].values.astype(float)
            pt2 = np.hstack([scn_p[['staX','staY']].values, 0]).astype(float)
            #check limits
            assert(np.logical_and(pt1>=c_lmin, pt1<=c_lmax).all()),'Error. Eq outside cell domain for rsn: %i'%rsn
            assert(np.logical_and(pt2>=c_lmin, pt2<=c_lmax).all()),'Error. Sta outside cell domain for rsn: %i'%rsn
            #cell paths for pt1 - pt2  
            cell_path[j,:] = pylib_cells.ComputeDistGridCells(pt1,pt2,ct4dist, flagUTM=True)
            
            
        #keep only cells with non-zero paths
        ca_valid = cell_path.sum(axis=0) > 0
        cell_path    = cell_path[:,ca_valid]
        df_cell_info = df_cell_info.loc[ca_valid,:]
        
        #iterate over anelastic attenuation components
        for nerg_ca in nerg_catten_info:
            #kernel type
            k_type = nerg_catten_info[nerg_ca]['kernel_type']
            
            #mean and std anelastic attenuation cells
            ca_mean_train = df_nerg_cellatten.loc[:,nerg_catten_info[nerg_ca]['catten'][0]].values
            ca_sig_train  = df_nerg_cellatten.loc[:,nerg_catten_info[nerg_ca]['catten'][1]].values
        
            #hyper-parameters
            hyp_param = nerg_catten_info[nerg_ca]['hyp']
            hyp_mean_ca = hyp_param['mean_ca'] if (('mean_ca' in hyp_param) and (not hyp_param['mean_ca'] is None)) else 0
            hyp_ell     = hyp_param['ell']     if (('ell'     in hyp_param) and (not hyp_param['ell']   is None)) else 0
            hyp_ell1    = hyp_param['ell1']    if (('ell1'    in hyp_param) and (not hyp_param['ell1']  is None)) else np.nan 
            hyp_ell2    = hyp_param['ell2']    if (('ell2'    in hyp_param) and (not hyp_param['ell2']  is None)) else np.nan 
            hyp_omega   = hyp_param['omega']   if (('omega'   in hyp_param) and (not hyp_param['omega']   is None)) else 0
            hyp_omega1  = hyp_param['omega1']  if (('omega1'  in hyp_param) and (not hyp_param['omega1']  is None)) else np.nan 
            hyp_omega2  = hyp_param['omega2']  if (('omega2'  in hyp_param) and (not hyp_param['omega2']  is None)) else np.nan 
            hyp_pi      = hyp_param['pi']      if (('pi'      in hyp_param) and (not hyp_param['pi']      is None)) else 0
            hyp_nu      = hyp_param['nu']      if (('nu'      in hyp_param) and (not hyp_param['nu']     is None)) else 0    

            #select kernel function
            if k_type == 1: #independent cells 
                #cell ids in training data
                # cid_train = df_nerg_cellatten.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                cid_train = df_nerg_cellatten.index.values
                #cell ids in prediction data
                # cid_prdct = df_cell_info.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                cid_prdct = df_cell_info.index.values
                #compute mean and cov of cell anelastic coeffs for prediction scenarios
                ca_prdct_mu, _, ca_prdct_cov = pylib_kern.PredictGroupKern(cid_prdct, cid_train,
                                                                           c_train_mu=ca_mean_train, c_train_sig=ca_sig_train,  
                                                                           hyp_mean_c=hyp_mean_ca , hyp_omega=hyp_omega)
            if k_type == 2: #exponetial
                #cell coordinates of training data
                ct_train = df_nerg_cellatten.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                #cell coordinates of prediction data
                ct_prdct = df_cell_info.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                #compute mean and cov of cell anelastic coeffs for prediction scenarios
                ca_prdct_mu, _, ca_prdct_cov = pylib_kern.PredictExpKern(ct_prdct, ct_train,
                                                                         c_train_mu=ca_mean_train, c_train_sig=ca_sig_train,  
                                                                         hyp_mean_c=hyp_mean_ca, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi)
            if k_type == 3: #squared exponetial
                #cell coordinates of training data
                ct_train = df_nerg_cellatten.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                #cell coordinates of prediction data
                ct_prdct = df_cell_info.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                #compute mean and cov of cell anelastic coeffs for prediction scenarios
                ca_prdct_mu, _, ca_prdct_cov = pylib_kern.PredictSqExpKern(ct_prdct, ct_train, 
                                                                           c_train_mu=ca_mean_train, c_train_sig=ca_sig_train,  
                                                                           hyp_mean_c=hyp_mean_ca, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi)
            if k_type == 4: #Matern 
                #cell coordinates of training data
                ct_train = df_nerg_cellatten.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                #cell coordinates of prediction data
                ct_prdct = df_cell_info.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                #compute mean and cov of cell anelastic coeffs for prediction scenarios
                ca_prdct_mu, _, ca_prdct_cov = pylib_kern.PredictSqMaternKern(ct_prdct, ct_train, 
                                                                              c_train_mu=ca_mean_train, c_train_sig=ca_sig_train,  
                                                                              hyp_mean_c=hyp_mean_ca, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi)
            if k_type == 5: #exponetial and spatially independent composite
                #cell coordinates of training data
                ct_train = df_nerg_cellatten.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                #cell coordinates of prediction data
                ct_prdct = df_cell_info.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                #compute mean and cov of cell anelastic coeffs for prediction scenarios
                ca_prdct_mu, _, ca_prdct_cov = pylib_kern.PredictNegExpSptInptKern(ct_prdct, ct_train, 
                                                                                   c_train_mu=ca_mean_train, c_train_sig=ca_sig_train,  
                                                                                   hyp_mean_c=hyp_mean_ca, hyp_ell1=hyp_ell1, hyp_omega1=hyp_omega1, 
                                                                                   hyp_omega2=hyp_omega2, hyp_pi=hyp_pi)
            
    
            #sample cell-specific anelastic coefficients for prediction scenarios
            ca_prdct_samp = MVNRnd(mean=ca_prdct_mu, cov=ca_prdct_cov, n_samp=n_samp)
            ca_prdct_sig  = np.sqrt( np.diag(ca_prdct_cov) )
            
            #effect of anelastic attenuation
            nerg_atten_prdct_samp += cell_path @ ca_prdct_samp
            nerg_atten_prdct_mu   += cell_path @ ca_prdct_mu
            nerg_atten_prdct_var  += np.square(cell_path) @ ca_prdct_sig**2
            #summarize individual anelastic components
            nerg_atten_cmp[nerg_ca] = [cell_path @ ca_prdct_mu, np.sqrt(np.square(cell_path) @ ca_prdct_sig**2),
                                       cell_path @ ca_prdct_samp]

    #total non-ergodic effects
    nerg_effects_prdct_samp = nerg_vcm_prdct_samp + nerg_atten_prdct_samp
    nerg_effects_prdct_mu   = nerg_vcm_prdct_mu   + nerg_atten_prdct_mu
    nerg_effects_prdct_sig  = np.sqrt(nerg_vcm_prdct_var  + nerg_atten_prdct_var)
    
    return nerg_effects_prdct_samp, nerg_vcm_prdct_samp, nerg_atten_prdct_samp, \
           nerg_effects_prdct_mu, nerg_effects_prdct_sig, \
           nerg_vcm_cmp, nerg_atten_cmp


# Multivariate normal distribution random samples
#--------------------------------------
def MVNRnd(mean = None, cov = None, seed = None, n_samp = None, flag_sp = False, flag_list = False):
    '''
    Draw random samples from a Multivariable Normal distribution

    Parameters
    ----------
    mean : np.array(n), optional
        Mean array. The default is None.
    cov : np.array(n,n), optional
        Covariance Matrix. The default is None.
    seed : int, optional
        Seed number of random number generator. The default is None.
    n_samp : int, optional
        Number of samples. The default is None.
    flag_sp : boolean, optional
        Sparse covariance matrix flag; if sparse flag_sp = True. The default is False.
    flag_list : boolean, optional
        Flag returning output as list. The default is False.

    Returns
    -------
    samp
        Sampled values.
    '''
    
    #if not already covert to list
    if flag_list:
        seed_list = seed if not seed is None else [None]
    else:
        seed_list = [seed]
        
    #number of dimensions
    n_dim = len(mean) if not mean is None else cov.shape[0]
    assert(cov.shape == (n_dim,n_dim)),'Error. Inconsistent size of mean array and covariance matrix'
        
    #set mean array to zero if not given
    if mean is None: mean = np.zeros(n_dim)

    #compute L D L' decomposition
    if flag_sp: cov = cov.toarray()
    L, D, _ = scipylinalg.ldl(cov)
    assert( not np.count_nonzero(D - np.diag(np.diagonal(D))) ),'Error. D not diagonal'
    assert( np.all(np.diag(D) > -1e-1) ),'Error. D diagonal is negative'
    #extract diagonal from D matrix, set to zero any negative entries due to bad conditioning
    d      = np.diagonal(D).copy()
    d[d<0] = 0
    #compute Q matrix
    Q = L @ np.diag(np.sqrt(d))

    #generate random sample
    samp_list = list()
    for k, seed in enumerate(seed_list):
        #genereate seed numbers if not given 
        if seed is None: seed = np.random.standard_normal(size=(n_dim, n_samp))
   
        #generate random multi-normal random samples
        samp = Q @ (seed )
        samp += mean[:,np.newaxis] if samp.ndim > 1 else mean
        
        #summarize samples
        samp_list.append( samp )

    return samp_list if flag_list else samp_list[0]
