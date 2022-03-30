#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 08:22:49 2022

@author: glavrent
"""

# Packages
# ---------------------------
#arithmetic libraries
import numpy as np
from scipy import linalg as scipylinalg 
#user functions
import pylib_cell_dist as pylib_cells

# Non-ergodic GMM effects prediction
# ---------------------------
def PredictNErgEffects(n_samp, nerg_coeff_info, df_scen_predict,  df_nerg_coeffs, 
                               nerg_catten_info=None, df_cell_info=None, df_nerg_cellatten=None):
    
    #number of prediction scenarios
    n_predict = len(df_scen_predict)
    
    # VCM component
    # ---   ---   ---   ---
    #initialize vcm samples
    nerg_vcm_prdct = np.zeros(shape=(n_predict,n_samp))
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
            c_prdct_mu, _, c_prdct_cov = PredictGroupCoeff(id_prdct, id_train, 
                                                           c_train_mu=c_mean_train, c_train_sig=c_sig_train,  
                                                           hyp_mean_c=hyp_mean_c, hyp_omega=hyp_omega)
            #sample non-ergodic coefficient for prediction scenarios
            c_prdct_samp = MVNRnd(mean=c_prdct_mu, cov=c_prdct_cov, n_samp=n_samp)
            c_prdct_samp = c_prdct_samp[inv_prdct,:]
            #mean and sigma
            c_prdct_mu  = c_prdct_mu[inv_prdct]
            c_prdct_sig = np.sqrt( np.diag(c_prdct_cov) )[inv_prdct]

        if k_type == 2: #negative exponetial
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
            c_prdct_mu, _, c_prdct_cov = PredictNegExp(t_prdct, t_train,
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
            c_prdct_mu, _, c_prdct_cov = PredictSqExp(t_prdct, t_train, 
                                                      c_train_mu=c_mean_train, c_train_sig=c_sig_train,  
                                                      hyp_mean_c=hyp_mean_c, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi)
            #sample non-ergodic coefficient for prediction scenarios
            c_prdct_samp = MVNRnd(mean=c_prdct_mu, cov=c_prdct_cov, n_samp=n_samp)
            c_prdct_samp = c_prdct_samp[inv_prdct,:]
            #mean and sigma
            c_prdct_mu  = c_prdct_mu[inv_prdct]
            c_prdct_sig = np.sqrt( np.diag(c_prdct_cov) )[inv_prdct]

        #add contribution to non-ergodic effects
        nerg_vcm_prdct += c_scl[:,np.newaxis] * c_prdct_samp
        #summarize individual components
        nerg_vcm_cmp[nerg_c] = [c_scl * c_prdct_mu, c_scl * c_prdct_sig, c_scl[:,np.newaxis] * c_prdct_samp]
        
    # Anelastic attenuation
    # ---   ---   ---   ---
    #initialize anelastic attenuation
    nerg_atten_prdct = np.zeros(shape=(n_predict,n_samp))
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
            pt1 = scn_p[['eqX','eqY','eqZ']].values
            pt2 = np.hstack([scn_p[['staX','staY']].values, 0])
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
            hyp_ell     = hyp_param['ell']     if (('ell'   in hyp_param) and (not hyp_param['ell']   is None)) else 0
            hyp_ell1    = hyp_param['ell1']    if (('ell1'  in hyp_param) and (not hyp_param['ell1']  is None)) else np.nan 
            hyp_ell2    = hyp_param['ell2']    if (('ell2'  in hyp_param) and (not hyp_param['ell2']  is None)) else np.nan 
            hyp_omega   = hyp_param['omega']   if (('omega'   in hyp_param) and (not hyp_param['omega']   is None)) else 0
            hyp_omega1  = hyp_param['omega1']  if (('omega1'  in hyp_param) and (not hyp_param['omega1']  is None)) else np.nan 
            hyp_omega2  = hyp_param['omega2']  if (('omega2'  in hyp_param) and (not hyp_param['omega2']  is None)) else np.nan 
            hyp_pi      = hyp_param['pi']      if (('pi'      in hyp_param) and (not hyp_param['pi']      is None)) else 0

            #select kernel function
            if k_type == 1: #independent cells 
                #cell ids in training data
                # cid_train = df_nerg_cellatten.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                cid_train = df_nerg_cellatten.index.values
                #cell ids in prediction data
                # cid_prdct = df_cell_info.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                cid_prdct = df_cell_info.index.values
                #compute mean and cov of cell anelastic coeffs for prediction scenarios
                ca_prdct_mu, _, ca_prdct_cov = PredictGroupCoeff(cid_prdct, cid_train,
                                                                 c_train_mu=ca_mean_train, c_train_sig=ca_sig_train,  
                                                                 hyp_mean_c=hyp_mean_ca , hyp_omega=hyp_omega)
            if k_type == 2: #negative exponetial
                #cell coordinates of training data
                ct_train = df_nerg_cellatten.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                #cell coordinates of prediction data
                ct_prdct = df_cell_info.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                #compute mean and cov of cell anelastic coeffs for prediction scenarios
                ca_prdct_mu, _, ca_prdct_cov = PredictNegExp(ct_prdct, ct_train,
                                                             c_train_mu=ca_mean_train, c_train_sig=ca_sig_train,  
                                                             hyp_mean_c=hyp_mean_ca, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi)
            if k_type == 3: #squared exponetial
                #cell coordinates of training data
                ct_train = df_nerg_cellatten.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                #cell coordinates of prediction data
                ct_prdct = df_cell_info.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                #compute mean and cov of cell anelastic coeffs for prediction scenarios
                ca_prdct_mu, _, ca_prdct_cov = PredictSqExp(ct_prdct, ct_train, 
                                                             c_train_mu=ca_mean_train, c_train_sig=ca_sig_train,  
                                                             hyp_mean_c=hyp_mean_ca, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi)
            if k_type == 4: #spatially varying and spatially independent component
                #cell coordinates of training data
                ct_train = df_nerg_cellatten.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                #cell coordinates of prediction data
                ct_prdct = df_cell_info.loc[:,nerg_catten_info[nerg_ca]['cor_info']].values
                #compute mean and cov of cell anelastic coeffs for prediction scenarios
                ca_prdct_mu, _, ca_prdct_cov = PredictNegExpSptInpt(ct_prdct, ct_train, 
                                                                    c_train_mu=ca_mean_train, c_train_sig=ca_sig_train,  
                                                                    hyp_mean_c=hyp_mean_ca, hyp_ell1=hyp_ell1, hyp_omega1=hyp_omega1, 
                                                                    hyp_omega2=hyp_omega2, hyp_pi=hyp_pi)
                
    
            #sample cell-specific anelastic coefficients for prediction scenarios
            ca_prdct_samp = MVNRnd(mean=ca_prdct_mu, cov=ca_prdct_cov, n_samp=n_samp)
            ca_prdct_sig  = np.sqrt( np.diag(ca_prdct_cov) )
            
            #effect of anelastic attenuation
            nerg_atten_prdct += cell_path @ ca_prdct_samp
            #summarize individual anelastic components
            nerg_atten_cmp[nerg_ca] = [cell_path @ ca_prdct_mu, np.sqrt(np.square(cell_path) @ ca_prdct_sig**2),
                                       cell_path @ ca_prdct_samp]

    #total non-ergodic effects
    nerg_effects_prdct = nerg_vcm_prdct + nerg_atten_prdct
    
    return nerg_effects_prdct, nerg_vcm_prdct, nerg_atten_prdct, nerg_vcm_cmp, nerg_atten_cmp


# Kernel Functions
#--------------------------------------
# group kernel function
#---  ---  ---  ---  ---  ---  ---  ---
def KernelGroup(grp_1, grp_2, hyp_omega = 0, delta = 1e-9):
    '''
    Compute kernel function for perfect correlation between group variables

    Parameters
    ----------
    grp_1 : np.array
        IDs for first group.
    grp_2 : np.array
        IDs for second group.
    hyp_omega : non-negative real, optional
        Scale of kernel function. The default is 0.
    delta : non-negative real, optional
        Diagonal widening. The default is 1e-9.

    Returns
    -------
    cov_mat : np.array
        Covariance Matrix.

    '''
    
    #tolerance for  station id comparison
    r_tol = np.min([0.01/np.max([np.abs(grp_1).max(), np.abs(grp_2).max()]), 1e-11])
    #number of grid nodes
    n_pt_1 = grp_1.shape[0]
    n_pt_2 = grp_2.shape[0]
    #number of dimensions
    n_dim = grp_1.ndim
    #create cov. matrix
    cov_mat = np.zeros([n_pt_1,n_pt_2]) #initialize
    if n_dim == 1:
        for i in range(n_pt_1):
            cov_mat[i,:] =  hyp_omega**2 * np.isclose(grp_1[i], grp_2, rtol=r_tol).flatten()
    else:
        for i in range(n_pt_1):
            cov_mat[i,:] =  hyp_omega**2 * (scipylinalg.norm(grp_1[i] - grp_2, axis=1) < r_tol)
        

    if n_pt_1 == n_pt_2:
        for i in range(n_pt_1):
            cov_mat[i,i] += delta

    return cov_mat

# negative exponential kernel
#---  ---  ---  ---  ---  ---  ---  ---
def KernelNegExp(t_1, t_2, hyp_ell = 0, hyp_omega = 0, hyp_pi = 0, delta = 1e-9):
    '''
    Compute negative exponential kernel function

    Parameters
    ----------
    t_1 : np.array
        Coordinates of first group.
    t_2 : np.array
        Coordinates of second group.
    hyp_ell : non-negative real, optional
        Correlation length. The default is 0.
    hyp_omega : non-negative real, optional
        Scale of kernel function. The default is 0.
    hyp_pi : non-negative real, optional
        Constant of kernel function. The default is 0.
    delta : non-negative real, optional
        Diagonal widening. The default is 1e-9.

    Returns
    -------
    cov_mat : np.array
        Covariance Matrix.

    '''

    #number of grid nodes
    n_pt_1 = t_1.shape[0]
    n_pt_2 = t_2.shape[0]
    #number of dimensions
    n_dim = t_1.ndim
    
    #create cov. matrix
    cov_mat = np.zeros([n_pt_1,n_pt_2]) #initialize
    for i in range(n_pt_1):
        dist = scipylinalg.norm(t_1[i] - t_2,axis=1) if n_dim > 1 else np.abs(t_1[i] - t_2)
        cov_mat[i,:] = hyp_pi**2 + hyp_omega**2 * np.exp(- dist/hyp_ell)
    
    if n_pt_1 == n_pt_2:
        for i in range(n_pt_1):
            cov_mat[i,i] += delta

    return cov_mat

# squared exponential kernel
#---  ---  ---  ---  ---  ---  ---  ---
def KernelSqExp(t_1, t_2, hyp_ell = 0, hyp_omega = 0, hyp_pi = 0, delta = 1e-9):
    '''
    Compute squared exponential kernel function

    Parameters
    ----------
    t_1 : np.array
        Coordinates of first group.
    t_2 : np.array
        Coordinates of second group.
    hyp_ell : non-negative real, optional
        Correlation length. The default is 0.
    hyp_omega : non-negative real, optional
        Scale of kernel function. The default is 0.
    hyp_pi : non-negative real, optional
        Constant of kernel function. The default is 0.
    delta : non-negative real, optional
        Diagonal widening. The default is 1e-9.

    Returns
    -------
    cov_mat : np.array
        Covariance Matrix.

    '''

    #number of grid nodes
    n_pt_1 = t_1.shape[0]
    n_pt_2 = t_2.shape[0]
    #number of dimensions
    n_dim = t_1.ndim
    
    #create cov. matrix
    cov_mat = np.zeros([n_pt_1,n_pt_2]) #initialize
    for i in range(n_pt_1):
        dist = scipylinalg.norm(t_1[i] - t_2,axis=1) if n_dim > 1 else np.abs(t_1[i] - t_2)
        cov_mat[i,:] = hyp_pi**2 + hyp_omega**2 * np.exp(- dist**2/hyp_ell**2)
    
    if n_pt_1 == n_pt_2:
        for i in range(n_pt_1):
            cov_mat[i,i] += delta

    return cov_mat

# composite exponential kernel and spatially independent
#---  ---  ---  ---  ---  ---  ---  ---
def KernelNegExpSptInpt(t_1, t_2, hyp_ell1 = 0, hyp_omega1 = 0,  hyp_omega2 = 0, hyp_pi = 0, delta = 1e-9):
    '''
    Compute composite kernel function, with negative exponential and 
    spatially idependent components

    Parameters
    ----------
    t_1 : np.array
        Coordinates of first group.
    t_2 : np.array
        Coordinates of second group.
    hyp_ell1 : non-negative real, optional
        Correlation length of neg. exponential component. The default is 0.
    hyp_omega1 : non-negative real, optional
        Scale of neg. exponential component. The default is 0.
    hyp_omega2 : non-negative real, optional
        Scale of spatially independent component. The default is 0.
    hyp_pi : non-negative real, optional
        Constant of kernel function. The default is 0.
    delta : non-negative real, optional
        Diagonal widening. The default is 1e-9.

    Returns
    -------
    cov_mat : TYPE
        DESCRIPTION.

    '''

    #number of grid nodes
    n_pt_1 = t_1.shape[0]
    n_pt_2 = t_2.shape[0]
    
    #negative exponetial component 
    cov_mat  = KernelNegExp(t_1, t_2, hyp_ell=hyp_ell1, hyp_omega=hyp_omega1, hyp_pi=hyp_pi, delta=1e-9)

    #spatially independent component
    cov_mat += KernelGroup(t_1, t_2, hyp_omega=hyp_omega2, delta=0)
        
    return cov_mat


# Coefficient Sampling Functions
#--------------------------------------
# sample coeffs with group kernel function
#---  ---  ---  ---  ---  ---  ---  ---
def PredictGroupCoeff(g_prdct, g_train, c_train_mu, c_train_sig = None,  
                      hyp_mean_c = 0, hyp_omega = 0, delta = 1e-9):
    '''
    Predict conditional coefficients based on group kernel function.

    Parameters
    ----------
    g_prdct : np.array
        Group IDs of prediction cases.
    g_train : np.array
        Group IDs of training cases.
    c_train_mu : np.array
        Mean values of non-ergodic coefficient of training cases.
    c_train_sig : np.array, optional
        Standard deviations of non-ergodic coefficient of training cases. The default is None.
    hyp_mean_c : real, optional
        Mean of non-ergodic coefficient. The default is 0.
    hyp_omega : non-negative real, optional
        Scale of kernel function. The default is 0.
    delta : non-negative real, optional
        Diagonal widening. The default is 1e-9.

    Returns
    -------
    c_prdct_mu : np.array
        Mean value of non-ergodic coefficient for prediction cases.
    c_prdct_sig : np.array
        Standard deviations of non-ergodic coefficient for prediction cases.
    c_prdct_cov : np.array
        Covariance matrix of non-ergodic coefficient for prediction cases.

    '''
    
    #remove mean effect from training coefficients
    c_train_mu = c_train_mu - hyp_mean_c
    
    #uncertainty in training data
    if c_train_sig is None: c_train_sig = np.zeros(len(c_train_mu))
    c_train_cov = np.diag(c_train_sig**2) if c_train_sig.ndim == 1 else c_train_sig
    
    #covariance between training data 
    K      = KernelGroup(g_train, g_train, hyp_omega=hyp_omega, delta=delta)
    #covariance between data and new locations
    k      = KernelGroup(g_prdct, g_train, hyp_omega=hyp_omega, delta=0)
    #covariance between new locations
    k_star = KernelGroup(g_prdct, g_prdct, hyp_omega=hyp_omega, delta=0)

    #inverse of covariance matrix
    K_inv = scipylinalg.inv(K)
    #product of k * K^-1
    kK_inv = k.dot(K_inv)
    
    #posterior mean and variance at new locations
    c_prdct_mu  = kK_inv.dot(c_train_mu)
    c_prdct_cov = k_star - kK_inv.dot(k.transpose()) + kK_inv.dot( c_train_cov.dot(kK_inv.transpose()) )
    #posterior standard dev. at new locations
    c_prdct_sig = np.sqrt(np.diag(c_prdct_cov))
    
    #add mean effect from training coefficients
    c_prdct_mu += hyp_mean_c
    
    return c_prdct_mu, c_prdct_sig, c_prdct_cov

# sample coeffs with negative exponential kernel
#---  ---  ---  ---  ---  ---  ---  ---
def PredictNegExp(t_prdct, t_train, c_train_mu, c_train_sig = None, 
                  hyp_mean_c = 0, hyp_ell = 0, hyp_omega = 0, hyp_pi = 0, delta = 1e-9):
    '''
    Predict conditional coefficients based on negative exponential kernel function.

    Parameters
    ----------
    t_prdct : np.array
        Coordinates of prediction cases.
    t_train : np.array
        Coordinates of training cases.
    c_train_mu : np.array
        Mean values of non-ergodic coefficient of training cases.
    c_train_sig : np.array, optional
        Standard deviations of non-ergodic coefficient of training cases. The default is None.
    hyp_mean_c : real, optional
        Mean of non-ergodic coefficient. The default is 0.
    hyp_ell : non-negative real, optional
        Correlation length of kernel function.. The default is 0.
    hyp_omega : non-negative real, optional
        Scale of kernel function. The default is 0.
    hyp_pi : postive real, optional
        Constant of kernel function. The default is 0.
    delta : non-negative real, optional
        Diagonal widening. The default is 1e-9.

    Returns
    -------
    c_prdct_mu : np.array
        Mean value of non-ergodic coefficient for prediction cases.
    c_prdct_sig : np.array
        Standard deviations of non-ergodic coefficient for prediction cases.
    c_prdct_cov : np.array
        Covariance matrix of non-ergodic coefficient for prediction cases.

    '''
    
    
    #remove mean effect from training coefficients
    c_train_mu = c_train_mu - hyp_mean_c
    
    #uncertainty in training data
    if c_train_sig is None: c_train_sig = np.zeros(len(c_train_mu))
    c_train_cov = np.diag(c_train_sig**2) if c_train_sig.ndim == 1 else c_train_sig

    #covariance between training data 
    K      = KernelNegExp(t_train, t_train, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi, delta=delta)
    #covariance between data and new locations
    k      = KernelNegExp(t_prdct, t_train, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi, delta=0)
    #covariance between new locations
    k_star = KernelNegExp(t_prdct, t_prdct, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi, delta=0)

    #inverse of covariance matrix
    K_inv = scipylinalg.inv(K)
    #product of k * K^-1
    kK_inv = k.dot(K_inv)
    
    #posterior mean and variance at new locations
    c_prdct_mu  = kK_inv.dot(c_train_mu)
    c_prdct_cov = k_star - kK_inv.dot(k.transpose()) + kK_inv.dot( c_train_cov.dot(kK_inv.transpose()) )
    #posterior standard dev. at new locations
    c_prdct_sig = np.sqrt(np.diag(c_prdct_cov))
    
    #add mean effect from training coefficients
    c_prdct_mu += hyp_mean_c
    
    return c_prdct_mu, c_prdct_sig, c_prdct_cov

# sample coeffs with squared exponential kernel
#---  ---  ---  ---  ---  ---  ---  ---
def PredictSqExp(t_prdct, t_train, c_train_mu, c_train_sig = None,  
                  hyp_mean_c = 0, hyp_ell = 0, hyp_omega = 0, hyp_pi = 0, delta = 1e-9):
    '''
    Predict conditional coefficients based on squared exponential kernel function.

    Parameters
    ----------
    t_prdct : np.array
        Coordinates of prediction cases.
    t_train : np.array
        Coordinates of training cases.
    c_train_mu : np.array
        Mean values of non-ergodic coefficient of training cases.
    c_train_sig : np.array, optional
        Standard deviations of non-ergodic coefficient of training cases. The default is None.
    hyp_mean_c : real, optional
        Mean of non-ergodic coefficient. The default is 0.
    hyp_ell : non-negative real, optional
        Correlation length of kernel function.. The default is 0.
    hyp_omega : non-negative real, optional
        Scale of kernel function. The default is 0.
    hyp_pi : postive real, optional
        Constant of kernel function. The default is 0.
    delta : non-negative real, optional
        Diagonal widening. The default is 1e-9.

    Returns
    -------
    c_prdct_mu : np.array
        Mean value of non-ergodic coefficient for prediction cases.
    c_prdct_sig : np.array
        Standard deviations of non-ergodic coefficient for prediction cases.
    c_prdct_cov : np.array
        Covariance matrix of non-ergodic coefficient for prediction cases.

    '''
    
    #remove mean effect from training coefficients
    c_train_mu = c_train_mu - hyp_mean_c

    #uncertainty in training data
    if c_train_sig is None: c_train_sig = np.zeros(len(c_train_mu))
    c_train_cov = np.diag(c_train_sig**2) if c_train_sig.ndim == 1 else c_train_sig
    
    #covariance between training data 
    K      = KernelNegExpSptInpt(t_train, t_train, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi, delta=delta)
    #covariance between data and new locations
    k      = KernelNegExpSptInpt(t_prdct, t_train, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi, delta=0)
    #covariance between new locations
    k_star = KernelNegExpSptInpt(t_prdct, t_prdct, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi, delta=0)

    #inverse of covariance matrix
    K_inv = scipylinalg.inv(K)
    #product of k * K^-1
    kK_inv = k.dot(K_inv)
    
    #posterior mean and variance at new locations
    c_prdct_mu  = kK_inv.dot(c_train_mu)
    c_prdct_cov = k_star - kK_inv.dot(k.transpose()) + kK_inv.dot( c_train_cov.dot(kK_inv.transpose()) )
    #posterior standard dev. at new locations
    c_prdct_sig = np.sqrt(np.diag(c_prdct_cov))
    
    #add mean effect from training coefficients
    c_prdct_mu += hyp_mean_c
    
    return c_prdct_mu, c_prdct_sig, c_prdct_cov

# sample coeffs with  composite negative exponential and spatially independent kernel function
#---  ---  ---  ---  ---  ---  ---  ---
def PredictNegExpSptInpt(t_prdct, t_train, c_train_mu, c_train_sig = None, 
                        hyp_mean_c = 0, hyp_ell1 = 0, hyp_omega1 = 0, 
                        hyp_omega2 = 0, hyp_pi = 0, delta = 1e-9):
    '''
    Predict conditional coefficients based on composite negative exponential and 
    spatially independent kernel function.


    Parameters
    ----------
    t_prdct : np.array
        Coordinates of prediction cases.
    t_train : np.array
        Coordinates of training cases.
    c_train_mu : np.array
        Mean values of non-ergodic coefficient of training cases.
    c_train_sig : np.array, optional
        Standard deviations of non-ergodic coefficient of training cases. The default is None.
    hyp_mean_c : real, optional
        Mean of non-ergodic coefficient. The default is 0.
    hyp_ell1 : non-negative real, optional
        Correlation length of negative exponential kernel function. The default is 0.
    hyp_omega1 : non-negative real, optional
        Scale of negative exponential kernel function. The default is 0.
    hyp_omega2 : non-negative real, optional
        Scale of spatially independent kernel function. The default is 0.
    hyp_pi : postive real, optional
        Constant of kernel function. The default is 0.
    delta : non-negative real, optional
        Diagonal widening. The default is 1e-9.

    Returns
    -------
    c_prdct_mu : np.array
        Mean value of non-ergodic coefficient for prediction cases.
    c_prdct_sig : np.array
        Standard deviations of non-ergodic coefficient for prediction cases.
    c_prdct_cov : np.array
        Covariance matrix of non-ergodic coefficient for prediction cases.

    '''
    
    #remove mean effect from training coefficients
    c_train_mu = c_train_mu - hyp_mean_c
    
    #uncertainty in training data
    if c_train_sig is None: c_train_sig = np.zeros(len(c_train_mu))
    c_train_cov = np.diag(c_train_sig**2) if c_train_sig.ndim == 1 else c_train_sig

    #covariance between training data 
    K      = KernelNegExpSptInpt(t_train, t_train, hyp_ell1=hyp_ell1, hyp_omega1=hyp_omega1,
                                 hyp_omega2=hyp_omega2, hyp_pi=hyp_pi, delta=delta)
    #covariance between data and new locations
    k      = KernelNegExpSptInpt(t_prdct, t_train, hyp_ell1=hyp_ell1, hyp_omega1=hyp_omega1, 
                                 hyp_omega2=hyp_omega2, hyp_pi=hyp_pi, delta=0)
    #covariance between new locations
    k_star = KernelNegExpSptInpt(t_prdct, t_prdct, hyp_ell1=hyp_ell1, hyp_omega1=hyp_omega1, 
                                 hyp_omega2=hyp_omega2, hyp_pi=hyp_pi, delta=0)

    #inverse of covariance matrix
    K_inv = scipylinalg.inv(K)
    #product of k * K^-1
    kK_inv = k.dot(K_inv)
    
    #posterior mean and variance at new locations
    c_prdct_mu  = kK_inv.dot(c_train_mu)
    c_prdct_cov = k_star - kK_inv.dot(k.transpose()) + kK_inv.dot( c_train_cov.dot(kK_inv.transpose()) )
    #posterior standard dev. at new locations
    c_prdct_sig = np.sqrt(np.diag(c_prdct_cov))
    
    #add mean effect from training coefficients
    c_prdct_mu += hyp_mean_c
    
    return c_prdct_mu, c_prdct_sig, c_prdct_cov



# Multivariate normal distribution random samples
#--------------------------------------
def MVNRnd(mean = None, cov = None, seed = None, n_samp = None, flag_sp = False, flag_list = False):
    
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
