#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 13:52:51 2022

@author: glavrent
"""

# Packages
# ---------------------------
#arithmetic libraries
import numpy as np
from scipy import linalg as scipylinalg 
from sklearn.gaussian_process.kernels import Matern

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

# exponential kernel
#---  ---  ---  ---  ---  ---  ---  ---
def KernelExp(t_1, t_2, hyp_ell = 0, hyp_omega = 0, hyp_pi = 0, delta = 1e-9):
    '''
    Compute exponential kernel function

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

# matern exponential kernel
#---  ---  ---  ---  ---  ---  ---  ---
def MaternKernel(t_1, t_2, hyp_ell = 0, hyp_omega = 0, hyp_pi = 0, hyp_nu=1.5, delta = 1e-9):
    '''
    Compute Matern kernel function
    

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
    hyp_nu : non-negative real, optional
        Smoothness parameter. The default is 1.5.
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

    #distance matrix
    dist_mat = np.array([scipylinalg.norm(t1 - t_2, axis=1) if n_dim > 1 else np.abs(t1 - t_2)
                         for t1 in t_1])
    
    #create cov. matrix
    cov_mat = hyp_omega**2 * Matern(nu=hyp_nu, length_scale=hyp_ell)(0, dist_mat.ravel()[:, np.newaxis]).reshape(dist_mat.shape)
    cov_mat += hyp_pi**2

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
    cov_mat  = KernelExp(t_1, t_2, hyp_ell=hyp_ell1, hyp_omega=hyp_omega1, hyp_pi=hyp_pi, delta=1e-9)

    #spatially independent component
    cov_mat += KernelGroup(t_1, t_2, hyp_omega=hyp_omega2, delta=0)
        
    return cov_mat

# Predictive Functions
#--------------------------------------
# predict coeffs with group kernel function
#---  ---  ---  ---  ---  ---  ---  ---
def PredictGroupKern(g_prdct, g_train, c_train_mu, c_train_sig = None,  
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

# predict coeffs with exponential kernel
#---  ---  ---  ---  ---  ---  ---  ---
def PredictExpKern(t_prdct, t_train, c_train_mu, c_train_sig = None, 
                   hyp_mean_c = 0, hyp_ell = 0, hyp_omega = 0, hyp_pi = 0, delta = 1e-9):
    '''
    Predict conditional coefficients based on exponential kernel function.

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
    K      = KernelExp(t_train, t_train, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi, delta=delta)
    #covariance between data and new locations
    k      = KernelExp(t_prdct, t_train, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi, delta=0)
    #covariance between new locations
    k_star = KernelExp(t_prdct, t_prdct, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi, delta=0)

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

# predict coeffs with squared exponential kernel
#---  ---  ---  ---  ---  ---  ---  ---
def PredictSqExpKern(t_prdct, t_train, c_train_mu, c_train_sig = None,  
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

# predict coeffs with Matern kernel
#---  ---  ---  ---  ---  ---  ---  ---
def PredictMaternKern(t_prdct, t_train, c_train_mu, c_train_sig = None, 
                      hyp_mean_c = 0, hyp_ell = 0, hyp_omega = 0, hyp_pi = 0, hyp_nu=1.5,
                      delta = 1e-9):
    '''
    Predict conditional coefficients based on Matern kernel function.

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
    hyp_nu: positive real, optional
        Smoothness parameter. The default is 1.5.
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
    K      = MaternKernel(t_train, t_train, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi, hyp_nu=hyp_nu, delta=delta)
    #covariance between data and new locations
    k      = MaternKernel(t_prdct, t_train, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi, hyp_nu=hyp_nu, delta=0)
    #covariance between new locations
    k_star = MaternKernel(t_prdct, t_prdct, hyp_ell=hyp_ell, hyp_omega=hyp_omega, hyp_pi=hyp_pi, hyp_nu=hyp_nu, delta=0)

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

# predict coeffs with composite exponential and spatially independent kernel function
#---  ---  ---  ---  ---  ---  ---  ---
def PredictNegExpSptInptKern(t_prdct, t_train, c_train_mu, c_train_sig = None, 
                             hyp_mean_c = 0, hyp_ell1 = 0, hyp_omega1 = 0, 
                             hyp_omega2 = 0, hyp_pi = 0, delta = 1e-9):
    '''
    Predict conditional coefficients based on composite exponential and 
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