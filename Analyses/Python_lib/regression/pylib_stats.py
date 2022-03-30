#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:56:13 2022

@author: glavrent

Other python statistics functions
"""

#imprort libraries
import numpy as np

def CalcRMS(samp_q, samp_p):
    '''
    Compute root mean square error between observation samples (samp_p) and 
    model samples (samp_p)    

    Parameters
    ----------
    samp_q : np.array()
        Model Samples.
    samp_p : np.array()
        Data Samples.

    Returns
    -------
    real
        root mean square error
    '''
    
    #errors
    e = samp_q - samp_p
    
    return np.sqrt(np.mean(e**2))


def CalcLKDivergece(samp_q, samp_p):
    '''
    Compute Kullback–Leibler divergence of observation samples (samp_p) based 
    on model samples (samp_p)    

    Parameters
    ----------
    samp_q : np.array()
        Model Samples.
    samp_p : np.array()
        Data Samples.

    Returns
    -------
    real
        Kullback–Leibler divergence.
    '''
    
    #create histogram bins
    _, hist_bins = np.histogram(np.concatenate([samp_p,samp_q]))
    
    #count of p and q distribution
    p, _ = np.histogram(samp_p, bins=hist_bins)
    q, _ = np.histogram(samp_q, bins=hist_bins)

    #remove bins empty in any dist, otherwise kl= +/- inf
    i_empty_bins = np.logical_or(p==0, q==0)
    p = p[~i_empty_bins]
    q = q[~i_empty_bins]
    
    #normalize to compute probabilites 
    p = p/p.sum()
    q = q/q.sum()
    
    return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))
