#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:39:12 2021

@author: glavrent
"""

#load libraries
#arithmetic libraries
import numpy as np

def IndexAvgColumns(df_data, col_idx, col2avg):
    '''
    Average columns based on index column

    Parameters
    ----------
    df_data : pd.dataframe
        Data data-frame.
    col_idx : str
        Name of index column.
    col2avg : list
        List of column names to be averaged.

    Returns
    -------
    df_data : pd.dataframe
        Data data-frame.

    '''
    
    #unique ids
    idx_array, inv_array = np.unique(df_data[col_idx], return_inverse=True)
    #iterate over columns
    for col in col2avg:
        #compute average values for all unique indices
        avg_vals = np.array([np.nanmean(df_data.loc[df_data[col_idx] == idx,col]) for idx in idx_array])
        df_data.loc[:,col] = avg_vals[inv_array]
            
    return df_data

def ColocatePt(df_flatfile, col_idx, col_coor, thres_dist=0.01, return_df_pt=False):
    '''
    Colocate points (assign same ID) based on threshold distance.

    Parameters
    ----------
    df_flatfile : pd.DataFrame
        Catalog flatfile.
    col_idx : str
        Name of index column.
    col_coor : list of str
        List of coordinate name columns.
    thres_dist : real, optional
        Value of threshold distance. The default is 0.01.
    return_df_pt : bool, optional
        Option for returning point data frame. The default is False.

    Returns
    -------
    df_flatfile : pd.DataFrame
        Catalog flatfile with updated index column.
    df_pt: pd.DataFrame
        Point data frame with updated index column.
    '''

    #dataframe with unique points
    _, pt_idx, pt_inv = np.unique(df_flatfile[col_idx], axis=0, return_index=True, return_inverse=True)
    df_pt = df_flatfile.loc[:,[col_idx] + col_coor].iloc[pt_idx,:]
    
    #find and merge collocated points
    for _, pt in df_pt.iterrows():
        #distance between points
        dist2pt = np.linalg.norm((df_pt[col_coor] - pt[col_coor]).astype(float), axis=1)
        #indices of collocated points
        i_pt_coll = dist2pt < thres_dist
        #assign new id for collocated points
        df_pt.loc[i_pt_coll,col_idx] = pt[col_idx].astype(int)
    
    #update pt info to main catalog
    df_flatfile.loc[:,col_idx] = df_pt[col_idx].values[pt_inv]
    
    if not return_df_pt:
        return df_flatfile
    else:
        return df_flatfile, df_pt
    
def UsableSta(mag_array, dist_array, df_coeffs):
    '''
    Find records that meet the mag-distance limits

    Parameters
    ----------
    mag_array : np.array
        Magnitude array.
    dist_array : np.array
        Distance array.
    df_coeffs : pd.DataFrame
        Coefficients dataframe.

    Returns
    -------
    rec_lim : np.array
        logical array with True for records that meet M/R limits.

    '''
    
    #rrup limit
    rrup_lim = dist_array <= df_coeffs.loc['max_rrup','coefficients']
    
    #mag limit
    mag_min = (df_coeffs.loc['b1','coefficients'] + 
               df_coeffs.loc['b1','coefficients'] * dist_array + 
               df_coeffs.loc['b2','coefficients'] * dist_array**2)
    mag_lim = mag_array >= mag_min 
    
    #find records that meet both conditions
    rec_lim = np.logical_and(rrup_lim, mag_lim)

    return rec_lim
