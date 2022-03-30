#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:04:25 2020

@author: glavrent
"""
# Required Packages
# ======================================
#load libraries
import os
import sys
import pathlib
import numpy as np
import pandas as pd
from scipy import sparse
#geographic libraries
import pyproj
#user libraries
sys.path.insert(0,'../../Python_lib/ground_motions')
import pylib_cell_dist as pylib_cells

# %% Define Input Data
# ======================================

#input flatfile
# fname_flatfile = 'CatalogNGAWest3CA'
# fname_flatfile = 'CatalogNGAWest3CA_2013'
fname_flatfile = 'CatalogNGAWest3CALite'
# fname_flatfile = 'CatalogNGAWest3NCA'
# fname_flatfile = 'CatalogNGAWest3SCA'
dir_flatfile   = '../../../Data/Verification/preprocessing/flatfiles/merged/'

#output files
dir_out = '../../../Data/Verification/preprocessing/cell_distances/'

# %% Read and Porcess Input Data
# ======================================
# read ground-motion data
fullname_flatfile = dir_flatfile + fname_flatfile + '.csv'
df_flatfile = pd.read_csv(fullname_flatfile)
n_rec = len(df_flatfile)

#define projection system 
assert(len(np.unique(df_flatfile.UTMzone))==1),'Error. Multiple UTM zones defined.'
utm_zone = df_flatfile.UTMzone[0]
utmProj = pyproj.Proj("+proj=utm +zone="+utm_zone+", +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

#create output directory
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 

#create object with source and station locations
#utm coordinates
data4celldist = df_flatfile.loc[:,['eqX','eqY','eqZ','staX','staY']].values
flagUTM = True
#add elevation for stations
data4celldist = np.hstack([data4celldist,np.zeros([n_rec,1])])


# %% Create Cell Grid
# ======================================
#grid range
grid_lims_x = [data4celldist[:,[0,3]].min(), data4celldist[:,[0,3]].max()]
grid_lims_y = [data4celldist[:,[1,4]].min(), data4celldist[:,[1,4]].max()]
grid_lims_z = [data4celldist[:,[2,5]].min(), data4celldist[:,[2,5]].max()]

#manual limits
#utm limits
# #NGAWest3 full
# grid_lims_x = [-200,    1100]
# grid_lims_y = [3300,    4800]
# grid_lims_z = [-100,     0]
#NGAWest3 lite
grid_lims_x = [-200,    800]
grid_lims_y = [3450,    4725]
grid_lims_z = [-50,     0]

#cell size
cell_size = [25, 25, 50]
#lat-lon grid spacing
grid_x = np.arange(grid_lims_x[0], grid_lims_x[1]+0.1, cell_size[0])
grid_y = np.arange(grid_lims_y[0], grid_lims_y[1]+0.1, cell_size[1])
grid_z = np.arange(grid_lims_z[0], grid_lims_z[1]+0.1, cell_size[2])

#cell schematic
#    (7)-----(8)  (surface, top face)
#   / |     / |
# (5)-----(6) |
#  |  |    |  |
#  | (3)---|-(4)  (bottom face)
#  |/      |/
# (1)-----(2)


#create cells
j1 = 0
j2 = 0
j3 = 0
cells = []
for j1 in range(len(grid_x)-1):
    for  j2 in range(len(grid_y)-1):
        for j3 in range(len(grid_z)-1):
            #cell corners (bottom-face)
            cell_c1 = [grid_x[j1],   grid_y[j2],   grid_z[j3]]
            cell_c2 = [grid_x[j1+1], grid_y[j2],   grid_z[j3]]
            cell_c3 = [grid_x[j1],   grid_y[j2+1], grid_z[j3]]
            cell_c4 = [grid_x[j1+1], grid_y[j2+1], grid_z[j3]]
            #cell corners (top-face)
            cell_c5 = [grid_x[j1],   grid_y[j2],   grid_z[j3+1]]
            cell_c6 = [grid_x[j1+1], grid_y[j2],   grid_z[j3+1]]
            cell_c7 = [grid_x[j1],   grid_y[j2+1], grid_z[j3+1]]
            cell_c8 = [grid_x[j1+1], grid_y[j2+1], grid_z[j3+1]]
            #cell center
            cell_cent = np.mean(np.stack([cell_c1,cell_c2,cell_c3,cell_c4,
                                          cell_c5,cell_c6,cell_c7,cell_c8]),axis = 0).tolist()
            #summarize all cell coordinates in a list
            cell_info = cell_c1 + cell_c2 + cell_c3 + cell_c4 + \
                        cell_c5 + cell_c6 + cell_c7 + cell_c8 + cell_cent
            #add cell info
            cells.append(cell_info)
del j1, j2, j3, cell_info
del cell_c1, cell_c2, cell_c3, cell_c4, cell_c5, cell_c6, cell_c7, cell_c8
cells = np.array(cells)
n_cells = len(cells)

#cell info
cell_ids   = np.arange(n_cells)
cell_names = ['c.%i'%(i) for i in cell_ids]
cell_q_names =  ['q1X','q1Y','q1Z','q2X','q2Y','q2Z','q3X','q3Y','q3Z','q4X','q4Y','q4Z',
                 'q5X','q5Y','q5Z','q6X','q6Y','q6Z','q7X','q7Y','q7Z','q8X','q8Y','q8Z',
                 'mptX','mptY','mptZ']


# Create cell info dataframe
# ----   ----   ----   ----   ----
#cell names
df_data1 = pd.DataFrame({'cellid': cell_ids, 'cellname': cell_names})
#cell coordinates
df_data2 = pd.DataFrame(cells, columns = cell_q_names)
df_cellinfo  = pd.merge(df_data1,df_data2,left_index=True,right_index=True)
# add cell utm zone
df_cellinfo.loc[:,'UTMzone'] = utm_zone

# Compute Lat\Lon of cells
# ----   ----   ----   ----   ----
#cell verticies
for q in range(1,9):
    c_X      = ['q%iX'%q,   'q%iY'%q]
    c_latlon = ['q%iLat'%q, 'q%iLon'%q]
    
    df_cellinfo.loc[:,c_latlon]  = np.flip( np.array([utmProj(pt_xy[0]*1e3, pt_xy[1]*1e3, inverse=True) 
                                                 for _, pt_xy in df_cellinfo[c_X].iterrows() ]),   axis=1)
#cell midpoints
c_X      = ['mptX',  'mptY']
c_latlon = ['mptLat','mptLon']
df_cellinfo.loc[:,c_latlon]  = np.flip( np.array([utmProj(pt_xy[0]*1e3, pt_xy[1]*1e3, inverse=True) 
                                                 for _, pt_xy in df_cellinfo[c_X].iterrows() ]),   axis=1)

# %% Compute Cell distances
# ======================================
cells4dist  = cells[:,[0,1,2,21,22,23]] 
distancematrix  = np.zeros([len(data4celldist), len(cells4dist)])
for i in range(len(data4celldist)):
    print('Computing cell distances, record',i)
    pt1 = data4celldist[i,(0,1,2)]
    pt2 = data4celldist[i,(3,4,5)]
  
    dm = pylib_cells.ComputeDistGridCells(pt1,pt2,cells4dist, flagUTM)
    distancematrix[i] = dm
    
#print Rrup missfits
dist_diff = df_flatfile.Rrup - distancematrix.sum(axis=1)
print('max R_rup misfit', max(dist_diff))
print('min R_rup misfit', min(dist_diff))

#convert cell distances to sparse matrix
distmatrix_sparce = sparse.coo_matrix(distancematrix)

# Create cell distances data-frame
# ----   ----   ----   ----   ----
#record info
df_recinfo = df_flatfile[['rsn','eqid','ssn']]

#cell distances
df_celldist = pd.DataFrame(distancematrix, columns = cell_names)
df_celldist = pd.merge(df_recinfo, df_celldist, left_index=True, right_index=True)

#spase cell distances
df_celldist_sp = pd.DataFrame({'row': distmatrix_sparce.row+1, 'col': distmatrix_sparce.col+1, 'data': distmatrix_sparce.data})

# %% Save data
# ======================================
#save cell info
fname_cellinfo = fname_flatfile + '_cellinfo'
df_cellinfo.to_csv(dir_out + fname_cellinfo + '.csv', index=False)

# #save distance metrics
fname_celldist = fname_flatfile + '_distancematrix'
df_celldist.to_csv(dir_out + fname_celldist + '.csv', index=False)

# #save distance matrix as sparce
fname_celldist = fname_flatfile + '_distancematrix_sparce'
df_celldist_sp.to_csv(dir_out + fname_celldist + '.csv', index=False)

