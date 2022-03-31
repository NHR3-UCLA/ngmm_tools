#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:32:37 2021

@author: glavrent
"""
# %% Required Packages
# ======================================
#load libraries
import os
import pathlib
#arithmetic libraries
import numpy as np
import pandas as pd
#plotting libraries
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

# %% Define variables
# ======================================
#input file names
fname_flatfile_NGA2 = '../../../Raw_files/nga_w2/Updated_NGA_West2_Flatfile_RotD50_d050_public_version.xlsx'
fname_mag_rrup_lim  = '../../../Data/Verification/preprocessing/flatfiles/usable_mag_rrup/usable_Mag_Rrup_coeffs.csv'

#output directoy
dir_fig = '../../../Data/Verification/preprocessing/flatfiles/usable_mag_rrup/'

# %% Load Data
# ======================================
#NGAWest2
df_flatfile_NGA2 = pd.read_excel(fname_flatfile_NGA2)
#M/R limit
df_m_r_lim = pd.read_csv(fname_mag_rrup_lim,index_col=0)

#remove rec with unavailable data
df_flatfile_NGA2 = df_flatfile_NGA2.loc[df_flatfile_NGA2.EQID>0,:]
df_flatfile_NGA2 = df_flatfile_NGA2.loc[df_flatfile_NGA2['ClstD (km)']>0,:]

#mag and distance arrays
mag_array  = df_flatfile_NGA2['Earthquake Magnitude']
rrup_array = df_flatfile_NGA2['ClstD (km)']

#compute limit 
rrup_lim1 = np.arange(0,1001)
mag_lim1  = (df_m_r_lim.loc['b0','coefficients'] +
             df_m_r_lim.loc['b1','coefficients'] * rrup_lim1 +
             df_m_r_lim.loc['b2','coefficients'] * rrup_lim1**2)
rrup_lim2 = df_m_r_lim.loc['max_rrup','coefficients']

# %% Process Data
# ======================================
if not os.path.isdir(dir_fig): pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True)

# create figures
# ----   ----   ----   ----   ----
# Mag-Dist distribution
fname_fig = 'M-R_limits'
#create figure   
fig, ax = plt.subplots(figsize = (10,9))
pl1 = ax.scatter(rrup_array, mag_array, label='NGAWest2 CA')
pl2 = ax.plot(rrup_lim1, mag_lim1,          linewidth=2, color='black')
pl3 = ax.vlines(rrup_lim2, ymin=0, ymax=10, linewidth=2, color='black', linestyle='--')
#edit figure properties
ax.set_xlabel(r'Distance ($km$)', fontsize=30)
ax.set_ylabel(r'Magnitude', fontsize=30)
ax.grid(which='both')
# ax.set_xscale('log')
ax.set_xlim([0, 1000])
ax.set_ylim([2, 8])
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
# ax.legend(fontsize=25, loc='upper left')
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', right='on')
fig.tight_layout()
#save figure
fig.savefig( dir_fig + fname_fig + '.png' )
