#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:01:54 2022

@author: glavrent
"""
# Working directory and Packages
# ---------------------------
import os
import sys
import pathlib
#load packages
import numpy as np
import pandas as pd
#plottign libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

# Define Problem
# ---------------------------
#number of samples
n_samp = 1000

#coefficients
c0  = -0.2
c1  =  0.6
sig =  0.7

#output directory
dir_out = 'data/'

# Create Dataset
# ---------------------------
#covariates
x1  = np.random.randn(n_samp )
#noise
eps = sig *np.random.randn(n_samp )
#response
mu_y = c0 + c1 * x1 
y = mu_y + eps

#model response
model_x1 = np.linspace(-5,5)
model_y  = c0 + c1 * model_x1

#regression data frame
df_data = pd.DataFrame({'x1':x1, 'mu_y':mu_y, 'y':y})

# Save Dataset
# ---------------------------
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
df_data.to_csv( dir_out + 'regression_dataset.csv', index=False )

# Summary Figures
# ---------------------------
#figure title
fname_fig = 'fig_dataset'
#create figure
fig, ax = plt.subplots(figsize = (10,10))
#obsevations
hl1 = ax.plot(df_data.x1, df_data.y, 'o')
#plot response
hl2 = ax.plot(model_x1, model_y, linewidth=3, color='black')
#figure properties
ax.grid(which='both')
#tick size
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
#labels
ax.set_xlabel(r'$x_1$', fontsize=35)
ax.set_ylabel(r'$y$', fontsize=35)
#figure limits
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
#save figure
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png' )



