#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:12:38 2019

@author: glavrent
"""

## load libraries
#arithmetic
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
#plotting
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
#base map
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class FormatScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,
                                                        useMathText=mathText)
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)
            

## Main functions
##---------------------------------------

def PlotContourMapObs(cont_latlondata, cmin=None,  cmax=None, flag_grid=False, title=None, cbar_label=None, log_cbar = False, frmt_clb = '%.2f',
                      prj_map = False):
    '''
    PlotContourMapObs:

        
    Input Arguments:
        cont_latlondata (np.array [n1,3]):       contains the latitude, logitude and contour values
                                                 cont_latlondata = [lat, long, data]
        cmin (double-opt):                       lower limit for color levels for contour plot 
        cmax (double-opt):                       upper limit for color levels for contour plot 
        title (str-opt):                         figure title
        cbar_label (str-opt):                    contour plot color bar label
        ptlevs (np.array-opt):                   color levels for points
        pt_label (str-opt):                      points color bar label
        log_cbar (bool-opt):                     if true use log-scale for contour plots
        frmt_clb                                 string format color bar ticks
    
    Output Arguments:
        
    '''
    
    plt_res = '50m'
    plt_scale = '50m'

    #number of interpolation points, x & y direction
    #ngridx = 5000
    #ngridy = 5000
    #ngridx = 500
    #ngridy = 500
    ngridx = 100
    ngridy = 100

    #create figure
    fig = plt.figure(figsize=(10, 10))
    #fig = plt.figure(figsize=(15, 15))
    #create basemap
    if prj_map == True:
        data_crs = ccrs.PlateCarree()
        ax = fig.add_subplot(1, 1, 1, projection=data_crs)
    else:
        data_crs = None
        ax = fig.add_subplot(1, 1, 1)
    
    #project contour data
    x_cont = cont_latlondata[:,1] 
    y_cont = cont_latlondata[:,0]
    #interpolation grid
    x_int = np.linspace(x_cont.min(), x_cont.max(), ngridx)
    y_int = np.linspace(y_cont.min(), y_cont.max(), ngridy)
    X_grid, Y_grid = np.meshgrid(x_int, y_int)
    #interpolate contour data on grid
    if log_cbar:
        data_cont = np.log(cont_latlondata[:,2])
    else:
        data_cont = cont_latlondata[:,2]
    data_grid = griddata((x_cont, y_cont) , data_cont, (X_grid, Y_grid), method='linear')
    #data colorbar
    cbmin = data_cont.min() if cmin is None else cmin
    cbmax = data_cont.max() if cmax is None else cmax
    clevs = np.linspace(cbmin, cbmax, 41).tolist()    
    
    #plot interpolated data
    if prj_map == True:
        cs = ax.contourf(X_grid, Y_grid, data_grid, transform = data_crs, vmin=cmin, vmax=cmax, levels = clevs, zorder=3, alpha = 0.75)
    else:
        cs = ax.contourf(X_grid, Y_grid, data_grid, vmin=cmin, vmax=cmax, levels = clevs, zorder=3, alpha = 0.75)

    #color bar
    fmt_clb = ticker.FormatStrFormatter(frmt_clb)
    cbar_ticks = clevs[0:41:8]
    cbar = fig.colorbar(cs, boundaries=clevs, ticks=cbar_ticks, pad=0.05, orientation="horizontal", format=fmt_clb) # add colorbar
    if log_cbar:
        cbar_labels = [frmt_clb%np.exp(c_t) for c_t in cbar_ticks]
        cbar.set_ticklabels(cbar_labels)
    #add tick labs
    cbar.ax.tick_params(labelsize=18) 
    if (not cbar_label is None): cbar.set_label(cbar_label, size=20)

    if prj_map == True:
        #add costal lines
        ax.coastlines(resolution=plt_res, edgecolor='black', zorder=5);
        #add state boundaries
        states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
                                              scale=plt_scale, facecolor='none')
        ax.add_feature(states, edgecolor='black', zorder=3)
        borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_countries', 
                                               scale=plt_scale, facecolor='none')
        ax.add_feature(borders, edgecolor='black', zorder=4)
        #add water bodies
        oceans = cfeature.NaturalEarthFeature(category='physical', name='ocean', facecolor='lightblue',
                                          scale=plt_scale)
        ax.add_feature(oceans, zorder=6)
    
    #add figure title
    if (not title is None): plt.title(title, fontsize=25)
    plt.xlabel('Latitude (deg)', fontsize=20)
    plt.ylabel('Longitude (deg)', fontsize=20)
    
    #grid lines
    if flag_grid:
        # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
    else:
        gl = None
    
    # fig.show()
    # fig.draw()
    fig.tight_layout()
    
    return fig, ax, cbar, data_crs, gl



# Original PlotContourCAMap function
#----  ----  ----  ----  ----  ----  ----
def PlotContourCAMapAdv(cont_latlondata, line_latlon=None, pt_latlondata=None, clevs=None, flag_grid=False, title=None, cbar_label=None, 
                        ptlevs = None, pt_label = None, log_cbar = False, frmt_clb = '%.2f', **kwargs):
    '''
    PlotContourCAMapAdv:
        create a contour plot of the data in cont_latlondata
        
    Input Arguments:
        cont_latlondata (np.array [n1,3]):       contains the latitude, logitude and contour values
                                                 cont_latlondata = [lat, long, data]
        line_latlon (np.array-opt [n2,2]):       contains the latitde and logitude coordinates of any lines
        pt_latlondata (np.array-opt [n3,(2,3)]): contains the latitude, logitude and values of disp points
                                                 pt_latlondata = [lat, long, data-optional]
        clevs (np.array-opt):                    color levels for contour plot 
        title (str-opt):                         figure title
        cbar_label (str-opt):                    contour plot color bar label
        ptlevs (np.array-opt):                   color levels for points
        pt_label (str-opt):                      points color bar label
        log_cbar (bool-opt):                     if true use log-scale for contour plots
        frmt_clb                                 string format color bar ticks
    
    Output Arguments:
        
    '''
    
    #additional input arguments
    flag_smooth = kwargs['flag_smooth'] if 'flag_smooth' in kwargs else False
    sig_smooth  = kwargs['smooth_sig'] if 'smooth_sig' in kwargs else 0.1
    
    plt_res = '10m'
    plt_scale = '10m'

    #number of interpolation points, x & y direction
    #ngridx = 5000
    #ngridy = 5000
    #ngridx = 500
    #ngridy = 500
    ngridx = 100
    ngridy = 100

    #create figure
    fig = plt.figure(figsize=(10, 10))
    #fig = plt.figure(figsize=(15, 15))
    #create basemap
    data_crs = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=data_crs)
    
    #project contour data
    x_cont = cont_latlondata[:,1] 
    y_cont = cont_latlondata[:,0]
    #interpolation grid
    x_int = np.linspace(x_cont.min(), x_cont.max(), ngridx)
    y_int = np.linspace(y_cont.min(), y_cont.max(), ngridy)
    X_grid, Y_grid = np.meshgrid(x_int, y_int)
    #interpolate contour data on grid
    data_cont = cont_latlondata[:,2]
    data_grid = griddata((x_cont, y_cont) , data_cont, (X_grid, Y_grid), method='linear')
    #smooth 
    if flag_smooth: 
        data_grid = gaussian_filter(data_grid, sigma=sig_smooth)   
    #data colorbar
    if clevs is None:
        if not log_cbar:
            clevs = np.linspace(data_cont.min(),data_cont.max(),11).tolist()    
        else:
            clevs = np.logspace(np.log10(data_cont.min()),np.log10(data_cont.max()),11).tolist()    
        
    #plot interpolated data
    if not log_cbar:
        cs =  ax.contourf(X_grid, Y_grid, data_grid, transform = data_crs, levels = clevs, zorder=3, alpha = 0.75)
    else:
        cs =  ax.contourf(X_grid, Y_grid, data_grid, transform = data_crs, levels = clevs, zorder=3, alpha = 0.75, 
                          locator=ticker.LogLocator())
    
    #color bar
    fmt_clb = ticker.FormatStrFormatter(frmt_clb)
    if not log_cbar:
        cbar = fig.colorbar(cs, boundaries = clevs, pad=0.05, orientation="horizontal", format=fmt_clb) # add colorbar
    else:
        cbar = fig.colorbar(cs, boundaries = clevs, pad=0.05, orientation="horizontal", format=fmt_clb) # add colorbar
    cbar.ax.tick_params(labelsize=18) 
    if (not cbar_label is None): cbar.set_label(cbar_label, size=20)
    
    #plot line
    if not line_latlon is None:
        ax.plot(line_latlon[:,1], line_latlon[:,0], latlon = True, linewidth=3, color='k', zorder= 5 )

    #plot points
    if not pt_latlondata is None:
        if np.size(pt_latlondata,1) == 2:
            ax.plot(pt_latlondata[:,1], pt_latlondata[:,0],  'o', latlon=True, color = 'k', markersize = 4, zorder = 8)
        elif np.size(pt_latlondata,1) == 2:
            raise ValueError('Unimplemented plotting option')

    #add costal lines
    ax.coastlines(resolution=plt_res, edgecolor='black', zorder=5);
    #add state boundaries
    states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
                                          scale=plt_scale, facecolor='none')
    ax.add_feature(states, edgecolor='black', zorder=3)
    ax.add_feature(cfeature.BORDERS, zorder=4)
    #add oceans
    oceans = cfeature.NaturalEarthFeature(category='physical', name='ocean', facecolor='lightblue',
                                          scale=plt_scale)
    ax.add_feature(oceans, zorder=6)
    
    #add figure title
    if (not title is None): plt.title(title, fontsize=25)
    plt.xlabel('Latitude (deg)', fontsize=20)
    plt.ylabel('Longitude (deg)', fontsize=20)
    
    #grid lines
    if flag_grid:
        # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
    else:
        gl = None
    
    fig.tight_layout()
    
    
    return fig, ax, cbar, data_crs, gl

# Updated PlotContourCAMap function
#----  ----  ----  ----  ----  ----  ----
def PlotContourCAMap(cont_latlondata, cmin=None,  cmax=None, flag_grid=False, title=None, cbar_label=None, log_cbar = False, 
                     frmt_clb = '%.2f', cmap = 'viridis', **kwargs):
    '''
    PlotContourCAMap:
        simplifed function to create a contour plot of the data in cont_latlondata
        
    Input Arguments:
        cont_latlondata (np.array [n1,3]):       contains the latitude, logitude and contour values
                                                 cont_latlondata = [lat, long, data]
        cmin (double-opt):                       lower limit for color levels for contour plot 
        cmax (double-opt):                       upper limit for color levels for contour plot 
        title (str-opt):                         figure title
        cbar_label (str-opt):                    contour plot color bar label
        ptlevs (np.array-opt):                   color levels for points
        pt_label (str-opt):                      points color bar label
        log_cbar (bool-opt):                     if true use log-scale for contour plots
        frmt_clb                                 string format color bar ticks
    
    Output Arguments:
        
    '''
    #additional input arguments
    flag_smooth  = kwargs['flag_smooth']   if 'flag_smooth'   in kwargs else False
    sig_smooth   = kwargs['smooth_sig']    if 'smooth_sig'    in kwargs else 0.1
    intrp_method = kwargs['intrp_method']  if 'intrp_method'  in kwargs else 'linear'
        
    plt_res = '50m'
    plt_scale = '50m'

    #number of interpolation points, x & y direction
    #ngridx = 5000
    #ngridy = 5000
    ngridx = 500
    ngridy = 500
    #ngridx = 100
    #ngridy = 100

    #create figure
    fig = plt.figure(figsize=(10, 10))
    #fig = plt.figure(figsize=(15, 15))
    #create basemap
    data_crs = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=data_crs)
    
    #project contour data
    x_cont = cont_latlondata[:,1] 
    y_cont = cont_latlondata[:,0]
    #interpolation grid
    x_int = np.linspace(x_cont.min(), x_cont.max(), ngridx)
    y_int = np.linspace(y_cont.min(), y_cont.max(), ngridy)
    X_grid, Y_grid = np.meshgrid(x_int, y_int)
    #interpolate contour data on grid
    if log_cbar:
        data_cont = np.log(cont_latlondata[:,2])
    else:
        data_cont = cont_latlondata[:,2]
    data_grid = griddata((x_cont, y_cont) , data_cont, (X_grid, Y_grid), method=intrp_method )
    #smooth 
    if flag_smooth: 
        data_grid = gaussian_filter(data_grid, sigma=sig_smooth)     
    #data colorbar
    cbmin = data_cont.min() if cmin is None else cmin
    cbmax = data_cont.max() if cmax is None else cmax
    clevs = np.linspace(cbmin, cbmax, 41).tolist()    
    
    #plot interpolated data    
    cs =  ax.contourf(X_grid, Y_grid, data_grid, transform = data_crs, vmin=cmin, vmax=cmax, 
                      levels = clevs, zorder=3, alpha = 0.75, cmap=cmap)
        
    #color bar
    #import pdb; pdb.set_trace() 
    fmt_clb = ticker.FormatStrFormatter(frmt_clb)
    cbar_ticks = clevs[0:41:10]
    cbar = fig.colorbar(cs, boundaries=clevs, ticks=cbar_ticks, pad=0.05, orientation="horizontal", format=fmt_clb) # add colorbar
    if log_cbar:
        cbar_labels = [frmt_clb%np.exp(c_t) for c_t in cbar_ticks]
        cbar.set_ticklabels(cbar_labels)

    cbar.ax.tick_params(labelsize=18) 
    if (not cbar_label is None): cbar.set_label(cbar_label, size=20)

    #add costal lines
    ax.coastlines(resolution=plt_res, edgecolor='black', zorder=5);
    #add state boundaries
    states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
                                          scale=plt_scale, facecolor='none')
    ax.add_feature(states, edgecolor='black', zorder=3)
    borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_countries', 
                                           scale=plt_scale, facecolor='none')
    ax.add_feature(borders, edgecolor='black', zorder=4)
    #add oceans
    oceans = cfeature.NaturalEarthFeature(category='physical', name='ocean', scale=plt_scale)
    ax.add_feature(oceans, zorder=6)
    
    #add figure title
    if (not title is None): plt.title(title, fontsize=25)
    plt.xlabel('Latitude (deg)', fontsize=20)
    plt.ylabel('Longitude (deg)', fontsize=20)
    
    #grid lines
    if flag_grid:
        # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
    else:
        gl = None
    
    # fig.show()
    # fig.draw()
    fig.tight_layout()
    
    return fig, ax, cbar, data_crs, gl


# PlotContourSloveniaMap function
#----  ----  ----  ----  ----  ----  ----
def PlotContourSloveniaMap(cont_latlondata, cmin=None,  cmax=None, flag_grid=False, title=None, cbar_label=None, log_cbar = False, 
                           frmt_clb = '%.2f', **kwargs):
    '''
    PlotContourCAMap:
        simplifed create a contour plot of the data in cont_latlondata
        
    Input Arguments:
        cont_latlondata (np.array [n1,3]):       contains the latitude, logitude and contour values
                                                 cont_latlondata = [lat, long, data]
        cmin (double-opt):                       lower limit for color levels for contour plot 
        cmax (double-opt):                       upper limit for color levels for contour plot 
        title (str-opt):                         figure title
        cbar_label (str-opt):                    contour plot color bar label
        ptlevs (np.array-opt):                   color levels for points
        pt_label (str-opt):                      points color bar label
        log_cbar (bool-opt):                     if true use log-scale for contour plots
        frmt_clb                                 string format color bar ticks
    
    Output Arguments:
        
    '''
    
    plt_res = '50m'
    plt_scale = '50m'

    #number of interpolation points, x & y direction
    #ngridx = 5000
    #ngridy = 5000
    #ngridx = 500
    #ngridy = 500
    ngridx = 100
    ngridy = 100

    #create figure
    fig = plt.figure(figsize=(10, 10))
    #fig = plt.figure(figsize=(15, 15))
    #create basemap
    data_crs = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=data_crs)
    
    #project contour data
    x_cont = cont_latlondata[:,1] 
    y_cont = cont_latlondata[:,0]
    #interpolation grid
    x_int = np.linspace(x_cont.min(), x_cont.max(), ngridx)
    y_int = np.linspace(y_cont.min(), y_cont.max(), ngridy)
    X_grid, Y_grid = np.meshgrid(x_int, y_int)
    #interpolate contour data on grid
    if log_cbar:
        data_cont = np.log(cont_latlondata[:,2])
    else:
        data_cont = cont_latlondata[:,2]
    data_grid = griddata((x_cont, y_cont) , data_cont, (X_grid, Y_grid), method='linear')
    #smooth 
    if (kwargs['flag_smooth'] if 'flag_smooth' in kwargs else False): 
        sig_smooth = kwargs['smooth_sig'] if 'smooth_sig' in kwargs else 0.1
        data_grid = gaussian_filter(data_grid, sigma=sig_smooth)   
    #data colorbar
    cbmin = data_cont.min() if cmin is None else cmin
    cbmax = data_cont.max() if cmax is None else cmax
    clevs = np.linspace(cbmin, cbmax, 41).tolist()    
    
    #plot interpolated data    
    cs =  ax.contourf(X_grid, Y_grid, data_grid, transform = data_crs, vmin=cmin, vmax=cmax, levels = clevs, zorder=3, alpha = 0.75)
        
    #color bar
 
    fmt_clb = ticker.FormatStrFormatter(frmt_clb)
    cbar_ticks = clevs[0:41:8]
    cbar = fig.colorbar(cs, boundaries=clevs, ticks=cbar_ticks, pad=0.05, orientation="horizontal", format=fmt_clb) # add colorbar
    if log_cbar:
        cbar_labels = [frmt_clb%np.exp(c_t) for c_t in cbar_ticks]
        cbar.set_ticklabels(cbar_labels)

    cbar.ax.tick_params(labelsize=18) 
    if (not cbar_label is None): cbar.set_label(cbar_label, size=20)

    #add costal lines
    ax.coastlines(resolution=plt_res, edgecolor='black', zorder=5);
    #add state boundaries
    #states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
    #                                      scale=plt_scale, facecolor='none')
    #ax.add_feature(states, edgecolor='black', zorder=3)
    borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_countries', 
                                           scale=plt_scale, facecolor='none')
    ax.add_feature(borders, edgecolor='black', zorder=4)
    #ax.add_feature(cfeature.BORDERS, zorder=4)
    #add oceans
    oceans = cfeature.NaturalEarthFeature(category='physical', name='ocean', facecolor='lightblue',
                                          scale=plt_scale)
    ax.add_feature(oceans, zorder=6)
    
    #add figure title
    if (not title is None): plt.title(title, fontsize=25)
    plt.xlabel('Latitude (deg)', fontsize=20)
    plt.ylabel('Longitude (deg)', fontsize=20)
    
    #grid lines
    if flag_grid:
        # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
    else:
        gl = None
        
    # fig.show()
    # fig.draw()
    fig.tight_layout()
    
    return fig, ax, cbar, data_crs, gl

# Scatter plot function
#----  ----  ----  ----  ----  ----  ----
def PlotScatterCAMap(scat_latlondata, cmin=None,  cmax=None, flag_grid=False, title=None, cbar_label=None, log_cbar = False, 
                     frmt_clb = '%.2f', alpha_v = 0.7, cmap='seismic', marker_size=10.):
    '''
    PlotContourCAMap:
        create a contour plot of the data in cont_latlondata
        
    Input Arguments:
        scat_latlondata (np.array [n1,(3,4)]):   contains the latitude, logitude, contour values, and size values (optional)
                                                 scat_latlondata = [lat, long, data_color, data_size]
        cmin (double-opt):                       lower limit for color levels for contour plot 
        cmax (double-opt):                       upper limit for color levels for contour plot 
        title (str-opt):                         figure title
        cbar_label (str-opt):                    contour plot color bar label
        ptlevs (np.array-opt):                   color levels for points
        pt_label (str-opt):                      points color bar label
        log_cbar (bool-opt):                     if true use log-scale for contour plots
        frmt_clb:                                string format color bar ticks
        alpha_v:                                 opacity value
        cmap: 					   color palette
        marker_size:				   marker size, if scat_latlondata dimensions is [n1, 3]
    
    Output Arguments:
        
    '''
    
    #import pdb; pdb.set_trace()
    
    plt_res = '10m'
    plt_scale = '10m'

    #create figure
    fig = plt.figure(figsize=(10, 10))
    #fig = plt.figure(figsize=(15, 15))
    #create basemap
    data_crs = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=data_crs)
    
    #project contour data
    x_scat = scat_latlondata[:,1] 
    y_scat = scat_latlondata[:,0]

    #color scale
    if log_cbar:
        data_scat_c = np.log(scat_latlondata[:,2])
    else:
        data_scat_c = scat_latlondata[:,2]
        
    #size scale
    if scat_latlondata.shape[1] > 3:
        data_scat_s = scat_latlondata[:,3]
    else:
        data_scat_s = marker_size * np.ones(data_scat_c.shape)
        
    #data colorbar
    cbmin = data_scat_c.min() if cmin is None else cmin
    cbmax = data_scat_c.max() if cmax is None else cmax
    clevs = np.linspace(cbmin, cbmax, 41).tolist()    
    
    #plot scatter bubble plot data    
    cs =  ax.scatter(x_scat, y_scat, s = data_scat_s, c = data_scat_c, 
                     transform = data_crs, vmin=cmin, vmax=cmax, zorder=3, alpha=alpha_v, cmap=cmap)
        
    #color bar
    #import pdb; pdb.set_trace() 
    fmt_clb = ticker.FormatStrFormatter(frmt_clb)
    cbar_ticks = clevs[0:41:8]
    cbar = fig.colorbar(cs, boundaries=clevs, ticks=cbar_ticks, pad=0.05, orientation="horizontal", format=fmt_clb) # add colorbar
    if log_cbar:
        cbar_labels = [frmt_clb%np.exp(c_t) for c_t in cbar_ticks]
        cbar.set_ticklabels(cbar_labels)

    cbar.ax.tick_params(labelsize=18) 
    if (not cbar_label is None): cbar.set_label(cbar_label, size=20)

    #add costal lines
    ax.coastlines(resolution=plt_res, edgecolor='black', zorder=5);
    #add state boundaries
    states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
                                          scale=plt_scale, facecolor='none')
    ax.add_feature(states, edgecolor='black', zorder=3)
    ax.add_feature(cfeature.BORDERS, zorder=4)
    #oceans
    oceans = cfeature.NaturalEarthFeature(category='physical', name='ocean', facecolor='lightblue',
                                          scale=plt_scale)
    ax.add_feature(oceans, zorder=2)
    
    #add figure title
    if (not title is None): plt.title(title, fontsize=25)
    plt.xlabel('Latitude (deg)', fontsize=20)
    plt.ylabel('Longitude (deg)', fontsize=20)

    #grid lines
    if flag_grid:
        # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
    else:
        gl = None
    
    # fig.show()
    # fig.draw()
    fig.tight_layout()
    
    return fig, ax, cbar, data_crs, gl

# Updated PlotContourCAMap function
#----  ----  ----  ----  ----  ----  ----
def PlotCellsCAMap(cell_latlondata, cmin=None,  cmax=None, flag_grid=False, title=None, cbar_label=None, log_cbar = False, frmt_clb = '%.2f',
                   alpha_v = .8, cell_size = 50, cmap='seismic'):
    '''
    PlotCellsCAMap:
        PlotCellsCAMap function to create a contour plot of the data in cont_latlondata
        
    Input Arguments:
        cell_latlondata (np.array [n1,3]):       contains the latitude, logitude and color values
                                                 cell_latlondata = [lat, long, data]
        cmin (double-opt):                       lower limit for color levels for contour plot 
        cmax (double-opt):                       upper limit for color levels for contour plot 
        title (str-opt):                         figure title
        cbar_label (str-opt):                    contour plot color bar label
        ptlevs (np.array-opt):                   color levels for points
        pt_label (str-opt):                      points color bar label
        log_cbar (bool-opt):                     if true use log-scale for contour plots
        frmt_clb                                 string format color bar ticks
    
    Output Arguments:
        
    '''
    
    plt_res = '50m'
    plt_scale = '50m'

    #create figure
    fig = plt.figure(figsize=(10, 10))
    #create basemap
    data_crs = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=data_crs)
    
    #project contour data
    x_cell = cell_latlondata[:,1] 
    y_cell = cell_latlondata[:,0]

    #contour transfomration
    if log_cbar:
        data_cell = np.log(cell_latlondata[:,2])
    else:
        data_cell = cell_latlondata[:,2]
    #data colorbar
    cbmin = data_cell.min() if cmin is None else cmin
    cbmax = data_cell.max() if cmax is None else cmax
    clevs = np.linspace(cbmin, cbmax, 41).tolist()    
    
    #plot interpolated data
    cs =  ax.scatter(x_cell, y_cell, s = cell_size, c = data_cell, transform = data_crs, vmin=cmin, vmax=cmax, zorder=3,
                     alpha = alpha_v, cmap=cmap)
   #cs =  ax.contourf(X_grid, Y_grid, data_grid, transform = data_crs, vmin=cmin, vmax=cmax, levels = clevs, zorder=3, alpha = 0.75)
        
    #color bar
    #import pdb; pdb.set_trace() 
    fmt_clb = ticker.FormatStrFormatter(frmt_clb)
    cbar_ticks = clevs[0:41:8]
    cbar = fig.colorbar(cs, boundaries=clevs, ticks=cbar_ticks, pad=0.05, orientation="horizontal", format=fmt_clb) # add colorbar
    if log_cbar:
        cbar_labels = [frmt_clb%np.exp(c_t) for c_t in cbar_ticks]
        cbar.set_ticklabels(cbar_labels)

    cbar.ax.tick_params(labelsize=18) 
    if (not cbar_label is None): cbar.set_label(cbar_label, size=20)

    #add costal lines
    ax.coastlines(resolution=plt_res, edgecolor='black', zorder=5);
    #add state boundaries
    states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
                                          scale=plt_scale, facecolor='none')
    ax.add_feature(states, edgecolor='black', zorder=3)
    borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_countries', 
                                           scale=plt_scale, facecolor='none')
    ax.add_feature(borders, edgecolor='black', zorder=4)
    #add oceans
    #ax.stock_img()
    oceans = cfeature.NaturalEarthFeature(category='physical', name='ocean', facecolor='lightblue',
                                          scale=plt_scale)
    ax.add_feature(oceans, zorder=2)
    
    #add figure title
    if (not title is None): ax.set_title(title, fontsize=25)
    ax.set_xlabel('Latitude (deg)', fontsize=20)
    ax.set_ylabel('Longitude (deg)', fontsize=20)
    
    #grid lines
    if flag_grid:
        # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
    else:
        gl = None
    
    # fig.show()
    # fig.draw()
    fig.tight_layout()
    
    return fig, ax, cbar, data_crs, gl

# Plotting coefficient function
#----  ----  ----  ----  ----  ----  ----
#plotting of median values coefficients
def PlotCoeffCAMapMed(cont_latlondata, cmin=None, cmax=None, flag_grid=False, title=None, cbar_label=None, log_cbar = False, frmt_clb = '%.2f', **kwargs):
    
    cmap = 'seismic'
    fig, ax, cbar, data_crs, gl = PlotContourCAMap(cont_latlondata, cmin=cmin, cmax=cmax, flag_grid=flag_grid, title=title, cbar_label=cbar_label, 
                                                   log_cbar = log_cbar, frmt_clb = frmt_clb, cmap = cmap, **kwargs)
    
    return fig, ax, cbar, data_crs, gl

#plotting of epistemic uncertainty coefficients
def PlotCoeffCAMapSig(cont_latlondata, cmin=None, cmax=None, flag_grid=False, title=None, cbar_label=None, log_cbar = False, frmt_clb = '%.2f', **kwargs):
    
    cmap = 'Purples_r'
    fig, ax, cbar, data_crs, gl = PlotContourCAMap(cont_latlondata, cmin=cmin,  cmax=cmax, flag_grid=flag_grid, title=title, cbar_label=cbar_label, 
                                                   log_cbar = log_cbar, frmt_clb = frmt_clb, cmap = cmap, **kwargs)
    
    return fig, ax, cbar, data_crs, gl

#plotting of median values of cells
def PlotCellsCAMapMed(cell_latlondata, cmin=None, cmax=None, flag_grid=False, title=None, cbar_label=None, log_cbar = False, frmt_clb = '%.2f',
                      alpha_v = .8, cell_size = 50):
    
    cmap = 'seismic'
    fig, ax, cbar, data_crs, gl = PlotCellsCAMap(cell_latlondata, cmin=cmin,  cmax=cmax, flag_grid=flag_grid, title=title, cbar_label=cbar_label, 
                                                 log_cbar=log_cbar, frmt_clb=frmt_clb,
                                                 alpha_v=alpha_v, cell_size=cell_size, cmap=cmap)
    
    return fig, ax, cbar, data_crs, gl

#plotting of mono-color increasing values of cells
def PlotCellsCAMapInc(cell_latlondata, cmin=None, cmax=None, flag_grid=False, title=None, cbar_label=None, log_cbar = False, frmt_clb = '%.2f',
                      alpha_v = .8, cell_size = 50):
    
    cmap = 'Reds'
    fig, ax, cbar, data_crs, gl = PlotCellsCAMap(cell_latlondata, cmin=cmin,  cmax=cmax, flag_grid=flag_grid, title=title, cbar_label=cbar_label, 
                                                 log_cbar=log_cbar, frmt_clb=frmt_clb,
                                                 alpha_v=alpha_v, cell_size=cell_size, cmap=cmap)
    
    return fig, ax, cbar, data_crs, gl

#plotting of epistemic uncertainty of cells
def PlotCellsCAMapSig(cell_latlondata, cmin=None, cmax=None, flag_grid=False, title=None, cbar_label=None, log_cbar = False, frmt_clb = '%.2f',
                      alpha_v = .8, cell_size = 50):
    
    cmap = 'Purples_r'
    fig, ax, cbar, data_crs, gl = PlotCellsCAMap(cell_latlondata, cmin=cmin,  cmax=cmax, flag_grid=flag_grid, title=title, cbar_label=cbar_label, 
                                                 log_cbar=log_cbar, frmt_clb=frmt_clb,
                                                 alpha_v=alpha_v, cell_size=cell_size, cmap=cmap )
    
    return fig, ax, cbar, data_crs, gl


# Base plot function
#----  ----  ----  ----  ----  ----  ----
def PlotMap(lat_lims = None, lon_lims = None, flag_grid=False, title=None):
    '''
    PlotContourCAMap:
        simplifed function to create a contour plot of the data in cont_latlondata
        
    Input Arguments:
        line_latlondata (np.array [n1,3]):       contains the latitude, logitude and contour values
                                                 cont_latlondata = [lat, long, data]
        cmin (double-opt):                       lower limit for color levels for contour plot 
        cmax (double-opt):                       upper limit for color levels for contour plot 
        title (str-opt):                         figure title
        cbar_label (str-opt):                    contour plot color bar label
        ptlevs (np.array-opt):                   color levels for points
        pt_label (str-opt):                      points color bar label
        log_cbar (bool-opt):                     if true use log-scale for contour plots
        frmt_clb                                 string format color bar ticks
    
    Output Arguments:
        
    '''
    
    plt_res = '50m'
    plt_scale = '50m'

    #create figure
    fig = plt.figure(figsize=(10, 10))
    #fig = plt.figure(figsize=(15, 15))
    #create basemap
    data_crs = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=data_crs)
    
    if lat_lims:
        ax.set_xlim(lon_lims)
    if lon_lims:
        ax.set_ylim(lat_lims)

    #add land zones
    lands = cfeature.LAND
    ax.add_feature(lands, zorder=1)
    #add costal lines
    ax.coastlines(resolution=plt_res, edgecolor='black', zorder=3);
    #add state boundaries
    states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
                                          scale=plt_scale, facecolor='none')
    ax.add_feature(states, edgecolor='black', zorder=4)
    borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_countries', 
                                           scale=plt_scale, facecolor='none')
    ax.add_feature(borders, edgecolor='black', zorder=5)
    #add oceans
    oceans = cfeature.NaturalEarthFeature(category='physical', name='ocean', facecolor='lightblue',
                                          scale=plt_scale)
    ax.add_feature(oceans, zorder=2)
    
    #add figure title
    if (not title is None): plt.title(title, fontsize=25)
    plt.xlabel('Latitude (deg)', fontsize=20)
    plt.ylabel('Longitude (deg)', fontsize=20)
    
    #grid lines
    if flag_grid:
        # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
    else:
        gl = None
    
    # fig.show()
    # fig.draw()
    # fig.tight_layout()
    
    return fig, ax, data_crs, gl
