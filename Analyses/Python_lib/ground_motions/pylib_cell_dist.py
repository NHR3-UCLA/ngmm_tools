#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:25:10 2020

@author: glavrent
"""

#load libraries
import numpy as np
import geopy.distance as geopydist

def ComputeDistUnGridCells(pt1, pt2, cells, diffx, diffy, flagUTM=False):
    '''
    Compute the path distances of uniformly gridded cells

    Parameters
    ----------
    pt1 : np.array(3)
        Latitude, Longitude, elevation coordinates of first point.
    pt2 : np.array(3)
        Latitude, Longitude, elevation coordinates of second point.
    cells : np.array(n_cells, 4)
        Cell coordinates: Cartesian or LatLon
            Latitude, Longitude, bottom and top elevation of cells
            [x, y, elev_bot, elev_top]        
        Lat Lon coordinates:
            Latitude, Longitude, bottom and top elevation of cells
            [lon, lat, elev_bot, elev_top]
    diffx : real
        Latitude interval of cells.
    diffy : real
        Longitude interval of cells.

    Returns
    -------
    dm : np.array(n_cells)
        Distance path on each cell.

    '''
    
    #import pdb; pdb.set_trace()

    #grid points     
    x_grid = np.unique(cells[:, 0])
    y_grid = np.unique(cells[:, 1])
    z_grid = np.unique(cells[:, 2])

    ## find x,y,z grid points which are between source and site
    x_v = np.sort([pt1[0], pt2[0]])
    x_g_pts = x_grid[(x_v[0] <= x_grid) & (x_grid < x_v[1])]

    y_v = np.sort([pt1[1], pt2[1]])
    y_g_pts = y_grid[(y_v[0] <= y_grid) & (y_grid < y_v[1])]

    z_v = np.sort([pt1[2], pt2[2]])
    z_g_pts = z_grid[(z_v[0] <= z_grid) & (z_grid < z_v[1])]

    #p1-pt2 vector
    vec = np.subtract(pt1, pt2)

    # intersection points for x
    normal = [1, 0, 0];
    ptx = np.ones(len(x_g_pts) * 3)
    if len(x_g_pts) > 0:
        ptx = ptx.reshape(len(x_g_pts), 3)
        for i, xv in enumerate(x_g_pts):
            ptplane = [xv, y_grid[0], 0]
            d = np.divide(np.dot(np.subtract(ptplane,pt1),normal), np.dot(vec,normal))
            pt = pt1 + d * vec
            ptx[i] = pt
    else:
        ptx = [[-999, -999, -999]]

    # intersection points for y
    normal = [0, 1, 0];
    pty = np.ones(len(y_g_pts) * 3)
    if len(y_g_pts) > 0:
        pty = pty.reshape(len(y_g_pts), 3)
        for i, yv in enumerate(y_g_pts):
            ptplane = [x_grid[0], yv, 0]
            d = np.divide(np.dot(np.subtract(ptplane,pt1),normal), np.dot(vec,normal))
            pt = pt1 + d * vec
            pty[i] = pt
    else:
        pty = [[-999, -999, -999]]

    # intersection points for z
    normal = [0, 0, 1]
    ptz = np.ones(len(z_g_pts) * 3)
    if len(z_g_pts) > 0:
        ptz = ptz.reshape(len(z_g_pts), 3)
        for i, zv in enumerate(z_g_pts):
            ptplane = [x_grid[0], y_grid[0], zv]
            d = np.divide(np.dot(np.subtract(ptplane,pt1),normal), np.dot(vec,normal))
            pt = pt1 + d * vec
            ptz[i] = pt
    else:
        ptz = [[-999, -999, -999]]

    #summarize all intersection points
    ptall = np.concatenate(([pt1], [pt2], ptx, pty, ptz))
    ptall = ptall[(ptall[:, 0] != -999) & (ptall[:, 1] != -999) & (ptall[:, 2] != -999)]
    ptall = np.unique(ptall, axis=0)
    if pt1[0] != pt2[0]: 
        ptall = ptall[ptall[:, 0].argsort()] #sort points by x coordinate
    else:
        ptall = ptall[ptall[:, 1].argsort()] #sort points by y coordinate

    #cell ids
    id_cells = np.arange(len(cells))

    #compute cell distance
    idx = np.zeros(len(ptall)-1)
    distances = np.ones(len(ptall)-1)
    for i in range(len(ptall) - 1):
        p1 = ptall[i]       #first intersection point
        p2 = ptall[i+1]     #second intersection point

        #cell indices of cells where the first intersection point belongs 
        idx1 = id_cells[(cells[:, 0] <= p1[0]) & (p1[0] <= cells[:, 0] + diffx) & \
                        (cells[:, 1] <= p1[1]) & (p1[1] <= cells[:, 1] + diffy) & \
                        (cells[:, 2] <= p1[2]) & (p1[2] <= cells[:, 3])]
        #cell indices of cells where the second intersection point belongs 
        idx2 = id_cells[(cells[:, 0] <= p2[0]) & (p2[0] <= cells[:, 0] + diffx) & \
                        (cells[:, 1] <= p2[1]) & (p2[1] <= cells[:, 1] + diffy) & \
                        (cells[:, 2] <= p2[2]) & (p2[2] <= cells[:, 3])]
        #common indices of first and second int points
        idx[i] = np.intersect1d(idx1, idx2)

        #compute path distance
        if not flagUTM:
            dxy = geopydist.distance(ptall[i,(1,0)],ptall[i + 1,(1,0)]).km
        else:
            dxy = np.linalg.norm(ptall[i,0:1] - ptall[i + 1,0:1])
        dz = ptall[i,2] - ptall[i + 1,2]
        distances[i] = np.sqrt(dxy** 2 + dz ** 2)

    dm = np.zeros(len(cells))
    dm[idx.astype(int)] = distances

    return dm

def ComputeDistGridCells(pt1, pt2, cells, flagUTM=False):
    '''
    Compute the path distances of gridded cells

    Parameters
    ----------
    pt1 : np.array(3)
        Latitude, Longitude, elevation coordinates of first point.
    pt2 : np.array(3)
        Latitude, Longitude, elevation coordinates of second point.
    cells : np.array(n_cells, 6)
        Latitude, Longitude, elevation of bottom left (q1) and top right (q8) corrners of cells
        [q1_lat, q1_lon, q1_elev, q8_lat, q8_lon, q8_elev]
    diffx : real
        Latitude interval of cells.
    diffy : real
        Longitude interval of cells.

    Returns
    -------
    dm : np.array(n_cells)
        Distance path on each cell.

    '''
    
    #import pdb; pdb.set_trace()

    #grid points     
    x_grid = np.unique(cells[:, 0])
    y_grid = np.unique(cells[:, 1])
    z_grid = np.unique(cells[:, 2])

    ## find x,y,z grid points which are between source and site
    x_v = np.sort([pt1[0], pt2[0]])
    x_g_pts = x_grid[(x_v[0] <= x_grid) & (x_grid < x_v[1])]

    y_v = np.sort([pt1[1], pt2[1]])
    y_g_pts = y_grid[(y_v[0] <= y_grid) & (y_grid < y_v[1])]

    z_v = np.sort([pt1[2], pt2[2]])
    z_g_pts = z_grid[(z_v[0] <= z_grid) & (z_grid < z_v[1])]

    #p1-pt2 vector
    vec = np.subtract(pt1, pt2)

    # intersection points for x
    normal = [1, 0, 0];
    ptx = np.ones(len(x_g_pts) * 3)
    if len(x_g_pts) > 0:
        ptx = ptx.reshape(len(x_g_pts), 3)
        for i, xv in enumerate(x_g_pts):
            ptplane = [xv, y_grid[0], 0]
            d = np.divide(np.dot(np.subtract(ptplane,pt1),normal), np.dot(vec,normal))
            pt = pt1 + d * vec
            ptx[i] = pt
    else:
        ptx = [[-999, -999, -999]]

    # intersection points for y
    normal = [0, 1, 0];
    pty = np.ones(len(y_g_pts) * 3)
    if len(y_g_pts) > 0:
        pty = pty.reshape(len(y_g_pts), 3)
        for i, yv in enumerate(y_g_pts):
            ptplane = [x_grid[0], yv, 0]
            d = np.divide(np.dot(np.subtract(ptplane,pt1),normal), np.dot(vec,normal))
            pt = pt1 + d * vec
            pty[i] = pt
    else:
        pty = [[-999, -999, -999]]

    # intersection points for z
    normal = [0, 0, 1]
    ptz = np.ones(len(z_g_pts) * 3)
    if len(z_g_pts) > 0:
        ptz = ptz.reshape(len(z_g_pts), 3)
        for i, zv in enumerate(z_g_pts):
            ptplane = [x_grid[0], y_grid[0], zv]
            d = np.divide(np.dot(np.subtract(ptplane,pt1),normal), np.dot(vec,normal))
            pt = pt1 + d * vec
            ptz[i] = pt
    else:
        ptz = [[-999, -999, -999]]

    #summarize all intersection points
    ptall = np.concatenate(([pt1], [pt2], ptx, pty, ptz))
    ptall = ptall[(ptall[:, 0] != -999) & (ptall[:, 1] != -999) & (ptall[:, 2] != -999)]
    #ptall = np.unique(ptall.round, axis=0, return_index=True)
    _, i_ptall_unq = np.unique(ptall.round(decimals=7), axis=0, return_index=True)
    ptall = ptall[i_ptall_unq,:]
    # if pt1[0] != pt2[0]:
    if abs(pt1[0] - pt2[0]) > 1e-6:
        ptall = ptall[ptall[:, 0].argsort()]
    else:
        ptall = ptall[ptall[:, 1].argsort()]

    #compute cell distance
    id_cells = np.arange(len(cells))
    idx = np.ones(len(ptall)-1)
    distances = np.ones(len(ptall)-1)
    for i in range(len(ptall) - 1):
        p1 = ptall[i]       #first intersection point
        p2 = ptall[i+1]     #second intersection point

        #cell indices where the first point belongs 
        tol = 1e-9
        idx1 = id_cells[(cells[:, 0]-tol <= p1[0]) & (p1[0] <= cells[:, 3]+tol) & \
                        (cells[:, 1]-tol <= p1[1]) & (p1[1] <= cells[:, 4]+tol) & \
                        (cells[:, 2]-tol <= p1[2]) & (p1[2] <= cells[:, 5]+tol)]
        #cell indices where the second point belongs 
        idx2 = id_cells[(cells[:, 0]-tol <= p2[0]) & (p2[0] <= cells[:, 3]+tol) & \
                        (cells[:, 1]-tol <= p2[1]) & (p2[1] <= cells[:, 4]+tol) & \
                        (cells[:, 2]-tol <= p2[2]) & (p2[2] <= cells[:, 5]+tol)]
    
        #common indices of first and second point
        try:
            idx[i] = np.intersect1d(idx1, idx2)
        except ValueError:
            print('i_pt: ', i)
            print('idx1: ', idx1)
            print('idx2: ', idx2)
            print('p1: ', p1)
            print('p2: ', p2)
            # import pdb; pdb.set_trace()
            raise
            

        #compute path distance
        if not flagUTM:
            dxy = geopydist.distance(ptall[i,(1,0)],ptall[i + 1,(1,0)]).km
        else:
            dxy = np.linalg.norm(ptall[i,0:2] - ptall[i + 1,0:2])
        dz = ptall[i,2] - ptall[i + 1,2]
        distances[i] = np.sqrt(dxy** 2 + dz ** 2)

    dm = np.zeros(len(cells))
    dm[idx.astype(int)] = distances

    return dm
