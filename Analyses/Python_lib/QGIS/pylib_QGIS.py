#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:04:00 2020

@author: glavrent
"""

#load libraries

#load GIS
from qgis.core import QgsVectorLayer, QgsPointXY
from qgis.core import QgsField, QgsFeature, QgsGeometry, QgsVectorFileWriter, QgsFeatureSink
from qgis.PyQt.QtCore import QVariant

def EQLayer(eq_data):
    '''
    Create earthquake source layer for QGIS

    Parameters
    ----------
    eq_data : pd.dataframe
        Dataframe for rupture points with fields:
            eqid, region, mag, SOF, Ztor, eqLat, eqLon
    
    Returns
    -------
    eq_layer : TYPE
        QGIS layer with earthquake sources.
    '''

    #create qgis layer for earthquake sources
    eq_layer = QgsVectorLayer("Point", "eq_pts", "memory")
    eq_pr = eq_layer.dataProvider()
    eq_pr.addAttributes([QgsField("eqid",      QVariant.Int),
                         QgsField("region",    QVariant.Int),
                         QgsField("mag",       QVariant.Double),
                         QgsField("SOF",       QVariant.Int),
                         QgsField("Ztor",      QVariant.Double),
                         QgsField("eqLat",     QVariant.Double),
                         QgsField("eqLon",     QVariant.Double)])

    #iterate over earthquakes, add on layer
    eq_layer.startEditing()
    for eq in eq_data.iterrows():
        #earthquake info
        eq_info   = eq[1][['eqid','region','mag','SOF','Ztor']].tolist()
        eq_latlon = eq[1][['eqLat','eqLon']].tolist()
        #define feature, earthquake  
        eq_f = QgsFeature()
        eq_f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(eq_latlon[1],eq_latlon[0])))
        eq_f.setAttributes(eq_info + eq_latlon)
        #add earthquake in layer
        eq_pr.addFeatures([eq_f])
    #commit changes
    eq_layer.commitChanges()
    #update displacement layer
    eq_layer.updateExtents()   
    
    return eq_layer

def STALayer(sta_data):
    '''
    Create station layer for QGIS

    Parameters
    ----------
    sta_data : pd.dataframe
        Dataframe for rupture points with fields:
            'ssn','region','Vs30','Z1.0','StaLat','StaLon'
            eqid','region','mag','SOF','eqLat','eqLon'
    
    Returns
    -------
    sta_layer : TYPE
        QGIS layer with station points.
    '''

    #create qgis layer for station locations
    sta_layer = QgsVectorLayer("Point", "sta_pts", "memory")
    sta_pr = sta_layer.dataProvider()
    sta_pr.addAttributes([QgsField("ssn",       QVariant.Int),
                         QgsField("region",     QVariant.Int),
                         QgsField("Vs30",       QVariant.Double),
                         QgsField("Z1.0",       QVariant.Double),
                         QgsField("staLat",     QVariant.Double),
                         QgsField("staLon",     QVariant.Double)])

    #iterate over station, add on layer
    sta_layer.startEditing()
    for sta in sta_data.iterrows():
        #earthquake info
        sta_info   = sta[1][['ssn','region','Vs30','Z1.0']].tolist()
        sta_latlon = sta[1][['staLat','staLon']].tolist()
        #define feature, earthquake  
        sta_f = QgsFeature()
        sta_f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(sta_latlon[1],sta_latlon[0])))
        sta_f.setAttributes(sta_info + sta_latlon)
        #add earthquake in layer
        sta_pr.addFeatures([sta_f])
    #commit changes
    sta_layer.commitChanges()
    #update displacement layer
    sta_layer.updateExtents()   
    
    return sta_layer
