#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:14:55 2019

This document defines the the data object in terms of experimental variables.

X -> 

Y -> 

Z ->

t21 -> 

t32 -> 

laser_spectrum -> 
size should be thirty by thirty...



@author: jkittell
"""

def import_2D_data(directory, suffix, monochrom1, monochrom2 = None, time_step = 2.66e-15, make_plots = False):
    '''creates a 2D data object from a directory of 2D data files with sparate
    runs denoted by the suffix "a", "b", etc. built up from Amr's matlab file
    single2D.m
    str, str -> Data_2D, Data_2D'''
    if monochrom2 == None:
        monochrom2 = monochrom1
    
    # data #
    rp_real = ((np.genfromtxt(directory+"/x3"+suffix))[1:])
    rp_imag = ((np.genfromtxt(directory+"/y3"+suffix))[1:])
    nrp_real  = ((np.genfromtxt(directory+"/x13"+suffix))[1:])
    nrp_imag  = ((np.genfromtxt(directory+"/y13"+suffix))[1:])
    #print("shape ", rp_real.shape)
    #plt.matshow(nrp_real)
    #plt.matshow(rp_real)
    
    nrp = nrp_real + nrp_imag * 1j
    rp = rp_real + rp_imag * 1j
    
    #phase

    nrp_phase = np.exp(1j*(-np.angle(nrp[0,0])))
    rp_phase  = np.exp(1j*(-np.angle( rp[0,0])))
    
    # axes #
    
    t21 = np.genfromtxt(directory+"/t21"+suffix)*1e-15
    t43 = np.genfromtxt(directory+"/t43"+suffix)*1e-15
    #print(t21[:,0])

    #print(t43)
    
    t21 = abs(t21 - t21[0])
    t43 = abs(t43 - t43[0])
    #debugging
    data = Data_2D(-rp_phase * rp.T, -nrp_phase * nrp, (t21, t43))
    data.FT(time_step, monochrom1, monochrom2)
    data.normalize()
    
    if make_plots:
        data.paper_plot()
    return data
