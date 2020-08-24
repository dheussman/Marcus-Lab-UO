#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:51:52 2019

This script displays all the 

@author: jkittell
"""
############# IMPORT STATEMENTS ##############
from JToolKit import *

# Tools
import numpy as np
import scipy as sp
from scipy.optimize import minimize, minimize_scalar # for minimization.

# Stop Warning Messages
import warnings
#warnings.filterwarnings('ignore')   # optional filtering of warnings..
###############################################

######### TEMPORARY VARIABLE DEFINITIONS #######
#nv = 2     # number of vibrational modes to consider
#ep = 18285      # elecronic transition energy
#hbar = 1.054e-34    # fund. constant
#lamb = 0.54**.5   # Huang-Rhys Parameter
#omega = 1116   # fund. vib. freq
################################################

########### VARIABLE DEFINITIONS ##############
Pi = np.pi

low = 16700
high = 22000 # Data ranges

time_step = 2.66e-15 
monochrom = 17857
################################################

data = []

for suffix in ('b','c',
               'd','e',
               'f','g',
               'h','i'):
    dat = import_2D_data("20181017", suffix, monochrom)
    data.append(dat)


print("data")
for dat in data:
    dat.FT(2.66e-15, 17534)
    dat.paper_plot()
    
#print("NRPs:")
#for nrp in nrps:
#    nrp.contour_plot(30, ((0, 50e-15),(0, 50e-15)))
#    nrp.FT(time_step, monochrom)
#    nrp.contour_plot(30)
#    
#    
    
