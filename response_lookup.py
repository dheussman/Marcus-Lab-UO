#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:22:14 2019


2D LOOK UP TABLE FOR 30X30 TIME GRIDS

@author: jkittell
"""
############# Constants ##################
c0 = 2.9979e8
H = 6.626e-34
Hbar = 1.054e-34
nubar2nu = 100*c0
permFree = 8.8542e-12
J2nubar = 1/(100*c0*H)
############# IMPORT STATEMENTS ##############

# Tools
import scipy as sp
from scipy.optimize import minimize, minimize_scalar
import numpy as np
import math as ma
import matplotlib.pyplot as plt
import scipy.linalg as la
import sys
Pi = np.pi
# Time/date stamp runs:
from datetime import datetime as dt

## Stop Warning Messages
#import warnings
#warnings.filterwarnings('ignore')

###############################################
############# Basic Functions ##############
def kr(a,b): return sp.kron(a,b)
def kr4(a,b,c,d): return kr(kr(kr(a,b),c),d)
def kr2(a,b): return kr(a,b)
def dot(a,b): return np.dot(a,b)
def Dot(a,b): return np.dot(a,b)
def Cross(a,b): return np.cross(a,b)
###############################################
############# Helper Functions ##############
def ranlen(x): return range(len(x))
def npa(x): return np.array(x)
def mag(x): return ma.sqrt(sum(i**2 for i in x))
###############################################
######## DATABASE STUFF #################
import sqlitedict as DB


time_db = DB.SqliteDict("signal_lookup_bad.db")

time_step = 2.66e-15 
ax21 = npa(range(0, 30, 1))*time_step
ax43 = npa(range(0, 30, 1))*time_step

def response(w21, w43):
    '''simulates a 2d response function'''
    
    w21 *= nubar2nu * 2 * Pi
    w43 *= nubar2nu * 2 * Pi
    
    # standard grid:
    time_step = 2.66e-15 
    ax21 = npa(range(0, 30, 1))*time_step
    ax43 = npa(range(0, 30, 1))*time_step

    base_grid = np.zeros((30,30), dtype = complex)
    
    for i in ranlen(ax21):
            for j in ranlen(ax43): 
                base_grid[i, j] = np.exp(1j * (w43 * ax43[j] + w21 * ax21[i]))
                
    plt.matshow(np.real(base_grid))
    
    return base_grid


for w21 in range(-7000, 7000, 10):
    print("<>", end = "")
    for w43 in range(-7000, 7000, 10):
        
        time_db[str((w21, w43))] = response(w21, w43)
        break
        time_db.commit()
        break

time_db.close(force = True)
