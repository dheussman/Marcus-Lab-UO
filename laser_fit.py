#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:46:27 2019


Laser Spectrum analysis:

figure out paramteres for a laser trim.


@author: jkittell
"""
############# IMPORT STATEMENTS ##############
from JToolKit import *

# Tools
import numpy as np
import scipy as sp
from scipy.optimize import minimize, minimize_scalar # for minimization.
from scipy.integrate import quad


# Stop Warning Messages
import warnings
#warnings.filterwarnings('ignore')   # optional filtering of warnings..
###############################################



x_data, y_data = np.genfromtxt(
        "/Users/jkittell/Desktop/JToolKit_051419/20190417/laseratsample_File3_17-39-56-003.txt", 
        skip_header = 2).T  
x_data = (1/x_data) * 10**7
print(x_data)
        
def super_gauss(x, amp, mu, sig, n):
    return amp * np.exp(-((x-mu)**2/(2*sig**2))**n)
def x_super_gauss(x, amp, mu, sig, n):
    return x * amp * np.exp(-((x-mu)**2/(2*sig**2))**n)




output = sp.optimize.curve_fit(super_gauss, 
                                  x_data, 
                                  y_data, 
                                  (10000, 18000, 1000, 1))
print(output)

plt.plot(x_data, y_data)
plt.plot(x_data, [super_gauss(x, 4.05720902e+04, 1.88312796e+04, 5.27583496e+02, 1.03688877e+00) for x in x_data])
plt.plot(x_data, y_data - np.array([super_gauss(x, 4.05720902e+04, 1.88312796e+04, 5.27583496e+02, 1.03688877e+00) for x in x_data]))
plt.xlim((16000,20000))
plt.show()
