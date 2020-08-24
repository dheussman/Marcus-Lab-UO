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
import math as ma
import numpy as np
import scipy as sp
from scipy.optimize import minimize, minimize_scalar # for minimization.
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Stop Warning Messages
import warnings
#warnings.filterwarnings('ignore')   # optional filtering of warnings..
###############################################

############# HELPER FUNCTIONS ##############

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

##############################################
    
############# LASER FUNCTIONS ##############

def gauss(x, amp, mu, sig):
    return amp * np.exp(-(x-mu)**2/(2*sig**2))

def super_gauss(x, amp, mu, sig, n):
    return amp * np.exp(-((x-mu)**2/(2*sig**2))**n)

##############################################

class Laser:
    '''Class for storing the laser parameters for a particular experiment,
    generated directly from a laser spectra file'''
    
    def __init__(self, file_name, half_bandwidth = 1.6, shift = 0):
        '''creates an instance of the Laser class from a spectra file'''
        x_data, y_data = np.genfromtxt(file_name, skip_header = 2).T
        x_data = (1/x_data) * 10**7 # convert to wavenumbers
        
        self.xs = x_data
        self.ys = normalize(y_data)
        #plt.plot(self.ys)
#        plt.show()
#        plt.cla()
        # fit data:
        
        self.params = sp.optimize.curve_fit(gauss, 
                                       x_data, 
                                       y_data**2, 
                                       (10000, 18000, 1000))[0]
                                        
        self.params[1] += shift
        
        self.high = self.params[1] + self.params[2] * half_bandwidth
        self.low  = self.params[1] - self.params[2] * half_bandwidth
        
    def overlap_params(self, transition_energy, inhomogeneous_broadening):
        '''returns gaussian parameters for the overlap of the 
        laser spectra and the transition feature
        
        Assuming laser spectra is gaussian for simplicity'''
        
        # reference: http://www.tina-vision.net/docs/memos/2003-003.pdf
        
        # new mu:
        mu_l = self.params[1]
        sig_l= self.params[2]
        
        mu_t = transition_energy
        sig_t= inhomogeneous_broadening
        
        mean = (mu_l * sig_t**2 + mu_t * sig_l**2)/(sig_l**2 + sig_t**2)
        alpha = gauss(mean, 1, mu_l, sig_l)
        #alpha = np.interp(mean, self.xs + shift, self.ys)
        
        #print(mean, alpha)
        
#        #calculate the transition spectra
#        t_spec_y = []
#        for x in self.xs:
#            t_spec_y.append(100*ma.exp(-(x - transition_energy)**2/(2 * inhomogeneous_broadening**2))/(ma.sqrt(2*Pi) * inhomogeneous_broadening))
#        t_spec_y = np.array(t_spec_y)
#        
#        product = t_spec_y * self.ys
#
#        alpha = max(product)
#        mean = self.xs[np.argmax(product)]
#        
        
        return (mean, alpha)
    
    def plot(self):
        fig1 = plt.figure(figsize = (4,4))
        
        ax = fig1.add_subplot(111)
        
        ax.plot(self.xs, self.ys)

        ax.set_xlim((17200, 20500))
        
        plt.show()


#l = Laser("20190417/laseratsample_File3_17-39-56-003.txt")
#
#print(l.overlap_params(18000, 500))
#    
    
    