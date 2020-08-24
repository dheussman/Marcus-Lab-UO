#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:12:35 2019

@author: jkittell
"""
# Tools
import scipy as sp
from scipy.optimize import minimize, minimize_scalar, brute
import numpy as np
import math as ma
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg as la
import matplotlib.colors as cols


from scipy import signal as sg
Pi = np.pi
# Time/date stamp runs:
from datetime import datetime as dt

# Stop Warning Messages
import warnings
warnings.filterwarnings('ignore')

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
############# Class Definitions ##############

###################################################################
######################### Data Management #########################
            
'''We need to have a more robust scheme for data handling. This class is for
handling the raw data from the Abs, CD, and 2DFS measurements.
'''

class Data:
    '''Parent class for handling the raw data from an experiment'''
    def __init__(self, data, axes, tag = None, parts = None):
        '''generates a data object, from a data set dat, with the ith axis of 
        the data given by the ith element of 'axes'  '''
        #check#

        self.tag = tag
        #print(dat.shape)
        #if list(dat.shape) == [len(i) for i in axes]:
        self.dat = npa(data)
        self.axes = npa(axes)
        self.dim = self.get_dimm
        self.ft_dat = None
        if parts != None:
            self.parts = parts
        #else:
        #    raise Exception('''Mismatch between data and data axes.''')
    
    def __repr__(self):
        "internal representation of Data object"
        return "Data: " + self.timetag
    
    def get_dimm(self):
        '''gives dimm of array'''
        return len(self.dat.shape)
    
    def get_timetag(self):
        return self.timetag
    
    def shape(self):
        return self.dat.shape
    
    def __sub__(self, other): # redefining "-" as CHI_SQUARED
        '''preforms a chi squared calculation between two data sets, e.g.
        sum of (element - element)^2, must be same shape.
        Data -> num'''

        dif_array = self.dat - other.dat
        
        

        return np.real((dif_array**2)).sum()
    
    def normalize(self):
        '''normalizes the objects spectrum
        Data_1D -> None'''
        self.dat = self.dat/sum(abs(self.dat))

    def normalized(self):
        '''returns a copy of the 1D_Data object with a normalized spectrum
        Data_1D -> Data_1D'''
        return Data_1D(self.dat/sum((self.dat)**2), self.axes, self.tag)
    def scale(self, num):
        '''multiplies every value in a data set by 'num'
        num -> None'''
        self.dat = self.dat*scale
        
class Data_1D(Data):
    '''subclass for absorbtion/cd data'''
    
    def __init__(self, dat, axes, tag = None):
        '''creates 1D data object'''
        Data.__init__(self, dat, axes, tag)
        
    def __repr__(self):
        return "1D Data: " + str(self.tag)
    
    
    def chop(self, high):
        '''trims the data in a Data_1D object to be below 'high' 
        for the x axis
        num, num -> none'''
        new_axes = []
        new_data = []
        for i in ranlen(self.axes):
            if (self.axes[i] > high):
                break
            new_axes.append(self.axes[i])
        self.axes = npa(new_axes)
        self.dat = self.dat[:len(new_axes)]
        
        
    def plot(self, lims = (16700, 22000), other = None):
        '''generates a 1D plot of the Data
        [Tuple] -> None'''
        if other == None:
            plt.plot(self.axes, self.dat)
            #plt.xlim(lims[0], lims[1])
            plt.suptitle(self.tag)
            plt.show()
        else:
            plt.plot(self.axes, self.dat)
            plt.plot(other.axes, other.dat)
            #plt.xlim(lims[0], lims[1])
            plt.suptitle(self.tag + other.tag)
            plt.show()
    
    def subtract_background(self, data_1d):
        '''subtracts a background scan (stored as a Data_1D) from the data object.
        Data_1D -> None'''
        if self.shape() == data_1d.shape():
            self.dat = self.dat - data_1D.shape()
        else:
            raise Exception("Data objects are different shapes")
        
    def save(self, filename):
        '''saves a data object to a text file.'''


def import_1D_data(filename, title = "Data 1D"):
    '''creates a 1D data object from a file, assuming standard format
    str -> Data_1D'''
    data_array = np.genfromtxt(filename)
    x = Data_1D(data_array[:,1], data_array[:,0], title)
    x.normalize()
    return x


class Data_2D(Data):
    '''subclass for 2DFS Data, with multidimmensional arrays for dat and axes'''
    
    def __init__(self, rp_dat, nrp_dat, axes = (npa(range(0, 30, 1))*2.66e-15, npa(range(0, 30, 1))*2.66e-15), tag = None, bad = False):
        '''creates a 2D data object'''
        
        if bad:
            self.rp_dat  = np.pad(rp_dat[1:,:],  ((0, 0), (0, 1)), mode = "edge")
            self.nrp_dat = np.pad(nrp_dat[1:,:], ((0, 0), (0, 1)), mode = "edge")
        else:
            self.rp_dat = rp_dat
            self.nrp_dat = nrp_dat
        self.axes = axes
        # optional description param
        self.tag = tag                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
            
    def __repr__(self):
        return "2D Data: " + str(self.tag)
    
    def scale(self, factor):
        '''applies a scale factor 'factor to a data set'''
        self.rp_dat *= factor
        self.nrp_dat *= factor
        return self
    
    def normalize(self):
        '''normalizes the objects spectrum
        Data_2D -> None'''
        
        self.rp_dat  = self.rp_dat/np.amax((np.real(self.rp_dat)))
        self.nrp_dat = self.nrp_dat/np.amax((np.real(self.nrp_dat)))
        try:
            len(self.ft_rp)
            self.ft_rp = self.ft_rp/np.amax((np.real(self.ft_rp)))
            self.ft_nrp = self.ft_nrp/np.amax((np.real(self.ft_nrp)))
        except TypeError:
            pass
    
    def __mul__(self, factor):
        '''defines multiplication as a scaling factor'''
        self.rp_dat *= factor
        self.nrp_dat *= factor
        
        return self
    
    def __add__(self, other):
        '''defines addition'''
        self.rp_dat
        new_rp = self.rp_dat + other.rp_dat
        new_nrp = self.nrp_dat + other.nrp_dat
    
        return Data_2D(new_rp, new_nrp)
    
    def __radd__(self, other):
        '''defines addition'''
        self.rp_dat
        new_rp = self.rp_dat + other.rp_dat
        new_nrp = self.nrp_dat + other.nrp_dat
        
        return Data_2D(new_rp, new_nrp)
    
    def chi_squared(self, other, FT = False): 
        '''preforms a chi squared calculation between two data sets, e.g.
        sum of (element - element)^2, must be same shape.
        Data -> num'''

        if FT == True:
            rp_dif  = self.ft_rp  - other.ft_rp
            nrp_dif = self.ft_nrp - other.ft_nrp
        else:
            rp_dif  = self.rp_dat  - other.rp_dat
            nrp_dif = self.nrp_dat - other.nrp_dat
            
        return ((np.abs(rp_dif)**2).sum() + (np.abs(nrp_dif)**2).sum())       
        
            
    
    def FT(self):
        '''preforms a 2D fourier transform of the data object
        
        num, num, [num] -> None'''
        
        #need time step
        time_step = self.axes[0][0] - self.axes[0][1]
        
        # first goal is 'window' the thing, meaning we drive down our noise
        # with a super gaussian, and double the number of sample points
        
        nubar2nu = 29979000000.0
        window_size = 30
        
        #plt.matshow(np.real(self.dat))
        rp_window  = np.zeros((window_size,window_size), dtype = np.complex_)
        nrp_window = np.zeros((window_size,window_size), dtype = np.complex_)
        for i in range(window_size):
            for j in range(window_size):
                rp_window[i,j]  = np.exp(-(np.sqrt((i**2 + j**2))/20)**10)
                nrp_window[i,j] = np.exp(-(np.sqrt((i**2 + j**2))/20)**10)
        #print(rp_window[0])
        
        
        rp_temp  = self.rp_dat  * rp_window
        rp_temp  = np.pad(rp_temp,  ((0,200),(0,200)), "linear_ramp", end_values = ((0,0),(0,0)))
        nrp_temp = self.nrp_dat * nrp_window
        nrp_temp = np.pad(nrp_temp, ((0,200),(0,200)), "linear_ramp", end_values = ((0,0),(0,0)))
        
        self.ft_rp =  np.fft.fftshift(np.fft.fft2(rp_temp))
        self.ft_nrp = np.fft.fftshift(np.fft.fft2(nrp_temp))

        
        ax1 = np.fft.fftshift(
                np.fft.fftfreq(window_size + 200, time_step)
                )/(nubar2nu)
        ax2 = np.fft.fftshift(
                np.fft.fftfreq(window_size + 200, time_step)
                )/(nubar2nu)
        
        self.ft_axes = [np.flip(ax1), ax2]     
#        self.ft_axes_nrp = [np.flip(ax1), np.flip(ax2)]
#        self.ft_axes_rp = [np.flip(ax1), ax2]
        self.normalize()
        
    def decayed(self, gam, sig):
        nubar2nu = 100*2.9979e8
        ###### Helper functions #########
        def rp_decay(t21, t43, gam, sig):
            
            gam = gam * nubar2nu * 2 * Pi
            sig = sig * nubar2nu * 2 * Pi
            ###EDIT###
            return np.exp(- gam * (t21 + t43)\
                          - (1/2)*(sig * (t21 - t43))**2)
        def nrp_decay(t21, t43, gam, sig):
            
            gam = gam * nubar2nu * 2 * Pi
            sig = sig * nubar2nu * 2 * Pi
            
            return np.exp(- gam * (t21 + t43)\
                          - (1/2)*(sig * (t21 + t43))**2)

        
        ###### 2D simulation, over some axes   ########
        ax1 = npa(range(0, 30, 1))*2.66e-15 # 30 2.66 fmptsec steps
        ax2 = npa(range(0, 30, 1))*2.66e-15
        
        
        decay_rp = np.zeros((30,30), dtype = float)
        decay_nrp = np.zeros((30,30), dtype = float)
        
        for i in ranlen(ax1):
            for j in ranlen(ax2):
                decay_rp[i,j] = rp_decay(ax1[i], ax2[j], gam, sig)
                decay_nrp[i,j] = nrp_decay(ax1[i], ax2[j], gam, sig)
        
        temp = Data_2D(self.rp_dat * decay_rp, self.nrp_dat * decay_nrp, self.axes)
        temp.FT()
        return temp
    
    
    def simple_plot(self, filename = None, title = None):
        '''simple, faster plotting, for internal use (you and i, the programmers)'''
        plt.figure(figsize=(15,7))
        
        if title != None:
            plt.title(title)
        
        ax0 = plt.subplot(241)
        ax0.matshow(np.real(self.rp_dat))
        ax0.text(20,25, "RP - Real")
        
        ax1 = plt.subplot(242)
        ax1.matshow(np.real(self.ft_rp))
        ax1.text(120,180, "FT RP - Real")
        
        ax2 = plt.subplot(243)
        ax2.matshow(np.real(self.nrp_dat))
        ax2.text(20,25, "NRP - Real")
    
        ax3 = plt.subplot(244)
        ax3.matshow(np.real(self.ft_nrp))
        ax3.text(120,180, "FT NRP - Real")
        
        ax4 = plt.subplot(245)
        ax4.matshow(np.imag(self.rp_dat))
        ax4.text(20,25, "RP - Imag")
        
        ax5 = plt.subplot(246)
        ax5.matshow(np.imag(self.ft_rp))
        ax5.text(120,180, "FT RP - Imag")

        ax6 = plt.subplot(247)
        ax6.matshow(np.imag(self.nrp_dat))
        ax6.text(20,25, "NRP - Imag")
    
        ax7 = plt.subplot(248)
        ax7.matshow(np.imag(self.ft_nrp))
        ax7.text(120,180, "FT NRP - Imag")


        if filename != None:
            plt.savefig(filename, dpi = 150)
        
        
        plt.show()
    def compare_plot(self, other, style = "interferogram", filename = None):
        '''create a side by side plot of two data sets'''
        
         
            
            
    def super_plot(self, title = None, filename = None):
        '''Plots both interferograms and FTs'''
        # let's begin by defining our contours
         

        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
            new_cmap = cols.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                cmap(np.linspace(minval, maxval, n)))
            return new_cmap
        
        cmap2 = plt.get_cmap('rainbow')
        colormap = truncate_colormap(cmap2, 0.5 - np.abs(np.amin(self.ft_nrp))/np.amax(np.real(self.ft_nrp)), 1) 
        
        print(np.amin(np.real(self.ft_rp)))
        print(np.amax(np.real(self.ft_nrp)))
        
        
        
        levels = np.linspace(-1, np.amax(np.abs(self.ft_rp)), 70)
        levelscon = np.linspace(np.amin(np.imag(self.ft_nrp)), np.amax(self.ft_rp), 20)
        
#        ax1 = (np.fft.fftshift(np.fft.fftfreq(len(data), stepsizex*(10**-15)))/nubar2nu  + 10**7/MonochromatorLHS)/1000
#        ax2 = (np.fft.fftshift(np.fft.fftfreq(len(data.T), stepsizex*(10**-15)))/nubar2nu + 10**7/MonochromatorRHS)/1000    
        xlims, ylims = ((-200, 3000), 
                        (-200, 3000))
        
        
        
        
        plt.figure(figsize=(10,10))
        
        if title != None:
            plt.title(title)
        
        x, y = self.ft_axes
        
        # define a figure w/ subplots
        fig1 = plt.figure(figsize=(12,8))
        
        #Interfereograms
        rp_int = fig1.add_subplot(231)
        nrp_int = fig1.add_subplot(234)
        
        #Transforms
        rp_ftr = fig1.add_subplot(232)
        rp_fti = fig1.add_subplot(233)
        
        nrp_ftr = fig1.add_subplot(235)
        nrp_fti = fig1.add_subplot(236)
        
        # labels
        rp_int.set_ylabel("RP Interferogram")
        nrp_int.set_ylabel("NRP Interferogram")
        rp_ftr.set_xlabel("Fourier Transform (Real)")
        rp_fti.set_xlabel("Fourier Transform (Imaginary)")
        nrp_ftr.set_xlabel("Fourier Transform (Real)")
        nrp_fti.set_xlabel("Fourier Transform (Imaginary)")
        
        # Contour Plots
        rp_int.matshow(np.real(self.rp_dat), cmap = "winter")
        nrp_int.matshow(np.real(self.nrp_dat), cmap = "winter")
        
        #rp_ftr.gca().set_aspect('equal', adjustable='box')
        rp_ftr.contourf(self.ft_axes[0], self.ft_axes[1], (np.real(self.ft_rp)), levels = levels, cmap = colormap)
#        rp_ftr.contour(self.ft_axes[0], self.ft_axes[1], (np.real(self.ft_rp)), n_cont)
        rp_ftr.plot(rp_ftr.get_xlim(), rp_ftr.get_ylim(), ls="--", c=".2")
        rp_ftr.plot((rp_ftr.get_xlim()[0], rp_ftr.get_ylim()[-1]), (rp_ftr.get_xlim()[-1], rp_ftr.get_ylim()[0]), ls="--", c=".2")
        rp_ftr.set_xlim(xlims)
        rp_ftr.set_ylim(ylims)
        
        CS = rp_ftr.contour(self.ft_axes[0], self.ft_axes[1], (np.real(self.ft_rp)), colors='black',levels=levelscon, linewidths=1.5, alpha=1)
        for line in CS.collections:
            if line.get_linestyle() != [(None, None)]: 
                line.set_color('white')
                line.set_linewidth(1.5)
                line.set_dashes('solid')
        
        #rp_fti.gca().set_aspect('equal', adjustable='box')
        rp_fti.contourf(self.ft_axes[0], self.ft_axes[1], (np.imag(self.ft_rp)), levels = levels, cmap = colormap)
#        rp_fti.contour(self.ft_axes[0], self.ft_axes[1], (np.imag(self.ft_rp)), n_cont)
        rp_fti.plot(rp_fti.get_xlim(), rp_fti.get_ylim(), ls="--", c=".2")
        rp_fti.plot((rp_fti.get_xlim()[0], rp_fti.get_ylim()[-1]), (rp_fti.get_xlim()[-1], rp_fti.get_ylim()[0]), ls="--", c=".2")
        rp_fti.set_xlim(xlims)
        rp_fti.set_ylim(ylims)
        
        CS = rp_fti.contour(self.ft_axes[0], self.ft_axes[1], (np.imag(self.ft_rp)), colors='black',levels=levelscon, linewidths=1.5, alpha=1)
        for line in CS.collections:
            if line.get_linestyle() != [(None, None)]: 
                line.set_color('white')
                line.set_linewidth(1.5)
                line.set_dashes('solid')
        
        #nrp_ftr.gca().set_aspect('equal', adjustable='box')
        nrp_ftr.contourf(self.ft_axes[0], self.ft_axes[1], np.flipud(np.real(self.ft_nrp)), levels = levels, cmap = colormap)
#        nrp_ftr.contour(self.ft_axes[0], self.ft_axes[1], np.flipud((np.real(self.ft_nrp))), n_cont)
        nrp_ftr.plot(nrp_ftr.get_xlim(), nrp_ftr.get_ylim(), ls="--", c=".2")
        nrp_ftr.plot((nrp_ftr.get_xlim()[0], nrp_ftr.get_ylim()[-1]), (nrp_ftr.get_xlim()[-1], nrp_ftr.get_ylim()[0]), ls="--", c=".2")
        nrp_ftr.set_xlim(xlims)
        nrp_ftr.set_ylim(ylims)
        
        CS = nrp_ftr.contour(self.ft_axes[0], self.ft_axes[1], np.flipud(np.real(self.ft_nrp)), colors='black',levels=levelscon, linewidths=1.5, alpha=1)
        for line in CS.collections:
            if line.get_linestyle() != [(None, None)]: 
                line.set_color('white')
                line.set_linewidth(1.5)
                line.set_dashes('solid')
        
        #nrp_fti.gca().set_aspect('equal', adjustable='box')
        nrp_fti.contourf(self.ft_axes[0], self.ft_axes[1], np.flipud(np.imag(self.ft_nrp)), levels = levels, cmap = colormap)
#        nrp_fti.contour(self.ft_axes[0], self.ft_axes[1], np.flipud(np.imag(self.ft_nrp)), n_cont)
        nrp_fti.plot(nrp_fti.get_xlim(), nrp_fti.get_ylim(), ls="--", c=".2")
        nrp_fti.plot((nrp_fti.get_xlim()[0], nrp_fti.get_ylim()[-1]), (nrp_fti.get_xlim()[-1], nrp_fti.get_ylim()[0]), ls="--", c=".2")
        nrp_fti.set_xlim(xlims)
        nrp_fti.set_ylim(ylims)
        
        CS = nrp_fti.contour(self.ft_axes[0], self.ft_axes[1], np.flipud(np.imag(self.ft_nrp)), colors='black',levels=levelscon, linewidths=1.5, alpha=1)
        for line in CS.collections:
            if line.get_linestyle() != [(None, None)]: 
                line.set_color('white')
                line.set_linewidth(1.5)
                line.set_dashes('solid')
        
        if filename != None:
            plt.savefig(filename, dpi = 150)

        plt.show()
        
    def paper_plot(self, colormap = "hsv", filename = None, title = None, grid = False, text = False):
        '''generates a publication quality plot, optional SVG output'''
        
        # let's begin by defining our contours
        cmap = matplotlib.cm.get_cmap(colormap)
        n = 500 # even num
        contours = npa(range(-n, n))/(.5*n)
        #print(contours[0:10])

        cont_col = [cmap(i/(2*n)) for i in range(2*n)]
        
        n_cont = 15
        xlims, ylims = ((self.ft_axes[0][0]+3000, self.ft_axes[0][-1]-3000), 
                        (self.ft_axes[1][0]+3000, self.ft_axes[1][-1]-3000))
        
        color_map = colormap
        
        plt.figure(figsize=(10,10))
        
        if title != None:
            plt.title(title)
        
        x, y = self.ft_axes
        
        x_new = []
        y_new = []
        
    
        for xx in x:
            for yy in y:
                x_new.append(xx)
                y_new.append(yy)
        
        plt.subplot(221)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("Real NRP")
        plt.contourf(self.ft_axes[0], self.ft_axes[1], (np.real(self.ft_nrp)), levels = contours, colors = cont_col)
        plt.contour(self.ft_axes[0], self.ft_axes[1], (np.real(self.ft_nrp)), n_cont)
        plt.plot(plt.xlim(), plt.ylim(), ls="--", c=".2")
        plt.plot((plt.xlim()[0], plt.ylim()[-1]), (plt.xlim()[-1], plt.ylim()[0]), ls="--", c=".2")
        plt.xlim(xlims)
        plt.ylim(ylims)
        if grid:
            plt.scatter(x_new, y_new, s = 1)
            plt.grid(False)
        plt.subplot(222)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("Real RP")
        plt.contourf(self.ft_axes[0], self.ft_axes[1], np.real(self.ft_rp), levels = contours, colors = cont_col)
        plt.contour(self.ft_axes[0], self.ft_axes[1], np.real(self.ft_rp), n_cont)
        plt.plot(plt.xlim(), plt.ylim(), ls="--", c=".2")
        plt.plot((plt.xlim()[0], plt.ylim()[-1]), (plt.xlim()[-1], plt.ylim()[0]), ls="--", c=".2")
        plt.xlim(xlims)
        plt.ylim(ylims)
        if grid:
            plt.scatter(x_new, y_new, s = 1)
            plt.grid(False)
#        plt.subplot(233)
#        plt.gca().set_aspect('equal', adjustable='box')
#        plt.xlabel("Total Real")
#        plt.contourf(self.ft_axes[0], self.ft_axes[1], (np.fliplr(np.real(self.ft_rp)).T + np.flipud(np.fliplr(np.real(self.ft_nrp)))),  levels = contours, colors = cont_col)
#        plt.contour(self.ft_axes[0], self.ft_axes[1], (np.fliplr(np.real(self.ft_rp)).T + np.flipud(np.fliplr(np.real(self.ft_nrp)))), n_cont)
#        plt.xlim(xlims)
#        plt.ylim(ylims)
        
        #plt.clear()
        plt.subplot(223)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("Imag NRP")
        plt.contourf(self.ft_axes[0], self.ft_axes[1], (np.imag(self.ft_nrp)), levels = contours, colors = cont_col)
        plt.contour(self.ft_axes[0], self.ft_axes[1], (np.imag(self.ft_nrp)), n_cont)
        plt.plot(plt.xlim(), plt.ylim(), ls="--", c=".2")       
        plt.plot((plt.xlim()[0], plt.ylim()[-1]), (plt.xlim()[-1], plt.ylim()[0]), ls="--", c=".2")
        plt.xlim(xlims)
        plt.ylim(ylims)
        if grid:
            plt.scatter(x_new, y_new, s = 1)
            plt.grid(False)
        plt.subplot(224)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("Imag RP")
        plt.contourf(self.ft_axes[0], self.ft_axes[1], (np.imag(self.ft_rp)),  levels = contours, colors = cont_col)
        plt.contour(self.ft_axes[0], self.ft_axes[1], (np.imag(self.ft_rp)), n_cont)
        plt.plot(plt.xlim(), plt.ylim(), ls="--", c=".2")
        plt.plot((plt.xlim()[0], plt.ylim()[-1]), (plt.xlim()[-1], plt.ylim()[0]), ls="--", c=".2")
        plt.xlim(xlims)
        plt.ylim(ylims)
        if grid:
            plt.scatter(x_new, y_new, s = 1)
            plt.grid(False)
#        plt.subplot(236)
#        plt.gca().set_aspect('equal', adjustable='box')
#        plt.xlabel("Total Imag")
#        plt.contourf(self.ft_axes[0], self.ft_axes[1], -(np.fliplr(np.imag(self.ft_rp)).T + np.flipud(np.fliplr(np.imag(self.ft_nrp)))),  levels = contours, colors = cont_col)
#        plt.contour(self.ft_axes[0], self.ft_axes[1], -(np.fliplr(np.imag(self.ft_rp)).T + np.flipud(np.fliplr(np.imag(self.ft_nrp)))), n_cont)
#        plt.xlim(xlims)
#        plt.ylim(ylims)
        
        if text != False:
            plt.text(15000, 15000, str(text), fontsize = 25)
        
        if filename != None:
            plt.savefig(filename, dpi = 150)
        

            

        plt.show()
        


class Dataset():
    '''class for storing data runs'''
    def __init__(self, data_dic, parameter, note):
        self.data_dic = data_dic
        self.keys = self.data_dic.keys()
        self.note = note
    
def import_2d_dataset(directory,
                      parameter,
                      vals,
                      suffixes,
                      note):
    return None

def import_2D_data(directory, suffix, monochrom1, monochrom2 = None, time_step = 2.66e-15, make_plots = False, bad1 = False):
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
    data = Data_2D(np.conjugate(rp_phase*rp), np.conjugate(nrp_phase*nrp), bad = bad1)
    #print((rp_phase * rp.T)[0,0], (nrp_phase * np.conjugate(nrp.T))[0,0])
    data.FT()
    data.normalize()
    
    
    if make_plots:
        data.paper_plot()
    return data

    
    
    
def pdf_plot(data1, data2, colormap = "rainbow", n = None):
    '''plots a combined layout for two data objects
    Data_2D, Data_2D -> '''
    
    # let's begin by defining our contours
    cmap = matplotlib.cm.get_cmap(colormap)
    n = 500 # even num
    contours = npa(range(-n, n))/(.5*n)
    #print(contours[0:10])
    
    cont_col = [cmap(i/(2*n)) for i in range(2*n)]
    
    n_cont = 15
    xlims, ylims = ((16500, 22000), (16500, 22000))
    
    color_map = colormap
    
    plt.figure(figsize=(12,12))
    plt.subplot(131)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Real NRP")
    plt.contourf(data1.ft_axes[0], data1.ft_axes[1], np.real(data1.ft_dat), levels = contours, colors = cont_col)
    plt.contour(data1.ft_axes[0], data1.ft_axes[1], np.real(data1.ft_dat), n_cont)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.subplot(132)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Real RP")
    plt.contourf(data2.ft_axes[0], data2.ft_axes[1], np.real(data2.ft_dat), levels = contours, colors = cont_col)
    plt.contour(data2.ft_axes[0], data2.ft_axes[1], np.real(data2.ft_dat), n_cont)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.subplot(133)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Total Real")
    plt.contourf(data2.ft_axes[0], data2.ft_axes[1], (np.real(data1.ft_dat) + np.real(data2.ft_dat)),  levels = contours, colors = cont_col)
    plt.contour(data2.ft_axes[0], data2.ft_axes[1], (np.real(data1.ft_dat) + np.real(data2.ft_dat)), n_cont)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.show()
    
    #plt.clear()

    plt.figure(figsize=(12,12))
    plt.subplot(131)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Imag NRP")
    plt.contourf(data1.ft_axes[0], data1.ft_axes[1], -np.imag(data1.ft_dat), levels = contours, colors = cont_col)
    plt.contour(data1.ft_axes[0], data1.ft_axes[1], -np.imag(data1.ft_dat), n_cont)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.subplot(132)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Imag RP")
    plt.contourf(data2.ft_axes[0], data2.ft_axes[1], -np.imag(data2.ft_dat),  levels = contours, colors = cont_col)
    plt.contour(data2.ft_axes[0], data2.ft_axes[1], -np.imag(data2.ft_dat), n_cont)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.subplot(133)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Total Imag")
    plt.contourf(data2.ft_axes[0], data2.ft_axes[1], -(np.imag(data1.ft_dat) + np.imag(data2.ft_dat)),  levels = contours, colors = cont_col)
    plt.contour(data2.ft_axes[0], data2.ft_axes[1], -(np.imag(data1.ft_dat) + np.imag(data2.ft_dat)), n_cont)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.show()
    
    
##test###
#nrp, rp = import_2D_data("2DFS_031119", "a")
#print(len(rp.axes[0]), len(rp.axes[1]))
#print(rp.shape())
#rp.contour_plot(100)
#rp.FT(2.66e-15, 17857)
#rp.contour_plot(100)
#
#nrp.contour_plot(100)
#nrp.FT(2.66e-15, 17857)
#nrp.contour_plot(100)
