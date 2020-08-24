#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:54:29 2019

              <><><>    J Tool Kit     <><><>

This module is an OOP implementation of the modeling document written by
Loni Kringle et al for generating Absorption, CD, and 2DFS spectra
for molecular systems.

This document contains definitions for the actual spectra generation, while 
the details of the classes necessary for the calculation are contained in 
imported files.

Classes, methods, and attributes from imports are outlined briefly below.
@author: jkittell
"""

from model_code import *
        # Defines the classes:
        #   
        #   <> Operator(mat, space, opname = "temp")
        #       - Methods:
        #           .eigs()     -   calculates and stores eigenvals and vecs for mat
        #           .dagger()   -   returns a conjugate transpose of self
        #           .text_save()-  
        #   <> Conformation
        #   <> HilbertSpace(subs, dimms, conf, ham=None)
        #       - Methods: 
        #           .ident(s)   -   identity on subspace s in subs
        #           .kron(ops)  -   takes a list of Operators
        #
from data_management import *
        # Defines the classes:
        #   
        #   <> Data(data, axes, opname = "temp")
        #       - Methods:
        #           .eigs()     -   calculates and stores eigenvals and vecs for mat
        #           .dagger()   -   returns a conjugate transpose of self
        #           .text_save()-   saves a data object to a txt.file
        #           .get_dimm()    -   returns the num of dimms of the data stored
        #   <> Data_1D(Data) - subclass of Data
        #       - Methods: 
        #           .shape()       -   returns the shape (,) of the stored data
        #           .x_trim(lo,hi) -   trims the data over the x axis, between lo, hi
from laser_objects import *
        # deals with laser spectra.

from scipy.optimize import brute, minimize, minimize_scalar # for minimization.
from scipy.signal import savgol_filter
import sqlitedict as DB


#     <><><>  Spectra Generators  <><><>

class Gen_1D_Spec:
    '''class capabale of generating different 1D spectra (Abs, CD, etc)
    given a hilbert space and broadening parameters
    attr: hilspac, hom_broad, inhom_broad'''
    
    def __init__(self, hilspac, broad_scheme):
        '''creates an instance of an SimSpectra, takes a hilbert space on
        which to make a "measurement", a scheme (one of 'gauss', 'lorentz', 'pv').
        
        HilbertSpace, int, (num or array_like) -> Spec_Gen.'''
  
        self.hs = hilspac
        self.scheme = broad_scheme


        # internally generated
        
        self.abs_stick = None
        self.cd_stick = None

    def __repr__(self):
        '''generatres an internal representation of the SimSpectra object.
        
        None -> str'''
        return "SimSpectra(" + str(self.hs) +")"
    
    def set_broad_scheme_to(new_scheme):
        '''sets the broadening scheme to the string 'new_scheme'
        str -> None'''
        self.scheme = new_scheme
    
    ####################################################
    ###########   Broadening Schemes   #################
    
    # These methods all take three args - x value, center value, (broad params,)
    
    # 0
    def gauss(self, x, mu, sigma):
        '''returns the value of a gaussian distribution centered on mu with
        braodening sig.
        num, num, num -> num'''
        sig = sigma
#        if len(sigma) == 1:
#            sig = sigma[0]
#        else:
#            raise Exception("bad broadening params")
        return ma.exp(-(x - mu)**2/(2 * sig**2))/(ma.sqrt(2*Pi) * sig)
    
    # 1
    def lorentz(self, x, mu, gamma):
        '''returns the value of a lorentzian distribution centered on mu with
        braodening gam.
        num, num, num -> num'''
        #if len(gamma) == 1:
        #    gam = gamma[0]
        #else:
        #    raise Exception("bad broadening params")
        # broken implementation...... needs to be able to handle multiple params for the pv dist
        
        return 1/(Pi * gamma * (((x - mu)/gamma)**2 + 1))

    # 2
    def pv_dist(self, x, epsilon, gam_sig):
        '''calculates and returns a pseudovoigt distr centered at epsilon with
        broadening gamma and sigma for the lorentz and gauss, respectively
        num, num, (num, num)-> num'''
        if len(gam_sig) == 2:
            gamma, sigma = gam_sig
        else:
            raise Exception("bad broadening params")
        g = ((gamma**5) + (sigma**5) + (2.69296)*(sigma**4)*(gamma) + \
        2.42843*(sigma**3)*(gamma**2) + 4.47163*(sigma**2)*(gamma**3) + \
                0.07842*(sigma)*(gamma**4))**(1/5)
        eta = gamma/g
        eta = eta * (1.36603 - 0.47719*eta + 0.11116*eta**2)
        return eta * self.lorentz(x, epsilon, g) + (1 - eta) * self.gauss(0, epsilon, g)
    
    # can add more!
    
    # wrapper for different distributions
    def broaden(self, x, center, broad_params):
        '''takes a point, a center, a scheme, and broadening params and gives
        the distribution value at that point
        num, num, int, int_or_tuple -> num'''
        if self.scheme == "gauss":
            return self.gauss(x, center, broad_params)
        elif self.scheme == "lorentz":
            return self.lorentz(x, center, broad_params)
        elif self.scheme == "pv":
            return self.pv_dist(x, center, broad_params)
        else:
            raise Exception("Not a valid distribution")
        
            
        ####################################################
        ########## STICK SPECTRA ##############
    
    def abs_stick_spec(self):
        '''returns a absorbance stick spectra for the hilbert space
        None -> Data_1D'''
        
        # We need to do a few things to calculate the transition intensities
        # First we need the dipole operator, which is generated by the
        # hilbert space.
        
        dip_x, dip_y, dip_z = self.hs.dipole_op()
        
        # we'll add each dimmensions contribution in quadrature.
        # the intensity of a particular absorption feature (eigenvalue) is
        # given by
        #
        # |<g|mutot|e_i>|^2
        # 
        # let's implement this in our program
        
        eigs = self.hs.ham.eigs
        vecs = self.hs.ham.vecs
        
        g_state = vecs[0]    # define the ground state
        #print(g_state.vec)
        
        stick_spec =  []
        #dip_x.__repr__()

        for i in ranlen(eigs):
            vec = vecs[i]

            Ix = g_state * (dip_x * vec)
            Iy = g_state * (dip_y * vec)
            Iz = g_state * (dip_z * vec)  #should give us 3 numbers
            
            Itot = (Ix**2 + Iy**2 + Iz**2) * (2/3)

            stick_spec.append([eigs[i], Itot])
        
        # playing arund with saving these... 
        #self.abs_stick = npa(stick_spec)
        stick_spec = npa(stick_spec)
        #plt.stem(stick_spec[:,0], stick_spec[:,1]*10,linefmt='b-', markerfmt='bo',basefmt='  ') 
        #plt.show()
        #plt.cla()
        return (stick_spec)
    
    def cd_stick_spec(self):
        '''returns a CD stick spectra for the hilbert space
        None -> Data_1D'''
        
        # "The Molecular Basis of Optical Activity - Elliot Charney" pg.s 48, 70, 104
        # Rosenfeld Equation: first term
        # This is the book this theory comes from, Andy should have a copy 
        # of the text if I don't, or if you're reading this after I've left.
        
        # get the dipole and magnetic moment operators:
        dip_x, dip_y, dip_z = self.hs.dipole_op()
        mag_x, mag_y, mag_z = self.hs.mag_op()
        
        eigs = self.hs.ham.eigs
        vecs = self.hs.ham.vecs
        
        g_state = vecs[0]    # define the ground state
                
        stick_spec = []
        
        for i in ranlen(eigs):
            vec = vecs[i]
            
            Ix = - (g_state * (dip_x * vec)) * (g_state * (mag_x * vec))
            Iy = - (g_state * (dip_y * vec)) * (g_state * (mag_y * vec))    
            Iz = - (g_state * (dip_z * vec)) * (g_state * (mag_x * vec))
            
            Itot = (Ix + Iy + Iz)
            
            stick_spec.append([eigs[i], Itot])
        
        stick_spec = npa(stick_spec)
        #plt.stem(stick_spec[:,0], stick_spec[:,1]*10,linefmt='b-', markerfmt='bo',basefmt='  ') 
        #plt.show()
        #plt.cla()
        return (stick_spec)
        
            
        
    ####################################################
    ########## FULL SPECTRA ##############
    
    def abs_spec(self, energy_axis, broad_params, filename = None):
        '''generates an absorbance spectra along a given energy axis. 
        If filename given, save spectra.
        int, array-like [, str] -> Data_1D'''
        
        # first generate a stick spectra
        sticks = self.abs_stick_spec()
        
        full = []
        for e in energy_axis:
            val = 0
            for stick in sticks:
                val += stick[1] * self.broaden(e, stick[0], broad_params)
            full.append([e, val])
        
        full = npa(full)
        
        temp = Data_1D(full[:,1], full[:,0], "Simuated Absorbance")
        temp.normalize()
        
        return temp
        
    def cd_spec(self, energy_axis, broad_params, filename = None):
        '''generates a CD spectra along a given energy axis with broad
        one of (0,1,2)~(gauss, lorentz, pv_dist). If filename given, save spectra.
        int, array-like [, str] -> None'''
        
        # first generate a stick spectra
        sticks = self.cd_stick_spec()
        
        full = []
        for e in energy_axis:
            val = 0
            for stick in sticks:
                val += stick[1] * self.broaden(e, stick[0], broad_params)
            full.append([e, val])
        
        full = npa(full)
        
        temp = Data_1D(full[:,1], full[:,0], "Simulated CD")
        temp.normalize()
        
        return temp

    def single_optimization_dif_ev(self, temp, model, abs_filename, cd_filename, start, limits):
        '''minimize chi squared between a simulated Abs and CD spectra and data
        by exploring the conformational space of the coupled dimer system.
        
        str, str, array_like -> OptimizeResult (see sp.optimize.minimize)'''
        # import Abs/CD data as normalized 1D data objects
        abs_temp = import_1D_data(abs_filename, "Absorbance Data")
        #abs_temp.plot()
        cd_temp = import_1D_data(cd_filename, "CD Data")
        cd_temp.chop(23000)
        cd_temp.normalize()
        #cd_temp.plot()
        
        # need to calculate signal to noise ratio of the two data sets 
        # for weighting their contributions to the total chi-squared
        
        abs_snr = (1/np.std(abs_temp.dat - savgol_filter(abs_temp.dat, 31, 3)))
        cd_snr  = (1/np.std( cd_temp.dat - savgol_filter( cd_temp.dat, 31, 3)))
        
        print("Abs SNR: " + str(abs_snr))
        print("CD SNR:  " + str( cd_snr))
        
        # We use the energy axis from the data to generate the simulated spectra
        abs_axis = abs_temp.axes
        cd_axis  =  cd_temp.axes
        
        # Now we need to define the function to be minimized
        
        def chi_squared(R12_phi_theta_broad, make_plots = True):
            '''calculates a chi_squared value between two data sets'''
            print(R12_phi_theta_broad)
            # first we unpack the input parameter array
            r, phi, theta, broad = R12_phi_theta_broad

            # change conformation:
            self.hs.conf.set_r_phi_theta(r, phi, theta)

            # build a new Cy-3 Dimer hamiltonian
            #                            temp,    model (point vs exten)
            self.hs.build_cy3dimer_ham(  temp,     model  )
            #print(broad)
            sim_abs = self.abs_spec(abs_axis, broad)
            sim_cd  = self.cd_spec(  cd_axis, broad)
            
            sim_abs.normalize()
            
            sim_cd.normalize()
            
            if make_plots:
                sim_abs.plot(other = abs_temp)
                sim_cd.plot(other = cd_temp)
            
            chi_squared = abs_snr*(abs_temp - sim_abs) + cd_snr*(cd_temp - sim_cd)
            #print(type(chi_squared))
            print("Chi^2: ", chi_squared)
            return chi_squared
        
        current_chi = chi_squared(start)
        print("Initial Chi^2: " + str(current_chi))
        
    
        output = sp.optimize.least_squares(chi_squared, 
                                           start, 
                                           jac='3-point', 
                                           bounds=([limits[i][0] for i in range(len(limits))], 
                                                    [limits[i][1] for i in range(len(limits))]), 
                                           method='trf', 
                                           ftol=1e-08, 
                                           xtol=1e-08, 
                                           gtol=1e-08, 
                                           x_scale=(1e-10, .1, .1, 100), 
                                           loss='linear', 
                                           f_scale=1.0, 
                                           diff_step=(.01e-10, .01,.01, 30), 
                                           tr_solver=None, 
                                           tr_options={}, 
                                           jac_sparsity=None, 
                                           max_nfev=None, 
                                           verbose=2, 
                                           args=(False,))
                                           
        return output

    def single_optimization(self, temp, model, abs_filename, cd_filename, start, limits):
        '''minimize chi squared between a simulated Abs and CD spectra and data
        by exploring the conformational space of the coupled dimer system.
        
        str, str, array_like -> OptimizeResult (see sp.optimize.minimize)'''
        # import Abs/CD data as normalized 1D data objects
        abs_temp = import_1D_data(abs_filename, "Absorbance Data")
        #abs_temp.plot()
        cd_temp = import_1D_data(cd_filename, "CD Data")
        cd_temp.chop(23000)
        cd_temp.normalize()
        #cd_temp.plot()
        
        # need to calculate signal to noise ratio of the two data sets 
        # for weighting their contributions to the total chi-squared
        
        abs_snr = (1/np.std(abs_temp.dat - savgol_filter(abs_temp.dat, 31, 3)))
        cd_snr  = (1/np.std( cd_temp.dat - savgol_filter( cd_temp.dat, 31, 3)))
        
        print("Abs SNR: " + str(abs_snr))
        print("CD SNR:  " + str( cd_snr))
        
        # We use the energy axis from the data to generate the simulated spectra
        abs_axis = abs_temp.axes
        cd_axis  =  cd_temp.axes
        
        # Now we need to define the function to be minimized
        
        def chi_squared(R12_phi_theta_broad, make_plots = True):
            '''calculates a chi_squared value between two data sets'''
            print(R12_phi_theta_broad)
            # first we unpack the input parameter array
            r, phi, theta, broad = R12_phi_theta_broad

            # change conformation:
            self.hs.conf.set_r_phi_theta(r, phi, theta)

            # build a new Cy-3 Dimer hamiltonian
            #                            temp,    model (point vs exten)
            self.hs.build_cy3dimer_ham(  temp,     model  )
            #print(broad)
            sim_abs = self.abs_spec(abs_axis, broad)
            sim_cd  = self.cd_spec(  cd_axis, broad)
            
            sim_abs.normalize()
            
            sim_cd.normalize()
            
            if make_plots:
                sim_abs.plot(other = abs_temp)
                sim_cd.plot(other = cd_temp)
            
            chi_squared = abs_snr*(abs_temp - sim_abs) + cd_snr*(cd_temp - sim_cd)
            #print(type(chi_squared))
            print("Chi^2: ", chi_squared)
            return chi_squared
        
        current_chi = chi_squared(start)
        print("Initial Chi^2: " + str(current_chi))
        
    
        output = sp.optimize.least_squares(chi_squared, 
                                           start, 
                                           jac='3-point', 
                                           bounds=([limits[i][0] for i in range(len(limits))], 
                                                    [limits[i][1] for i in range(len(limits))]), 
                                           method='trf', 
                                           ftol=1e-08, 
                                           xtol=1e-08, 
                                           gtol=1e-08, 
                                           x_scale=(1e-10, .1, .1, 100), 
                                           loss='linear', 
                                           f_scale=1.0, 
                                           diff_step=(.01e-10, .001,.001, 3), 
                                           tr_solver=None, 
                                           tr_options={}, 
                                           jac_sparsity=None, 
                                           max_nfev=None, 
                                           verbose=2, 
                                           args=(False,))
                                           
        return output
        
    

class Gen_2DFS:        
    '''class for generating 2D spectra, since it's complicated enough to
    deserve it's own class. Also, dividing it up is proabbly always better
    than blocking it out.
    This is based heavily on "Conformation and Electronic Population Transfer 
    in Membrane Supported Self-assembled Porphyrin Dimers by 2D Fluorescence 
    Spectroscopy" - A. Ortiz.
    
    attr: Hil_Spa, dipole_array
    meth: dip_mom, rot_avg_dip, '''

    def __init__(self, hil_spac, laser_file_name, params = None, l_sig = 1.2, l_shift = 0, shift12 = 0, shift34 = 0):
        '''generates an instance of a 2DFS Spectra generator'''
        self.shifts = (shift12, shift34)
        self.hs = hil_spac
        self.params = params
        self.l_sig = l_sig
        # need a mu_between[vec_a,vec_b] look up table
        vecs = self.hs.ham.vecs
        tot_dim = self.hs.tot_dim
        
        dip_x, dip_y, dip_z = self.hs.dipole_op()
        
        mu_between = np.zeros((tot_dim, tot_dim), dtype=object)
        
        for i in range(tot_dim):
            for j in range(tot_dim):
                #print(dip_x * vecs[j])
                mu_x_between = vecs[i] * (dip_x * vecs[j])
                mu_y_between = vecs[i] * (dip_y * vecs[j])
                mu_z_between = vecs[i] * (dip_z * vecs[j])
                
                mu_between[i,j] = npa([mu_x_between, mu_y_between, mu_z_between])
        self.mu_between = mu_between
        #print("<><><>")
        # need a transition energy look up:
        eigs = self.hs.ham.eigs
        
        delta_e = np.zeros((tot_dim, tot_dim), dtype=float)
        for i in range(tot_dim):
            for j in range(tot_dim):
                delta_e[i, j] = abs(eigs[i] - eigs[j])
        self.delta_e = delta_e
        
        # need a laser amplitude transition array
        
        # need laser shifted transitions:
        alpha_array = np.zeros((tot_dim, tot_dim), dtype=float)
        delta_e_shifted = np.zeros((tot_dim, tot_dim), dtype=float)
        
        l = Laser(laser_file_name, l_shift)
        #print(l.params)
        self.laser = l
        
        for i in range(self.hs.tot_dim):
            #print("<>", end = "")
            for j in range(self.hs.tot_dim):
                ps = l.overlap_params(self.delta_e[i,j], self.params[0])
                alpha_array[i, j] = ps[1]
                delta_e_shifted[i, j] = ps[0]
                
        #delta_e_shifted = delta_e_shifted/np.amax(delta_e_shifted)
        self.delta_e_shifted = delta_e
        #print("laser shifted transitions")
        #print(delta_e_shifted)
        #plt.matshow(delta_e_shifted)
        #plt.show()
        #plt.cla()
        alpha_array = alpha_array/np.amax(alpha_array)
        self.alpha_array = alpha_array
        #print("laser intensity at transition")   
        #print(alpha_array)
        #plt.matshow(alpha_array)
        #plt.show()
        #plt.cla()
        #print("dipole terms")   
                #print(dmag[i,j])
        #plt.matshow(dmag)
        #plt.show()
        #plt.cla()
        
        
        ###-----### load and store databases ###-------###
        
        self.time_db_rp  = DB.SqliteDict("rp_lookup.db",  flag = "r")
        
        self.time_db_nrp = DB.SqliteDict("nrp_lookup.db", flag = "r")
        
        #print("<><><>")
        
    def __repr__(self):
        return "<> 2DFS Gen: "+str(self.hs)+" <>"
    
    def rp_laser_filter(self, steps):
        '''determines whether to add a particular pathway contribution or not.
        returns the weighted signal, or False'''
        # <> using a 2 sigma standard, but this can be easily changed
        
        # defining 'limits'
        
        #print(self.laser.params)
        
        low = self.laser.params[1]  - self.l_sig*self.laser.params[2]
        high = self.laser.params[1] + self.l_sig*self.laser.params[2]
        #print(low, high)
        
        # now look at transitions from the pathway and make sure they are
        # within bounds
        #print(self.delta_e_shifted[steps[0][0], steps[0][1]])
        #print(self.delta_e_shifted[steps[3][0], steps[3][1]])
        w12 = tens_place(self.delta_e_shifted[steps[0][0], steps[0][1]])
        w34 = tens_place(self.delta_e_shifted[steps[3][0], steps[3][1]])
        #print(w12, w34)
        if ((w12<high) and (w12>low) and (w34<high) and (w34>low)):
            #print(w12, w34)
            
            return (self.alpha_array[steps[0][0], steps[0][1]] *\
                    self.alpha_array[steps[1][0], steps[1][1]] *\
                    self.alpha_array[steps[2][0], steps[2][1]] *\
                    self.alpha_array[steps[3][0], steps[3][1]]) *\
                    self.time_db_rp[str((w12+self.shifts[0], w34+self.shifts[1]))]
        
        else:
            return False # don't use!
        
    def nrp_laser_filter(self, steps):
        '''determines whether to add a particular pathway contribution or not.
        returns the weighted signal, or False'''
        # <> using a 2 sigma standard, but this can be easily changed
        
        # defining 'limits'
        
        low = self.laser.params[1]  - self.l_sig*self.laser.params[2]
        high = self.laser.params[1] + self.l_sig*self.laser.params[2]
        
        # now look at transitions from the pathway and make sure they are
        # within bounds
        
        w12 = tens_place(self.delta_e_shifted[steps[0][0], steps[0][1]])
        w34 = tens_place(self.delta_e_shifted[steps[3][0], steps[3][1]])
        
        if (w12<high) and (w12>low) and (w34<high) and (w34>low):
            
            return (self.alpha_array[steps[0][0], steps[0][1]] *\
                    self.alpha_array[steps[1][0], steps[1][1]] *\
                    self.alpha_array[steps[2][0], steps[2][1]] *\
                    self.alpha_array[steps[3][0], steps[3][1]]) *\
                    self.time_db_nrp[str((w12+self.shifts[0], w34+self.shifts[1]))]
        
        else:
            return False # don't use!
        
    def get_steps(self, signal, v, e, ep, f = 0):
        '''determines the intensity of a 2DFS peak based on the
           state trajectory. 
           
           5 State_Vecs -> num'''
        path_dic = {"Q5A":((e, 0),(v, e),(ep, v),(v, ep)),
                    "Q2A":((e, 0),(0, ep),(ep, v),(v, e)),
                    "Q3B":((e, 0),(0, ep),(f, e),(ep, f)),
                    "Q4A":((0, e),(e, v),(v, ep),(v, ep)),
                    "Q3A":((v, e),(ep, v),(e, v),(v, ep)),
                    "Q2B":((0, e),(ep, 0),(f, ep),(e, f))
                    }
        
        steps = path_dic[signal]
        return steps
    
    
    
    
    def rot_avg_mu(self, steps):
        '''determines the intensity of a 2DFS peak based on the
           state trajectory. '''
        
        # copying this directly from prior code, I think it makes sense.
        def k_del(a1, a2, b1, b2): return int(a1 == a2) * int(b1 == b2)

        def I4xxxx(a,b,c,d):    # 4 intermediates
            '''perform an "orientational averaging" over a particular path'''
            return  (1/30)*(7*k_del(a, d, b, c) -  \
                     3*k_del(a, c, b, d) + \
                     2*k_del(a, b, c, d))
        
        
        tot = 0
        for initial_final in steps:
            tot += np.sum((self.mu_between[initial_final[0], initial_final[1]])**2)

        '''
        for m1 in range(3):
            for m2 in range(3):
                for m3 in range(3):
                    for m4 in range(3):
                        tot += I4xxxx(m1, m2, m3, m4) * \
                                self.mu_between[steps[0][0], steps[0][1]][m1]*\
                                self.mu_between[steps[1][0], steps[1][1]][m2]*\
                                self.mu_between[steps[2][0], steps[2][1]][m3]*\
                                self.mu_between[steps[3][0], steps[3][1]][m4]
        #'''
        return tot
    
    
    
    def RP(self, Gamma = .5):
        '''generates and stores an undamped rp spectra for the system
        nv = self.chs.conf.
        
        '''
        nv = self.hs.conf.nv
        eps = self.hs.ham.eigs
        
        # need e, e prime (ep), f indices (singly and doubly excited)
        vs  = range(0, nv**2)
        es  = range(nv**2, 3*nv**2)
        eps = range(nv**2, 3*nv**2)
        fs  = range(3*nv**2, 4*nv**2)
        
        ################ HELPER FUNCTIONS ##################
        ##       - have access to es, eps, fs
        
        def gsb_rp():
            '''adds up the contributions from all the gsb transitions
            None -> [Data_2D, Data_2D, Data_2D, Data_2D] '''       
            
            tot = np.zeros((30,30), dtype = complex)
            
            # sum over singly excited states
            for v in vs:
                for e in es:
                    for ep in eps:
                        
                        steps = self.get_steps("Q4A", v, e, ep)
                        
                        l = self.rp_laser_filter(steps)
                        if type(l) == bool:
                            pass
                        
                        else:
                            tot += self.rot_avg_mu(steps) * l
            return tot
        
        def se_rp():
            '''adds up the contributions from all the gsb transitions
            None -> [Data_2D, Data_2D, Data_2D, Data_2D] '''       
            
            # sum over singly excited states
            
            tot = np.zeros((30,30), dtype = complex)
            
            for v in vs:
                for e in es:
                    for ep in eps:
                        
                        steps = self.get_steps("Q3A", v, e, ep)
                        
                        l = self.rp_laser_filter(steps)
                        if type(l) == bool:
                            pass
                        
                        else:
                            tot += self.rot_avg_mu(steps) * l
            return tot
        
        def esa_rp():
            '''adds up the contributions from all the gsb transitions
            None -> [Data_2D, Data_2D, Data_2D, Data_2D] '''       
            
            # sum over singly excited states
            
            tot = np.zeros((30,30), dtype = complex)
            
            for f in fs:
                for e in es:
                    for ep in eps:
                        
                        steps = self.get_steps("Q2B", 0, e, ep, f)
                        
                        l = self.rp_laser_filter(steps)
                        if type(l) == bool:
                            pass
                        
                        else:
                            tot += self.rot_avg_mu(steps) * l
            return tot
        #####################################################
        
        a = gsb_rp()
        b = se_rp()
        c = esa_rp()
        
        
        return npa([a, b, c])
        
    def NRP(self, Gamma = .5):
        '''generates and stores an undamped rp spectra for the system
        
        
        '''
        nv = self.hs.conf.nv
        eps = self.hs.ham.eigs
        
        # need e, e prime (ep), f indices (singly and doubly excited)
        vs  = range(0, nv**2)
        es  = range(nv**2, 3*nv**2)
        eps = range(nv**2, 3*nv**2)
        fs  = range(3*nv**2, 4*nv**2)
        
        ################ HELPER FUNCTIONS ##################
        ##       - have access to es, eps, fs
        
        def gsb_nrp():
            '''adds up the contributions from all the gsb transitions
            None -> [Data_2D, Data_2D, Data_2D, Data_2D] '''       
            
            # sum over singly excited states
            tot = np.zeros((30,30), dtype = complex)
            
            for v in vs:
                for e in es:
                    for ep in eps:
                        
                        steps = self.get_steps("Q5A", v, e, ep)
                        
                        l = self.nrp_laser_filter(steps)
                        if type(l) == bool:
                            pass
                        
                        else:
                            tot += self.rot_avg_mu(steps) * l
            return tot
        
        def se_nrp():
            '''adds up the contributions from all the gsb transitions
            None -> [Data_2D, Data_2D, Data_2D, Data_2D] '''       
            
            # sum over singly excited states
            tot = np.zeros((30,30), dtype = complex)
            
            for v in vs:
                for e in es:
                    for ep in eps:
                        
                        steps = self.get_steps("Q2A", v, e, ep)
                        
                        l = self.nrp_laser_filter(steps)
                        if type(l) == bool:
                            pass
                        
                        else:
                            tot += self.rot_avg_mu(steps) * l
            return tot
        
        def esa_nrp():
            '''adds up the contributions from all the gsb transitions
            None -> [Data_2D, Data_2D, Data_2D, Data_2D] '''       
            
            # sum over singly excited states
            tot = np.zeros((30,30), dtype = complex)
            
            for f in fs:
                for e in es:
                    for ep in eps:
                        
                        steps = self.get_steps("Q3B", 0, e, ep, f)
                        
                        l = self.nrp_laser_filter(steps)
                        if type(l) == bool:
                            pass
                        
                        else:
                            tot += self.rot_avg_mu(steps) * l
            return tot
        #####################################################
        
        a = gsb_nrp()
        b = se_nrp()
        c = esa_nrp()
        
        
        return npa([a, b, c])

                
    def gen_2dfs(self, cap_gam, monochrom1, monochrom2 = None):
        '''creates an instance of a New_Data_2D, which stores all information for
        both RP and NRP in one object.
        
        num, num, [num] -> [Data_2D, RP parts, NRP parts'''
        if monochrom2 == None:
            monochrom2 = monochrom1
            
        output = self.RP()
        output2 = self.NRP()

        
                # 2D simulation, over some axes
        ax1 = npa(range(0, 30, 1))*2.66e-15 # 30 2.66 fmptsec steps
        ax2 = npa(range(0, 30, 1))*2.66e-15
#        
#        
#        decay_rp = np.zeros((30,30), dtype = float)
#        decay_nrp = np.zeros((30,30), dtype = float)
#        
#        for i in ranlen(ax1):
#            for j in ranlen(ax2):
#                decay_rp[i,j] = rp_decay(ax1[i], ax2[j], 100, 200)
#                decay_nrp[i,j] = nrp_decay(ax1[i], ax2[j], 100, 200)
#        
#        output *= -decay_rp
#        output2*= -decay_nrp
        
        
        dat= Data_2D(-1*(output[0] + output[1] + output[2]*(1-cap_gam)),
                     -1*(output2[0] + output2[1] + output2[2]*(1-cap_gam)), (ax1, ax2))
        dat.FT(2.66e-15, monochrom1, monochrom2)
        #dat.paper_plot()
        
        return [dat, output, output2]
    
    
    def broadening_opt(self, data_directory, suffix, start, limits, scan, monochrom = 17557, 
                       omega = True, quick_plot= False):
        '''minimize chi squared between a simulated Abs and CD spectra and data
        by exploring the conformational space of the coupled dimer system.
        
        str, str, array_like -> OptimizeResult (see sp.optimize.minimize)'''
        # import Abs/CD data as normalized 1D data objects
        data = import_2D_data(data_directory, suffix, monochrom)
        data.FT(2.66e-15, monochrom)
        data.paper_plot()#filename = filename_temp)
        
        # We use the energy axis from the data to generate the simulated spectra
        x, y = data.axes
        
#        # Now we need to define the function to be minimized
#        self.hs.build_cy3dimer_ham(  temp,     model    )
#            
        sim, rp_parts, nrp_parts = self.gen_2dfs(1, monochrom)
            
        def chi_squared_(params, scan, plot = True):
            '''calculates a chi_squared value between two data sets'''
            # first we unpack the input parameter array
            homo, inhomo, cap_gam = params
            
            # build a new Cy-3 Dimer hamiltonian
            #                            temp,    model (point vs exten)
            
            sim2 = Data_2D(-1*(rp_parts[0] + rp_parts[1] + rp_parts[2]*(1-cap_gam)),
                           -1*(nrp_parts[0] + nrp_parts[1] + nrp_parts[2]*(1-cap_gam)), sim.axes)

            new_sim = sim2.decayed(homo, inhomo)
            new_sim.FT(2.66e-15, 17534)
            if quick_plot:
                new_sim.paper_plot(filename = filename_temp)
            chi = new_sim.chi_squared(data,  scan = True)
            
            #print("Chi^2: ", chi_squared)
            return chi
        
        print("Initial Chi^2: " + str(chi_squared_(start, scan, plot = False)))
        
        
        #Time for the actual minimization!!!
        #

        #
        # Output from minimization:
        output = minimize(chi_squared_, 
                          start,
                          args = (True, False),
                          bounds = limits,
                          options = {"maxiter":55, "ftol":1.11e-12, "factr":1e7})
        
        print()
        print(output)
        chi_squared_(output.x, True, plot = True)
        
        return output
    
    


            


