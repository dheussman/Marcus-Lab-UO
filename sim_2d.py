#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:49:54 2019

Designed to simulate 2DFS data, simplified and stripped down.

    Import Statements:
        
    
    # Database Building # Coimmented, only run once.
    
    Class Sim_2D:
    
        
    
    


@author: jkittell
"""
##### Helper Functions  #######
def tens_place(x): return int(x - x%10)
def npa(array_like): return np.array(array_like)
def debug(): print("<>")

def generator2data(generator_output, cap_theta):
    return None

import numpy as np
import sqlitedict as DB
from data_management import *
import matplotlib.pyplot as plt



class Sim_2D():     
    '''class for generating 2D spectra, since it's complicated enough to
    deserve it's own class. Also, dividing it up is proabbly always better
    than blocking it out.
    This is based heavily on "Conformation and Electronic Population Transfer 
    in Membrane Supported Self-assembled Porphyrin Dimers by 2D Fluorescence 
    Spectroscopy" - A. Ortiz.
    '''
    
    def __init__(self, hil_spac, laser_object, monochromator):
        '''generates an instance of a 2DFS Spectra generator'''
        self.hs = hil_spac
        self.l = laser_object
        
        
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
        
        self.mc = monochromator
        
        
        # need a transition energy look up:
        eigs = self.hs.ham.eigs - self.mc
        self.eigs = eigs
        
        delta_e = np.zeros((tot_dim, tot_dim), dtype=float)
        for i in range(tot_dim):
            for j in range(tot_dim):
                delta_e[i, j] = abs(eigs[i] - eigs[j])
        
        self.delta_e = delta_e
        
        
        
        
        # need a laser amplitude transition array
        
        # need laser shifted transitions:
        alpha_array = np.zeros((tot_dim, tot_dim), dtype=float)
        delta_e_shifted = np.zeros((tot_dim, tot_dim), dtype=float)
        
        
        for i in range(self.hs.tot_dim):
            #print("<>", end = "")
            for j in range(self.hs.tot_dim):                  # SHOULD BE CLOSE.........
                ps = self.l.overlap_params(self.delta_e[i,j], 300)
                alpha_array[i, j] = ps[1]
                delta_e_shifted[i, j] = ps[0]
        
        #delta_e_shifted
        self.delta_e_shifted = delta_e_shifted
        
        alpha_array = alpha_array/np.amax(alpha_array)
        self.alpha_array = alpha_array
        plt.matshow(self.alpha_array)
        #plt.show()
        
        
        
        ###-----### load and store database ###-------###
        
        self.time_db  = DB.SqliteDict("signal_lookup.db",  flag = "r")
        
        
        
        # state groupings
        
        nv = self.hs.conf.nv
        
        self.vs  = range(      0,   nv**2)
        self.es  = range(  nv**2,   3*nv**2)
        self.eps = range(  nv**2,   3*nv**2)
        self.fs  = range(3*nv**2,   4*nv**2)
        
        
        

        
    def __repr__(self):
        return "<> 2DFS Gen: "+str(self.hs)+" <>"
    
        
    def mu_block(self, a, b, c, d):
        '''a in a duple for mu_between[a[0], a[1]]
        also calculates laser intesity'''
        mu_tot = 0
        for initial_final in (a, b, c, d):
            i, f = initial_final
            mu_tot += np.sum((self.mu_between[initial_final[0], initial_final[1]])**2)
        #print("mu_tot: ", mu_tot)
        #print("alpha ", i, f, self.alpha_array[i, f])
        tot = mu_tot * self.alpha_array[i, f]
            
            #print("alpha: ", self.alpha_array[i, f])
        #### NEED AVERAGING HERE FOR CONSISTENCY  #####
        #print("w_alpha: ", tot)
        return tot
    
    
    def rp_response(self, two,one,four,three):
        '''calculates response for both rpo and nrp
        
        num, num, num, num -> array'''
        
        w21 = tens_place(self.delta_e_shifted[two, one]-self.mc)
        #print(w21)
              
        w43 = tens_place(self.delta_e_shifted[four, three]-self.mc)
        
        try:
            return self.time_db[str((-w21, w43))]
        except KeyError:
            print(".")
            return np.zeros((30,30), dtype = complex)

    
    
    def nrp_response(self, two,one,four,three):
        '''calculates response for both rpo and nrp
        
        num, num, num, num -> array'''
        
        w21 = tens_place(self.delta_e_shifted[two, one]-self.mc)
        #print(w21)
              
        w43 = tens_place(self.delta_e_shifted[four, three]-self.mc)
        try:
            return np.conjugate(self.time_db[str((w21, w43))])
        except KeyError:
            print(".")
            return np.zeros((30,30), dtype = complex)
    
    
    
    def orientational_averaging(self, mu):
        "BROKEN"   
        return mu
    
    def gsb(self, v, e, ep):
        # laser trim:
        #print((self.l.low<=self.delta_e_shifted[e, 0]<=self.l.high)and(self.l.low<=self.delta_e_shifted[ep, v]<=self.l.high))
        if (self.l.low <= self.delta_e_shifted[e, 0] <= self.l.high) and \
            (self.l.low <= self.delta_e_shifted[ep, v] <= self.l.high):
            #debug()

            # REPHASING:
            rp = self.rp_response(e, 0,          ep, v)
            #print(rp)
            averaged_mu = self.mu_block((0,e), (e,v), (0,ep), (v,ep))
            rp *=  averaged_mu
            
            # NONREPHASING:
            nrp = self.nrp_response(e, 0,          ep, v)
            avergaed_mu = self.mu_block((e,0), (v,e), (ep,v), (0,ep))
            nrp *= averaged_mu
            #print(rp)
            #1/0
            return (np.conjugate(rp.T), nrp.T)
    
        else:   
            #debug()
            #debug()
            return (np.zeros((30,30)), np.zeros((30,30)))
                
    def se(self, v, e, ep):

        # laser trim:
        if (self.l.low <= self.delta_e_shifted[e, 0] <= self.l.high) and \
            (self.l.low <= self.delta_e_shifted[ep, v] <= self.l.high):
            
            # REPHASING:
            rp = self.rp_response(e, 0,          ep, v)
            averaged_mu = self.mu_block((0,e), (e,v), (0,ep), (v,ep))
            rp *=  averaged_mu
            
            # NONREPHASING:'
            nrp = self.nrp_response(e, 0,          ep, v)
            avergaed_mu = self.mu_block((e,0), (v,e), (ep,v), (0,ep))
            nrp *= averaged_mu
                        
            return (np.conjugate(rp.T), nrp.T)
    
        else:
            return (np.zeros((30,30)), np.zeros((30,30)))
        
    def esa(self, e, ep, f):

        # laser trim:
        if (self.l.low <= self.delta_e_shifted[e, 0] <= self.l.high) and \
            (self.l.low <= self.delta_e_shifted[f, e] <= self.l.high):

            # REPHASING:
            rp = self.rp_response(e, 0, f, e)
            averaged_mu = self.mu_block((0,e), (ep,0), (f,ep), (e,f))
            rp *=  averaged_mu
            
            # NONREPHASING:
            nrp = self.nrp_response(e, 0, f, e)
            avergaed_mu = self.mu_block((e,0), (0,ep), (f,ep), (ep,f))
            nrp *= averaged_mu
                        
            return (np.conjugate(rp.T), nrp.T)
    
        else:
            return (np.zeros((30,30)), np.zeros((30,30)))


                        

    def gen_2dfs(self, cap_theta = 1):
        '''returns a simulated 2dfs spectra for a given hilbert space and laser object
        
        none -> (full, gsb, se, esa)
        '''
        
        #gsb
        rp, nrp = (np.zeros((30,30), dtype = complex), np.zeros((30,30), dtype = complex))
        for v in self.vs:
            for e in self.es:
                for ep in self.eps:
                    temp_rp, temp_nrp = self.gsb(v,e,ep)
                    #plt.matshow(np.real(temp_rp))
                    #plt.show()
                    rp  += temp_rp
                    nrp += temp_nrp

        #`1`print(nrp)
        gsb = Data_2D(rp, nrp)
        
        # se
        rp, nrp = (np.zeros((30,30), dtype = complex), np.zeros((30,30), dtype = complex))
        for v in self.vs:
            for e in self.es:
                    temp_rp, temp_nrp = self.se(v,e,ep)
                    rp  += temp_rp
                    nrp += temp_nrp
        se = Data_2D(rp, nrp)
        
        # esa
        rp, nrp = (np.zeros((30,30), dtype = complex), np.zeros((30,30), dtype = complex) )
        for e in self.es:
            for ep in self.eps:
                for f in self.fs:
                    temp_rp, temp_nrp = self.esa(e,ep,f)
                    rp  += temp_rp
                    nrp += temp_nrp
        esa = Data_2D(rp, nrp)
        
        return [gsb + se + esa.scale(1-cap_theta), gsb, se, esa]



    def optimize_gam_sig_thet(self,
                              data_object, 
                              start, limits,
                              plot = False,
                              brute_force = True,
                              filename1 = None):
        '''takes a simulated, unbroadened data object and broadens it to best
        fit the actual data'''
        
        if plot:
            data_object.plot_with_tc()
        
        # MAKE SIMULATION 
        sim = self.gen_2dfs()
        
        def broad_chi(gam_sig, plot = False):
            '''apply a broadening function to a simulation and return the chi^2
            between it and a function'''
            #print("<>", end = "")
            gam, sig = gam_sig
            #print(gam_sig_thet)
            undamped = sim[1] + sim[2] + sim[3].scale(1-(0/100))
            #undamped.FT()
            #undamped.super_plot()
            damped   = undamped.decayed(gam, sig)
            damped.FT()
            damped.normalize()
            
            if plot:
                damped.plot_with_tc(filename = filename1)
            
            return data_object.chi_squared(damped)
        
        #Time for the actual minimization!!!
        #

        #
        # Output from minimization:
        if brute_force == True:
            output =    brute(broad_chi,
                              limits,
                              Ns = 80,
                              full_output = True,
                              finish = None)
            
            print()
            print(output[0], output[1])
            broad_chi(output[0], plot = True)
        else: 
            output = minimize(broad_chi, 
                          start,
                          bounds = limits,
                          options = {"maxiter":55, "ftol":1.11e-12, "factr":1e7})
        
            print()
            print(output)
            broad_chi(output.x, plot = True)
        return output
        










