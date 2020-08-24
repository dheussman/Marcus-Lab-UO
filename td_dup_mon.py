print("TD Duplex Monomer")
############# IMPORT STATEMENTS ##############
from JToolKit import *
from sim_2d import *
# Tools
import numpy as np
import scipy as sp
from scipy.optimize import minimize, minimize_scalar # for minimization.
#from data_management import import_2d_data
# Stop Warning Messages
import warnings
#warnings.filterwarnings('ignore')   # optional filtering of warnings..
def  array(array_like): return np.array(array_like)
###############################################

######### TEMPORARY VARIABLE DEFINITIONS #######

########### VARIABLE DEFINITIONS ##############
Pi = np.pi

low = 16700
high = 22000 # Data ranges
time_step = 2.66e-15 
monochrom = 19243

monochrom1 =  17543
monochrom2 =  17543

laser_filename = "20190417/laseratsample_File3_17-39-56-003.txt"
################################################

###############################################

### <> Hilbert Space Definition <> ###

# The springboard for all of our system calculations is the definition of our 
# Hilbert Space. For our system, we have 4 subspaces, two electronic and two 
# vibrational. For 6 vibrational modes, we end up with dimmensionality of 
# (2 * 2 * 6 * 6) = 144. The HilbSpace also requires a Conformation object,
# which defines the system structure.
# Lets build one for our Dimer Model

inds = ('d','c','g','f',"g",'h','i','j')
ts   = ( 1,  5,  15, 25, 35, 45, 55, 65)


############ Conformational Parameters #################

params = {1:(500e-10, 0, 0),
          5:(500e-10, 0, 0),
          15:(500e-10, 0, 0),
          25:(500e-10, 0, 0),
          35:(500e-10, 0, 0),
          45:(500e-10, 0, 0),
          55:(500e-10, 0, 0),
          65:(500e-10, 0, 0)
          }

directory = "20190515"

temp_index = {1: "d", 5: "c", 15:"g", 25:"f", 35:"g", 45:"h", 55:"i", 65:"j"}

laser_file_name = "oceanview/20190422/laser at sample_File3_20-43-23-673.txt"

model = "exten" # "point" - point dipole model
                # "exten" - extended dipole model
                


results = []
plus_bars = []
minus_bars = []

chis = []
for i in range(0, 8):
    nv = 7
    r, phi, theta = params[ts[i]]
    print("Temp: ", ts[i])
    dat = import_2D_data(directory, inds[i], monochrom1, bad1 = False)
    dat.FT()
    dat.plot_with_tc(filename = str(ts[i]) + "dup_mon_dat.pdf")
    #'''
    params[ts[i]]
    
#    nv = 5                          # vibrational levels
#    r = 5.8 * 1e-10                       # interchromophore sep
#    phi = 86 * (np.pi/180)          # twist angle
#    theta = 2 * (np.pi/180)         # tilt angle
#    
    
    
    conf = Conformation(r, phi, theta, nv)  # stores all structural details
    
    hs = HilbertSpace(('eA', 'eB', 'vA', 'vB'),   # subspaces
                      (2, 2, conf.nv, conf.nv),   # dimmensionality
                      conf)                       # structural details
    
    hs.build_cy3dimer_ham(ts[i], "exten")  # "point" - point dipole model
                                        # "exten" - extended dipole model
    
    # need laser object:
    l = Laser(laser_filename, shift = 0)
        
                                # hilbert space,  laser spec file,    broadening params
        
        
        
        
    spec_gen = Sim_2D(      hs,         l,          monochrom1)
    
    sim = spec_gen.gen_2dfs()[0]
    
    output = spec_gen.optimize_gam_sig_thet(dat, 
                                            (100,100), 
                                            [(-50,300),(0, 400)], 
                                            brute_force = True
                                            )
        
    results.append(output[0])
    chis.append(output[1])
    gam, sig = output[0]
    
    ### Error Bars ###
    sim_copy = sim.decayed(gam, sig)


    base = sim_copy.chi_squared(dat)
    new_chi = base
    # +:
    i = 0
    while new_chi < base*1.01:
        print("*", end = '')
        new_chi = (sim.decayed(gam+i, sig)).chi_squared(dat)
        i += .1
    new_chi = base
    j = 0
    while new_chi < base*1.01:
        new_chi = (sim.decayed(gam, sig+j)).chi_squared(dat)
        j += .1
    plus_bars.append([gam + i, sig + j])
    new_chi = base
    # -:
    i = 0
    while new_chi < base*1.01:
        new_chi = (sim.decayed(gam-i, sig)).chi_squared(dat)
        i += .1
    new_chi = base
    j = 0
    while new_chi < base*1.01:
        new_chi = (sim.decayed(gam, sig-j)).chi_squared(dat)
        j += .1
    minus_bars.append([gam - i, sig - j])
    
        
        
    
    
print(results)
plt.plot(ts, npa(results)[:,0], c = 'b', label = "Gamma_H")
plt.plot(ts, npa(plus_bars)[:,0], c = 'b', dashes=[10, 5, 10, 5])
plt.plot(ts, npa(minus_bars)[:,0], c = 'b', dashes=[10, 5, 10, 5])
plt.plot(ts, npa(results)[:,1], c = 'r', label = "Sigma_I")
plt.plot(ts, npa(plus_bars)[:,1], c = 'r', dashes=[10, 5, 10, 5])
plt.plot(ts, npa(minus_bars)[:,1], c = 'r', dashes=[10, 5, 10, 5])
plt.legend()
plt.xlabel("Temperature (C)")
plt.ylabel("Wave #s")
plt.savefig("dup_mon_results.pdf")
plt.show()



print(chis)
plt.plot(ts, npa(chis)/10, label = "Chi Squared")
plt.ylim((0,5))
plt.legend()
plt.xlabel("Temperature (C)")
plt.ylabel("arb.")
plt.savefig("dup_mon_chi_squared.pdf")
plt.show()

print(results)

import csv
my_dict = {'gam':npa(results)[:,0], 
           '+':npa(plus_bars)[:,0], 
           '-':npa(minus_bars)[:,0], 
           'sig':npa(results)[:,1],
           '+':npa(plus_bars)[:,1], 
           '-':npa(minus_bars)[:,1], 
           "chi":npa(chis)/10
           }


with open('td_dup_mon.csv', 'w') as f:
    for key in my_dict.keys():
        f.write("%s,%s\n"%(key,my_dict[key]))



