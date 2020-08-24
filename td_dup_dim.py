print("TD Duplex Dimer")
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
###############################################

######### TEMPORARY VARIABLE DEFINITIONS #######

########### VARIABLE DEFINITIONS ##############
Pi = np.pi

low = 16700
high = 22000 # Data ranges
time_step = 2.66e-15 
#monochrom = 19243
#
monochrom1 =  17543 #17543
#monochrom2 =  17400 #17543

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

inds = ('e','g','i','a','l','m','n','o')
ts   = (1, 5, 15, 22.5, 35, 45, 55, 65)

#inds = ('i','a','l','m','n','o')
#ts   = (15, 22.5, 35, 45, 55, 65)

s=0 #I added this to not break code
############ Conformational Parameters #################
params = {1:(2.0e-10, 84.86,0.181),#(6.9e-10, 79.8, 0.0), #using the parameters from 15 degrees 
          5:(2.0e-10, 84.91, 0.042),#(6.1e-10, 82.5, 0.0),
          15:(5.8e-10, 82.9, 1.52),
          22.5:(6.6e-10, 79.9, 7.02),
          35:(7.4e-10, 76.0, 0.00),
          45:(8.1e-10, 72.0, 6.90),
          55:(8.4e-10, 70.2, 9.28),
          65:(8.6e-10, 73.5, 0.71)
          }

#params = {1:(500e-10, 0, 0), #using the parameters from 15 degrees 
#          5:(500e-10, 0, 0),
#          15:(500e-10, 0, 0),
#          22.5:(500e-10, 0, 0),
#          35:(500e-10, 0, 0),
#          45:(500e-10, 0, 0),
#          55:(500e-10, 0, 0),
#          65:(500e-10, 0, 0)
#          }

directory = "20170417"

temp_ind = {1:"e", 5:"g", 15:"i", 22.5:"a", 35:"l", 45:"m", 55:"n", 65:"o"}

laser_file_name = "20190417/laseratsample_File3_17-39-56-003.txt"

model = "exten" # "point" - point dipole model
                # "exten" - extended dipole model

results = [] 
plus_bars = []
minus_bars = [] 

chis = []
#for r in range(3, 10):
for i in range(2):
    nv = 7
    r, phi, theta = params[ts[i]]

    print("Temp: ", ts[i])
    print("Shift: ", s)
    dat = import_2D_data(directory, inds[i], monochrom1, bad1 = False)
    dat.FT()
    dat.plot_with_tc(filename = str(ts[i]) + "dup_dim_dat.pdf")
    #'''
    r, phi, theta = params[ts[i]]
    
    nv = 6                          # vibrational levels
#    r = 5.8 * 1e-10                       # interchromophore sep
#    phi = 86 * (np.pi/180)          # twist angle
#    theta = 2 * (np.pi/180)         # tilt angle
    
    
    
    conf = Conformation(r, phi* (np.pi/180), theta* (np.pi/180), nv)  # stores all structural details
    
    hs = HilbertSpace(('eA', 'eB', 'vA', 'vB'),   # subspaces
                      (2, 2, conf.nv, conf.nv),   # dimmensionality
                      conf)                       # structural details
    
    hs.build_cy3dimer_ham(ts[i], model)  # "point" - point dipole model
                                        # "exten" - extended dipole model
    
    # need laser object:
    l = Laser(laser_filename, shift = 0)
        
                                # hilbert space,  laser spec file,    broadening params
        
        
        
        
    spec_gen = Sim_2D(      hs,         l,          monochrom1)
    sim = spec_gen.gen_2dfs()[0]
    
    output = spec_gen.optimize_gam_sig_thet(dat, 
                                            (199,499), 
                                            [(-50,300),(0, 400)], 
                                            brute_force = True,
                                            filename1 = str(ts[i]) + "dup_dim_sim.pdf")
    
    results.append(output[0])
    chis.append(output[1])
    gam, sig = output[0]
    
    ########################## Error Bars ##########################
    
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
plt.savefig("dup_dim_results.pdf")
plt.show()



print(chis)
plt.plot(ts, npa(chis)/10, label = "Chi Squared")
plt.ylim((0,5))
plt.legend()
plt.xlabel("Temperature (C)")
plt.ylabel("arb.")
plt.savefig("dup_dim_chi_squared.pdf")
plt.show()

print(results)

###################### SAVE TO CSV #####################


import csv
my_dict = {'gam':npa(results)[:,0], 
           '+':npa(plus_bars)[:,0], 
           '-':npa(minus_bars)[:,0], 
           'sig':npa(results)[:,1],
           '+':npa(plus_bars)[:,1], 
           '-':npa(minus_bars)[:,1], 
           "chi":npa(chis)/10
           }

with open('td_dup_dim.csv', 'w') as f:
    for key in my_dict.keys():
        f.write("%s,%s\n"%(key,my_dict[key]))