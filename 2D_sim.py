
############# IMPORT STATEMENTS ##############
from JToolKit import *

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
monochrom = 19243

monochrom1 =  17543
monochrom2 =  17543

laser_file_name = "20190417/laseratsample_File3_17-39-56-003.txt"
################################################

gams =  [91.06510587, 111.7308839, 86.62108276, 97.86720903, 100.20715624, 47.30258699, 28.10687566, 32.25628965]

sigs =  [254.76221343, 258.87063616, 260.41641466, 242.24480546, 251.53552774, 248.62063343, 267.52723949, 265.36941733]

thets = [0.96530034, 1.54578324, 1.66135842, 1.44652546, 0.79551749, 1.28776652, 1.27532797, 1.66385752]

directory = "20190422"

temp_ind = {1: "i", 5: "d", 15:"b", 25:"k", 35:"m", 45:"n", 55:"o", 65:"p"}

laser_file_name = "oceanview/20190422/laser at sample_File3_20-43-23-673.txt"

temps = [1,5,15,25,35,45,55,65]

###############################################

### <> Hilbert Space Definition <> ###

# The springboard for all of our system calculations is the definition of our 
# Hilbert Space. For our system, we have 4 subspaces, two electronic and two 
# vibrational. For 6 vibrational modes, we end up with dimmensionality of 
# (2 * 2 * 6 * 6) = 144. The HilbSpace also requires a Conformation object,
# which defines the system structure.
# Lets build one for our Dimer Model


#dat = import_2D_data("20170417", "k", monochrom1)
#dat.FT(2.66e-15, monochrom1)
#dat.simple_plot()

omega = 590
l_sigg = 1.5

for i in range(8):
    print("Temperature: ", temps[i])
    print("                              <> Data <>" )

    data = import_2D_data(directory, temp_ind[temps[i]], monochrom1)
    data.FT(2.66e-15, monochrom1)
    data.super_plot()
    print("                              <> Simulation <>" )
    nv = 6                           # vibrational levels
    r = 500 * 1e-10                       # interchromophore sep
    phi = 0 * (np.pi/180)          # twist angle
    theta = 0 * (np.pi/180)         # tilt angle
    
    
    
    conf = Conformation(r, phi, theta, nv)  # stores all structural details
    
    hs = HilbertSpace(('eA', 'eB', 'vA', 'vB'),   # subspaces
                      (2, 2, conf.nv, conf.nv),   # dimmensionality
                      conf)                       # structural details
    
    hs.build_cy3dimer_ham(25, "point", omega)  # "point" - point dipole model
                                                # "exten" - extended dipole model
    
        
    
                            # hilbert space,  laser spec file,    broadening params
    
    
    
    
    spec_gen = Gen_2DFS(      hs,         laser_file_name,       (130, 240)    , l_sig = l_sigg, l_shift = -250, shift12 = 0, shift34 = 0)
    
    dat = spec_gen.gen_2dfs(thets[i], monochrom1)[0]
    dat.FT(2.66e-15, monochrom1)
    for gam in range(200, 500, 20):
        
        dat2 = dat.decayed(gam, 50)
        print()
        dat2.super_plot()
    print("gamma: ", gams[i])
    print("sigma: ", sigs[i])
    print("cap thet: ", cap_thets[i])
    
    print()
    print()
    
    '''    
    sim = spec_gen.gen_2dfs(1, monochrom1)[0]
    
    sim = sim.decayed(150, 250)
    sim.FT(2.66e-15, 17543)
    
    sim.simple_plot(filename = "gif_temp/l_shift_" + str(shift1).zfill(4), 
                    title = str(shift1-500))
    
    
    output = spec_gen.broadening_opt("20170417", "k", [50, 50, 1], [(50,300),(50,300),(0,2)],True, monochrom1, 
                                     filename_temp = "gif_temp/" + str(l_shiftt).zfill(5))
    
    '''

