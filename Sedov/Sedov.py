import re
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from pyranda import pyrandaSim, pyrandaIBM, pyrandaBC, pyrandaTimestep

from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', help='Test NN model', action="store_true")
parser.add_argument('--res', help='Use resolved simulation', action="store_true")
parser.add_argument('--N', help='Number of points before/after in domain', default=3, type=int)
parser.add_argument('--model', help='Name of model', default=None, type=str)
args = parser.parse_args()

TEST = args.test
RESOLVED = args.res
N = args.N
Model = args.model

if TEST and Model == None: #Must load model if testing neural network
    raise RuntimeError("Must include model name")

if RESOLVED: #Use 50x points in resolved simulation
    Npts = 128 * 8
else:
    Npts = 128

if (not RESOLVED) and (not TEST):
    BASE = True
else:
    BASE = False

"""Initialize simulation"""
if BASE:
    title = "sedov_base"
elif RESOLVED:
    title = "sedov_res"
elif Test:
    title = "sedov_%s"%Model

## Define a mesh
L = 1.0
gamma = 1.4
dim = 2
thick = 4.0e-2
rho0 = 1.0
E0 = 0.8510718547582291/rho0
zeta0 = 1.032777467761425
alpha = E0


Lp = L * (Npts-1.0) / Npts
mesh_options = {}
mesh_options['coordsys'] = 0
mesh_options['periodic'] = np.array([False, False, True])
mesh_options['dim'] = 2
mesh_options['x1'] = [-Lp  ,-Lp  ,  0.0 ]
mesh_options['xn'] = [ Lp  , Lp  ,  Lp ]
mesh_options['nn'] = [ Npts, Npts,  1  ]



# Initialize a simulation object on a mesh
ss = pyrandaSim(title,mesh_options)
ss.addPackage( pyrandaBC(ss) )
ss.addPackage( pyrandaTimestep(ss) )   # Timestep package allows for "dt.*" functions


# Define the equations of motion
eom ="""
# Primary Equations of motion here
ddt(:rho:)  =  -ddx(:rho:*:u:)                  - ddy(:rho:*:v:)
ddt(:rhou:) =  -ddx(:rhou:*:u: + :p: - :tau:)   - ddy(:rhou:*:v:)
ddt(:rhov:) =  -ddx(:rhov:*:u:)                 - ddy(:rhov:*:v: + :p: - :tau:)
ddt(:Et:)   =  -ddx( (:Et: + :p: - :tau:)*:u: -:tx:*:kappa:   ) - ddy( (:Et: + :p: - :tau:)*:v: -:ty:*:kappa: )
# Conservative filter of the EoM
:rho:       =  fbar( :rho:  )
:rhou:      =  fbar( :rhou: )
:rhov:      =  fbar( :rhov: )
:Et:        =  fbar( :Et:   )
# Update the primatives and enforce the EOS
:u:         =  :rhou: / :rho:
:v:         =  :rhov: / :rho:
:p:         =  ( :Et: - .5*:rho:*(:u:*:u: + :v:*:v:) ) * ( :gamma: - 1.0 )
:R:         =  1.0
:cp:        = :R: / (1.0 - 1.0/:gamma: )
:cv:        = :cp: - :R:
:T:         = :p: / :rho: / :R:
# Artificial bulk viscosity (old school way)
:div:       =  ddx(:u:) + ddy(:v:)
:beta:      =  gbar(abs(ring(:div:))) * :rho: * 7.0e-2
:tau:       =  :beta:*:div:
[:tx:,:ty:,:tz:] = grad(:T:)
:kappa:     = gbar( abs(:T: * ring(:T:)* :rho:*:cv:/(:T: * :dt: ) ) ) * 1.0e-3
# Apply constant BCs
bc.extrap(['rho','Et'],['x1','xn','y1','yn'])
bc.const(['u','v'],['x1','xn','y1','yn'],0.0)
# Compute some max time steps
:cs:  = where( :p: < 0.01 , 1.0 , sqrt( :p: / :rho: * :gamma: ) )
:dtC: = dt.courant(:u:,:v:,:w:,:cs:)
:dtB: = 0.01 * dt.diff(:beta:,:rho:)
:dt:  = numpy.minimum(:dtC:,:dtB:)
"""



# Add the EOM to the solver
ss.EOM(eom)




# Initialize variables
ic = """
:gamma: = 1.4
rad     = sqrt(meshx**2 + meshy**2)/thick
:wgt:     = exp(-rad**2)/(thick**2 * sqrt( pi**2) )
:rho:  += rho0
:Et:   += E0 * :wgt: + .001
"""
if TEST:
    ic += """
    :ml_beta: = :beta: * 0.0
    :ml_betaX: = :beta: * 0.0
    :ml_betaY: = :beta: * 0.0"""

icDict = {}
icDict['rho0'] = rho0
icDict['thick'] = thick
icDict['E0'] = E0

# Set the initial conditions
ss.variables['dt'].data = 1.0
ss.setIC(ic,icDict)

# Write a time loop
time = 0.0
tt = 0.5
viz = True

# Start time loop
dt_max = .00015 #ss.var('dt').data * .1
dt = dt_max
cnt = 1
viz_freq = 25

if TEST: #Load NN model if testing
    import tensorflow as tf
    import tensorflow.keras
    model = tf.keras.models.load_model('%s/my_model'%Model)

data_x = np.zeros( ( (ss.nx-(2 * N)),(ss.ny-(2 * N)),2*N+1) ) #Initialize dimensionless x vel data
data_y = np.zeros( ( (ss.nx-(2 * N)),(ss.ny-(2 * N)),2*N+1) ) #Initialize dimensionless y vel data


wvars = ['rho','p','u','v','beta','ml_beta','ml_betaX','ml_betaY']
ss.write(wvars)

while tt > time:
    if RESOLVED:
        time = ss.rk4(time,dt)
    else:
        # Update the EOM and get next dt
        phi = None
        for rkstep in range(5):
            phi = ss.rk4_step(dt,rkstep,phi)
            if TEST:
                beta_tmp = ss.var('beta').data / ss.var('rho').data / ss.var('cs').data / ss.dx #Remove dimensions
                u_tmp = ss.var('u').data / ss.var('cs').data #Remove dimensions
                v_tmp = ss.var('v').data / ss.var('cs').data #Remove dimensions
                for i in range(N,(ss.nx-N)):
                    for j in range(N,(ss.ny-N)):  # BJO - should be ny
                        uu = u_tmp[i-N:i+N+1,j,0]
                        data_x[i-N,j-N,:] = uu
                        vv = v_tmp[i,j-N:j+N+1,0]
                        data_y[i-N,j-N,:] = vv

                nx = data_x.shape[0]
                ny = data_x.shape[1]
                #####  Note the crazy reshaping... on the way in and out of the predict function.
                beta_x = model.predict( data_x.reshape( (nx*ny,2*N+1) )   )[:,0].reshape( (nx,ny) )
                beta_y = model.predict( data_y.reshape( (nx*ny,2*N+1) )   )[:,0].reshape( (nx,ny) )
                val = np.sqrt( beta_x**2 + beta_y**2 )
                beta_tmp[N:-N,N:-N,0] = val

                # Save the directions for vizualization
                ss.var('ml_betaX').data[N:-N,N:-N,0] = beta_x
                ss.var('ml_betaY').data[N:-N,N:-N,0] = beta_y

                # Scaling and masks
                dx = ss.dx
                mask = beta_tmp < 0 #Remove vals < 0
                beta_tmp[mask] = 0

                # Replace dimensions
                rho_tmp = ss.var('rho').data
                cs_tmp = ss.var('cs').data
                beta_tmp = beta_tmp * rho_tmp * dx * cs_tmp

                ss.var('ml_beta').data = beta_tmp #Set data as variable
                ss.parse(":tau:  =  :ml_beta: * :div:")

    ss.cycle += 1
    time += dt
    dt = min(dt_max, (tt - time) )

    wvars = ['rho','p','u','v','beta','ml_beta','ml_betaX','ml_betaY']
    ss.write(wvars)

    # Print some output
    ss.iprint("%s -- %s" % (cnt,time)  )
    cnt += 1
    if viz and (not test):
        if (cnt%viz_freq == 1):
            ss.write(wvars)


rho_data = ss.var('rho').data
rho_data = rho_data.flatten()

if TEST:
    beta_data = ss.var('ml_beta').data
    beta_data = beta_data.flatten()
else:
    beta_data = ss.var('beta').data
    beta_data = beta_data.flatten()

if not os.path.exists('Data'):
    os.makedirs('Data')

if BASE:
    np.save("Data/sedov_base_density.npy",rho_data) #Save base density data
    np.save("Data/sedov_base_av.npy",beta_data) #Save base beta data
elif RESOLVED:
    x = np.linspace(ss.meshOptions['x1'][0],ss.meshOptions['xn'][0],num=200)
    xp = np.linspace(ss.meshOptions['x1'][0],ss.meshOptions['xn'][0],num=ss.nx)
    fp_rho = rho_data
    interpPoints_rho = np.interp(x,xp,fp_rho)
    np.save("Data/sedov_res_density.npy",interpPoints_rho) #Save resolved density data
    fp_beta = beta_data
    interpPoints_beta = np.interp(x,xp,fp_beta)
    np.save("Data/sedov_res_av.npy",interpPoints_beta) #Save resolved density data
if TEST:
    np.save("Data/sedov_active_density_N%i.npy"%N,rho_data) #Save NN model density data
    np.save("Data/sedov_active_av_N%i.npy"%N,beta_data) #Save NN model beta data
