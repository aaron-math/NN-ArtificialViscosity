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
    title = "2d_sod_base"
elif RESOLVED:
    title = "2d_sod_res"
elif Test:
    title = "2d_sod_%s"%Model

## Define a mesh
L = np.pi * 2.0
gamma = 1.4
dim = 2

problem = 'sod'

Lp = L * (Npts-1.0) / Npts
mesh_options = {}
mesh_options['coordsys'] = 0
mesh_options['periodic'] = np.array([False, False, True])
mesh_options['dim'] = 3
mesh_options['x1'] = [ 0.0 , 0.0  ,  0.0 ]
mesh_options['xn'] = [ Lp   , Lp    ,  Lp ]
mesh_options['nn'] = [ Npts, 1 ,  1  ]
if dim == 2:
    mesh_options['nn'] = [ Npts, Npts ,  1  ]

# Initialize a simulation object on a mesh
ss = pyrandaSim(title,mesh_options)
ss.addPackage( pyrandaBC(ss) )

# Define the equations of motion
eom ="""
# Primary Equations of motion here
ddt(:rho:)  =  -ddx(:rho:*:u:)                  - ddy(:rho:*:v:)
ddt(:rhou:) =  -ddx(:rhou:*:u: + :p: - :tau:)   - ddy(:rhou:*:v:)
ddt(:rhov:) =  -ddx(:rhov:*:u:)                 - ddy(:rhov:*:v: + :p: - :tau:)
ddt(:Et:)   =  -ddx( (:Et: + :p: - :tau:)*:u: ) - ddy( (:Et: + :p: - :tau:)*:v: )
# Conservative filter of the EoM
:rho:       =  fbar( :rho:  )
:rhou:      =  fbar( :rhou: )
:rhov:      =  fbar( :rhov: )
:Et:        =  fbar( :Et:   )
# Update the primatives and enforce the EOS
:u:         =  :rhou: / :rho:
:v:         =  :rhov: / :rho:
:p:         =  ( :Et: - .5*:rho:*(:u:*:u: + :v:*:v:) ) * ( :gamma: - 1.0 )
# Artificial bulk viscosity (old school way)
:div:       =  ddx(:u:) + ddy(:v:)
:beta:      =  gbar(abs(ring(:div:))) * :rho: * 7.0e-2
:tau:       =  :beta:*:div:
:cs:  = sqrt( :p: / :rho: * :gamma: )
"""
if dim == 2:
    eom += """# Apply constant BCs
bc.extrap(['rho','Et'],['x1','xn','y1','yn'])
bc.const(['u','v'],['x1','xn','y1','yn'],0.0)
"""
else:
    eom += """# Apply constant BCs
bc.extrap(['rho','Et'],['x1'])
bc.const(['u','v'],['x1','xn'],0.0)
"""

print(eom)

# Add the EOM to the solver
ss.EOM(eom)

# Initialize variables
if dim == 2:
    ic = """
    rad = sqrt( (meshx-pi)**2  +  (meshy-pi)**2 ) """
    if TEST:
        ic += """
        :ml_beta: = :beta: * 0.0
        :ml_betaX: = :beta: * 0.0
        :ml_betaY: = :beta: * 0.0"""

# Linear wave propagation in 1d and 2d
if (problem == 'linear'):
    pvar = 'p'
    ic += """
    :gamma: = 1.4
    ratio = 1.0 + 0.01 * exp( -(rad)**2/(.2**2) )
    :Et: = ratio
    :rho: = 1.0
    """

# SOD shock tube in 1d and 2d
if (problem == 'sod'):
    pvar = 'rho'
    if dim == 1:
        ic = 'rad = meshx / 2.0'
    ic += """
    :gamma: = 1.4
    :Et:  = gbar( where( rad < pi/2.0, 1.0/(:gamma:-1.0) , .1 /(:gamma:-1.0) ) )
    :rho: = gbar( where( rad < pi/2.0, 1.0    , .125 ) )
    """

# Set the initial conditions
ss.setIC(ic)

# Length scale for art. viscosity
# Initialize variables
x = ss.mesh.coords[0].data
y = ss.mesh.coords[1].data
z = ss.mesh.coords[2].data

# Write a time loop
time = 0.0
viz = False

# Approx a max dt and stopping time
v = 1.0
dt_max = v / ss.mesh.nn[0] * 0.75
#tt = L/v * .125 #dt_max
tt = 0.4

# Mesh for viz on master
xx   =  ss.PyMPI.zbar( x )
yy   =  ss.PyMPI.zbar( y )
ny = ss.PyMPI.ny

# Start time loop
dt = dt_max
cnt = 1
viz_freq = 25

if TEST: #Load NN model if testing
    import tensorflow as tf
    import tensorflow.keras
    model = tf.keras.models.load_model('%s/my_model'%Model)

data_x = np.zeros( ( (ss.nx-(2 * N)),(ss.ny-(2 * N)),2*N+1) ) #Initialize dimensionless x vel data
data_y = np.zeros( ( (ss.nx-(2 * N)),(ss.ny-(2 * N)),2*N+1) ) #Initialize dimensionless y vel data

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
    pvar = 'ml_beta'
    if viz and (not TEST):
        v = ss.PyMPI.zbar( ss.variables[pvar].data )
        if (ss.PyMPI.master and (cnt%viz_freq == 1)) and True:
            plt.figure(1)
            plt.clf()
            if ( ny > 1):
                plt.plot(xx[:,int(ny/2)],v[:,int(ny/2)] ,'k.-')
                plt.title(pvar)
                plt.pause(.001)
                plt.figure(2)
                plt.clf()
                plt.contourf( xx,yy,v ,64 , cmap=cm.jet)
            else:
                plt.plot(xx[:,0],v[:,0] ,'k.-')
            plt.title(pvar)
            plt.pause(.001)


rho_data = ss.var('rho').data

if TEST:
    beta_data = ss.var('ml_beta').data
else:
    beta_data = ss.var('beta').data

if not os.path.exists('Data'):
    os.makedirs('Data')

if BASE:
    np.save("Data/2D_sod_base_density.npy",rho_data) #Save base density data
    np.save("Data/2D_sod_base_av.npy",beta_data) #Save base beta data
elif RESOLVED:
    np.save("Data/2D_sod_res_density.npy",rho_data) #Save resolved density data
    np.save("Data/2D_sod_res_av.npy",beta_data) #Save resolved density data
if TEST:
    np.save("Data/2D_sod_active_density_N%i.npy"%N,rho_data) #Save NN model density data
    np.save("Data/2D_sod_active_av_N%i.npy"%N,beta_data) #Save NN model beta data
