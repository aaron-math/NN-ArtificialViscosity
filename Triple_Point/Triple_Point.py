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

if RESOLVED:
    nx = 350 * 4                             # Points in x
    ny = 150 * 4                             # Points in y
else:
    nx = 350                             # Points in x
    ny = 150                             # Points in y

if (not RESOLVED) and (not TEST):
    BASE = True
else:
    BASE = False

"""Initialize simulation"""
if BASE:
    title = "triple_point_base"
elif RESOLVED:
    title = "triple_point_res"
elif Test:
    title = "triple_point%s"%Model

## Define a mesh
dim = 2                             # Dimension of the problem
open_angle  = 45.0                  # Total opening angle in degrees
open_angle *= (numpy.pi / 180.0)    # Convert to radians
L_min = 2.0                         # Width of tube at inner radius
h = 10.0                            # Height is y direction
mwL = 1.0                           # Molecular weight of light
mwH = 1.0                           # Molecular weight of heavy
myGamma = 1.4                       # Gamma of gases (single gamma)
gammaL = 1.4
gammaH = 1.5

mesh_options = {}
mesh_options['dim'] = dim             # Dimension of the problem
mesh_options['nn'] = [ nx, ny , 1 ]   # Grid spacing
mesh_options['x1'] = [ 0   , 0    ,  0.0 ]
mesh_options['xn'] = [ 7.0 , 3.0  ,  0.0 ]
mesh_options['periodic'] = [False, False, False]


# Initialize a simulation object on a mesh
ss = pyrandaSim(title,mesh_options)
# Add packages to simulation object
ss.addPackage( pyrandaBC(ss) )          # BC package allows for "bc.*" functions
ss.addPackage( pyrandaTimestep(ss) )    # Timestep package allows for "dt.*" functions

# Define the equations of motion
eom ="""
# Primary Equations of motion here
ddt(:rhoYh:) =  -div(:rhoYh:*:u: - :Jx: , :rhoYh:*:v: - :Jy:  )
ddt(:rhoYl:) =  -div(:rhoYl:*:u: + :Jx: , :rhoYl:*:v: + :Jy:  )
ddt(:rhou:)  =  -div(:rhou:*:u: - :tauxx:, :rhou:*:v: - :tauxy:)
ddt(:rhov:)  =  -div(:rhov:*:u: - :tauxy:, :rhov:*:v: - :tauyy:)
ddt(:Et:)    =  -div( (:Et: - :tauxx:)*:u: - :tauxy:*:v: - :tx:*:kappa:, (:Et: - :tauyy:)*:v: - :tauxy:*:u: - :ty:*:kappa: )
# Conservative filter of the EoM
:rhoYh:     =  fbar( :rhoYh:  )
:rhoYl:     =  fbar( :rhoYl:  )
:rhou:      =  fbar( :rhou: )
:rhov:      =  fbar( :rhov: )
:Et:        =  fbar( :Et:   )
# Update the primatives and enforce the EOS
:rho:       = :rhoYh: + :rhoYl:
:Yh:        =  :rhoYh: / :rho:
:Yl:        =  :rhoYl: / :rho:
:u:         =  :rhou: / :rho:
:v:         =  :rhov: / :rho:
:gamma: = :Yh:* gammaH + :Yl: * gammaL
:p:         =  ( :Et: - .5*:rho:*(:u:*:u: + :v:*:v:) ) * ( :gamma: - 1.0 )
:mw: = 1.0 / ( :Yh: / mwH + :Yl: / mwL )
:R: = 1.0 / :mw:
:cp: = :R: / (1.0 - 1.0/:gamma: )
:cv: = :cp: - :R:
:T:         = :p: / (:rho: * :R: )
# Artificial bulk viscosity (old school way)
:div:       =  div(:u:,:v:)
[:ux:,:uy:,:tz:] = grad( :u: )
[:vx:,:vy:,:tz:] = grad( :v: )
:S:         = sqrt( :ux:*:ux: + :vy:*:vy: + .5*((:uy:+:vx:)**2))
:beta:      =  gbar( ring(:div:) * :rho: ) * 7.0e-2
:mu:        =  gbar( abs(ring(:S:  )) ) * :rho: * 1.0e-3
:taudia:    =  (:beta:-2./3.*:mu:) *:div: - :p:
:tauxx:     =  2.0*:mu:*:ux:   + :taudia:
:tauyy:     =  2.0*:mu:*:vy:   + :taudia:
:tauxy:     =  :mu:*(:uy:+:vx:)
[:tx:,:ty:,:tz:] = grad(:T:)
:kappa:     = gbar( ring(:T:)* :rho:*:cv:/(:T: * :dt: ) ) * 1.0e-3
# Artificial species diffusivities
:Dsgs:      =  ring(:Yh:) * 2.0e-4
:Ysgs:      =  1.0e1*(abs(:Yh:) - 1.0 + abs(1.0-:Yh: ) )*gridLen**2
:adiff:     =  gbar( :rho:*numpy.maximum(:Dsgs:,:Ysgs:) / :dt: )
[:Yx:,:Yy:,:Yz:] = grad( :Yh: )
:Jx:        =  :adiff:*:Yx:
:Jy:        =  :adiff:*:Yy:
# Apply boundary conditions
bc.extrap(['rho','p'],['x1','xn','y1','yn'] ,order=1)
bc.extrap(['Yh'],['x1','xn','y1','yn'], order=1 )
bc.const( ['v'] , ['y1','yn'], 0.0 )
bc.const( ['u'] , ['x1','xn'], 0.0 )
bc.extrap(['u'] , ['y1','yn'], order=1 )
bc.extrap(['v'] , ['x1','xn'], order=1 )
:Et:  = :p: / ( :gamma: - 1.0 )  + .5*:rho:*(:u:*:u: + :v:*:v:)
:Yl:  = 1.0 - :Yh:
:rhoYh: = :rho:*:Yh:
:rhoYl: = :rho:*:Yl:
:rhou: = :rho:*:u:
:rhov: = :rho:*:v:
# Compute some max time steps
:cs:  = sqrt( :p: / :rho: * :gamma: )
:dt: = dt.courant(:u:,:v:,:w:,:cs:)
:dtB: = 0.2* dt.diff(:beta:,:rho:)
:dt:  = numpy.minimum(:dt:,:dtB:)
:dtM: = 0.2 * dt.diff(:mu:,:rho:)
:dt: = numpy.minimum(:dt:,:dtM:)
:dtY: = 0.2 * dt.diff(:adiff:,:rho:)
:dt: = numpy.minimum(:dt:,:dtY:)
:umag: = sqrt( :u:*:u: + :v:*:v: )
:lrho: = numpy.log( :rho: )
"""

# Add the EOM to the solver
ss.EOM(eom,{'mwH':mwH,'mwL':mwL,'gammaL':gammaL,'gammaH':gammaH})


# Initial conditions of the flow
ic = """
driver = where( meshx< 1.0, 1.0, 0.0 )
top =  where( driver < 1.0, where( meshy > 1.5, 1.0, 0.0 ) , 0.0)
driver = gbar( gbar(driver) )
top = gbar( gbar(top) )
bot = 1.0 - driver - top
:Yh: = numpy.minimum(driver + top,1.0)
:Yl: = 1.0 - :Yh:
# Mixed EOS (partial pressures method)
:gamma: = :Yh:* gammaH + :Yl: * gammaL
:mw: = 1.0 / ( :Yh: / mwH + :Yl: / mwL )
:R: = 1.0 / :mw:
:cp: = :R: / (1.0 - 1.0/:gamma: )
:cv: = :cp: - :R:
# Energy pill for detonation wave
:p:   = 1.0*driver + .1*(1.-driver)
:rho: = 1.0*(driver+bot) + .125*top
:T: = :p: / (:rho: * :R:)
# :rho: = :p: / ( :R: * :T: )
:u: = 3d()
:v: = 3d()
# Form conservatives from primatives
:rhoYh: = :rho:*:Yh:
:rhoYl: = :rho:*:Yl:
:Et: = :p:/( :gamma: - 1.0 ) + .5*:rho:*(:u:*:u: + :v:*:v:)
:rhou: = :rho:*:u:
:rhov: = :rho:*:v:
# Sound speed and initial time step size
:cs:  = sqrt( :p: / :rho: * :gamma: )
:dt: = dt.courant(:u:,:v:,:w:,:cs:)
"""
if TEST:
    ic += """
    :ml_beta: = :beta:   * 0.0
    :ml_betaX: = :beta: * 0.0
    :ml_betaY: = :beta: * 0.0"""

icDict = {}
icDict['gammaL'] = gammaL
icDict['gammaH'] = gammaH
icDict['mwH']     = mwH
icDict['mwL']     = mwL



# Set the initial conditions
ss.setIC(ic,icDict)

# Write a time loop
time = 0.0
viz = False

# Stopping time of simulation
tt = 5.0

# Variables to write to viz.
wvars = ['Yh','rho','u','v','p','beta','kappa','adiff','mu']

# Start time loop
dump_freq = 2000
CFL = 1.0
dt = ss.variables['dt'].data * CFL*.01

viz_freq = tt / 20.0
viz_dump = viz_freq

def myIO():
    ss.write( wvars )
    ss.plot.figure(1)
    ss.plot.clf()
    ss.plot.contourf('lrho',64 )


if viz:
    myIO()

if TEST: #Load NN model if testing
    import tensorflow as tf
    import tensorflow.keras
    model = tf.keras.models.load_model('%s/my_model'%Model)

data_x = numpy.zeros( ( (ss.nx-(2 * N)),(ss.ny-(2 * N)),2*N+1) ) #Initialize dimensionless x vel data
data_y = numpy.zeros( ( (ss.nx-(2 * N)),(ss.ny-(2 * N)),2*N+1) ) #Initialize dimensionless y vel data


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
                ss.parse(":taudia:    =  (:ml_beta:-2./3.*:mu:) *:div: - :p:")
                ss.parse( ":tauxx:     =  2.0*:mu:*:ux:   + :taudia: " )
                ss.parse( ":tauyy:     =  2.0*:mu:*:vy:   + :taudia: "

    ss.cycle += 1
    time += dt
    dt = min(dt_max, (tt - time)
    dt = min( ss.variables['dt'].data * CFL, dt*1.1)
    dt = min(dt, (tt - time) )

    stab_type = ''
    # Simulation heart-beat
    if dt == ss.variables['dtB'].data:
        stab_type = 'bulk'
    elif dt == ss.variables['dtM'].data:
        stab_type = 'shear'
    elif dt == ss.variables['dtY'].data:
        stab_type = 'diffusion'
    else:
        stab_type = "CFL"

    ss.iprint("Cycle: %5d --- Time: %10.4e --- deltat: %10.4e ( %s )" % (ss.cycle,time,dt,stab_type)  )


    # Constant time
    if viz:
        if time > viz_dump:
            viz_dump += viz_freq
            myIO()

    if (ss.cycle%dump_freq == 0) :
        ss.writeRestart()


rho_data = ss.var('rho').data

if TEST:
    beta_data = ss.var('ml_beta').data
else:
    beta_data = ss.var('beta').data

if not os.path.exists('Data'):
    os.makedirs('Data')

if BASE:
    np.save("Data/triple_point_base_density.npy",rho_data) #Save base density data
    np.save("Data/triple_point_base_av.npy",beta_data) #Save base beta data
elif RESOLVED:
    np.save("Data/triple_point_res_density.npy",rho_data) #Save resolved density data
    np.save("Data/triple_point_res_av.npy",beta_data) #Save resolved density data
if TEST:
    np.save("Data/triple_point_active_density_N%i.npy"%N,rho_data) #Save NN model density data
    np.save("Data/triple_point_active_av_N%i.npy"%N,beta_data) #Save NN model beta data

if viz:
    myIO()
ss.writeRestart()
