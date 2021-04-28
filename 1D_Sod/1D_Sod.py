from pyranda import pyrandaSim, pyrandaBC
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Create training data', action="store_true")
parser.add_argument('--test', help='Test NN model', action="store_true")
parser.add_argument('--res', help='Use resolved simulation', action="store_true")
parser.add_argument('--N', help='Number of points before/after in domain', default=3, type=int)
parser.add_argument('--model', help='Name of model', default=None, type=str)
args = parser.parse_args()

TRAIN = args.train
TEST = args.test
RESOLVED = args.res
N = args.N
Model = args.model

if TEST and Model == None: #Must load model if testing neural network
    raise RuntimeError("Must include model name")

if RESOLVED: #Use 50x points in resolved simulation
    Npts = 10000
else:
    Npts = 200

if (not RESOLVED) and (not TEST):
    BASE = True
else:
    BASE = False

"""Initialize simulation"""
if BASE:
    title = "sod_base"
elif RESOLVED:
    title = "sod_res"
elif Test:
    title = "sod_%s"%Model
xdom = "xdom = (0.0,6.0,%i,periodic=True)"%Npts
pysim = pyrandaSim(title,xdom)
pysim.addPackage( pyrandaBC(pysim) )#Turn on the pyrandaBC package

"""Set equations of motion"""
eom ="""
# Primary Equations of motion here
ddt(:rho:)  =  -ddx(:rho:*:u:)
ddt(:rhou:) =  -ddx(:rhou:*:u: + :p: - :tau:)
ddt(:Et:)   =  -ddx( (:Et: + :p: - :tau:)*:u: )
# Conservative filter of the EoM
:rho:       =  fbar( :rho:  )
:rhou:      =  fbar( :rhou: )
:Et:        =  fbar( :Et:   )
# Update the primatives and enforce the EOS
:u:         =  :rhou: / :rho:
:p:         =  ( :Et: - .5*:rho:*(:u:*:u:) ) * ( :gamma: - 1.0 )
# Artificial bulk viscosity (old school way)
:div:       =  ddx(:u:)
:beta:      =  gbar( ring(:div:) * :rho:) * 7.0e-2
:tau:       =  :beta:*:div:

:cs:  = sqrt( :p: / :rho: * :gamma: )
# Apply constant BCs
bc.extrap(['rho','Et'],['x1'])
bc.const(['u'],['x1','xn'],0.0)
"""
pysim.EOM(eom)


"""Evaluate the initial conditions and then update variables"""
ic = """
:gamma: = 1.4
:Et:  = gbar( where( meshx < pi, 1.0/(:gamma:-1.0) , .1 /(:gamma:-1.0) ) )
:rho: = gbar( where( meshx < pi, 1.0    , .125 ) )
"""
if TEST:
    ic += """
    :ml_beta: = :beta:   * 0.0"""
pysim.setIC(ic)


"""Integrate in time"""
dt = 1.0 / float(pysim.nx) * 0.75
time = 0.0
tt = 1.5

if TEST: #Load NN model if testing
    import tensorflow as tf
    import tensorflow.keras
    model = tf.keras.models.load_model('%s/my_model'%Model)

maxSamples = (pysim.nx * 3 * (pysim.nx - 2*N)) * 500 #RK4 adjustment
ML_data = np.zeros((2*N+2,maxSamples))
ML_data_rev = np.zeros((2*N+2,maxSamples))
ml_cnt = 0

while time < tt:
    if RESOLVED:
        time = pysim.rk4(time,dt)
    else:
        phi = None
        for rkstep in range(5):
            phi = pysim.rk4_step(dt,rkstep,phi)
            if TRAIN:
                beta_tmp = pysim.var('beta').data / pysim.var('rho').data / pysim.var('cs').data / pysim.dx #Remove dimensions
                u_tmp = pysim.var('u').data / pysim.var('cs').data #Remove dimensions
                for i in range(N,pysim.nx-N):
                    ML_data[0,ml_cnt] = beta_tmp[i,0,0] #Get temporary data so not to change pysim.var('beta').data values
                    ML_data[0,ml_cnt+1] = beta_tmp[i,0,0] #Get temporary data so not to change pysim.var('beta').data values
                    vals = range(i-N,i+N+1)
                    vals_rev = vals[::-1]
                    for j in range(1,2*N+1+1):
                        exec("ML_data[%i,ml_cnt] = u_tmp[%i,0,0]"%(j,vals[j-1])) # Gather data for 2N+1 'rho' points
                        exec("ML_data[%i,ml_cnt + 1] = -1.0*u_tmp[%i,0,0]"%(j,vals_rev[j-1])) # Gather data for 2N+1 'rho' points
                    ml_cnt += 2
            if TEST:
                beta_tmp = pysim.var('beta').data / pysim.var('rho').data / pysim.var('cs').data / pysim.dx #Remove dimensions
                u_tmp = pysim.var('u').data / pysim.var('cs').data #Remove dimensions
                data = np.zeros( ( (pysim.nx-(2 * N)),2*N+1) )
                for i in range(N,pysim.nx-N):
                    ML_data[0,ml_cnt] = pysim.var("beta").data[i,0,0] #Clean up
                    vals = [r for r in  range(i-N,i+N+1)]
                    for j in range(1,2*N+1+1): ##Fix to u instead of rho
                        exec("ML_data[%i,ml_cnt] = u_tmp[%i,0,0] / pysim.var('cs').data[%i,0,0]"%(j,vals[j-1],vals[j-1])) # Gather data for 2N+1 'u' points
                    data[i-N,:] = ML_data[1:,ml_cnt]
                    ml_cnt += 1
                beta_tmp[N:-N,0,0] = model.predict( data )[:,0]
                mask = beta_tmp < 0
                beta_tmp[mask] = 0
                rho_tmp = pysim.var('rho').data
                dx = pysim.dx
                cs_tmp = pysim.var('cs').data
                beta_tmp = beta_tmp * rho_tmp * dx * cs_tmp
                pysim.var('ml_beta').data = beta_tmp
                pysim.parse(":tau:  =  :ml_beta: * :div:")
        time += dt


rho_data = pysim.var('rho').data
rho_data = rho_data.flatten()

if TEST:
    beta_data = pysim.var('ml_beta').data
    beta_data = beta_data.flatten()
else:
    beta_data = pysim.var('beta').data
    beta_data = beta_data.flatten()

if not os.path.exists('Data'):
    os.makedirs('Data')

if TRAIN:
    ML_data_trim = ML_data[:,:ml_cnt] #Trim training data, removing extra zero columns
    np.savetxt("Data/N%i_data.txt"%N,ML_data_trim.T,fmt="%.4e",newline="\n",delimiter=" ") #Save training data
if BASE:
    np.save("Data/1D_sod_base_density.npy",rho_data) #Save base density data
    np.save("Data/1D_sod_base_av.npy",beta_data) #Save base beta data
elif RESOLVED:
    x = np.linspace(pysim.meshOptions['x1'][0],pysim.meshOptions['xn'][0],num=200)
    xp = np.linspace(pysim.meshOptions['x1'][0],pysim.meshOptions['xn'][0],num=pysim.nx)
    fp_rho = rho_data
    interpPoints_rho = np.interp(x,xp,fp_rho)
    np.save("Data/1D_sod_res_density.npy",interpPoints_rho) #Save resolved density data
    fp_beta = beta_data
    interpPoints_beta = np.interp(x,xp,fp_beta)
    np.save("Data/1D_sod_res_av.npy",interpPoints_beta) #Save resolved density data
if TEST:
    np.save("Data/1D_sod_active_density_N%i.npy"%N,rho_data) #Save NN model density data
    np.save("Data/1D_sod_active_av_N%i.npy"%N,beta_data) #Save NN model beta data
