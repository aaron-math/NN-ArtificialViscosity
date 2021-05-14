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

TRAIN = args.train #If training mode
TEST = args.test #If testing mode
RESOLVED = args.res #If running a high-res simulation
N = args.N #2N+1 points
Model = args.model #Name of model if using

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
#Title of simulation
if BASE:
    title = "burger_base"
elif RESOLVED:
    title = "burger_res"
elif TEST:
    title = "burger_%s"%Model
xdom = "xdom = (0.0,1.0,%i,periodic=True)"%Npts #Domain of simulation
pysim = pyrandaSim(title,xdom) #Begin simulation
pysim.addPackage( pyrandaBC(pysim) ) #Turn on the pyrandaBC package

"""Set equations of motion"""
eom = """
ddt(:u:) = - :u: * ddx(:u:) + :nu: * ddx(ddx(:u:)) # Viscous Burgers equation
:u: = fbar(:u:)                                    # Conservative filter to prevent high-frequency ringing
:div: = ddx( :u: )                                 # Velocity gradient
:nu: = .1 * gbar( ring( :div: ) )                  # Artificial viscosity model
"""
pysim.EOM(eom) #Save eom

"""Evaluate the initial conditions and then update variables"""
ic = """
:u: = exp(-(meshx-.5)**2/(.1**2))"""
if TEST:
    ic += """
    :ml_nu: = :nu:   * 0.0"""
pysim.setIC(ic) #Set initial conditions

"""Integrate in time"""
dt_max = .001 #Max dt
if RESOLVED:
    dt_max = 0.00001 #Smaller timestep needed to avoid runtime errors
time = 0.0

if TEST: #Load NN model if testing
    import tensorflow as tf
    import tensorflow.keras
    model = tf.keras.models.load_model('%s/my_model'%Model)

maxSamples = (400 * (pysim.nx - 2*N)) * 5 #Arbitrary size large enough to collect data samples
ML_data = np.zeros((2*N+2,maxSamples)) #Data used to collect for training
ml_cnt = 0
dt = dt_max

while time < 0.3:
    if RESOLVED:
        time = pysim.rk4(time,dt) #RK4 calculation
    else:
        phi = None
        for rkstep in range(5):
            phi = pysim.rk4_step(dt,rkstep,phi) #Partial step of RK4
            if TRAIN:
                for i in range(N,pysim.nx-N):
                    ML_data[0,ml_cnt] = pysim.var("nu").data[i,0,0] #Save nu values
                    vals = [r for r in  range(i-N,i+N+1)] #get N points before and after
                    for j in range(1,2*N+1+1):
                        exec("ML_data[%i,ml_cnt] = pysim.var('u').data[%i,0,0]"%(j,vals[j-1])) # Gather data for 2N+1 'u' points
                    ml_cnt += 1
            if TEST:
                ml_nu = pysim.var('ml_nu').data #Get ml variable data
                data = np.zeros( ( (pysim.nx-(2 * N)),2*N+1) )
                for i in range(N,pysim.nx-N):
                    ML_data[0,ml_cnt] = pysim.var("nu").data[i,0,0] #Save nu values
                    vals = [r for r in  range(i-N,i+N+1)] #get N points before and after
                    for j in range(1,(2*N+1)+1):
                        exec("ML_data[%i,ml_cnt] = pysim.var('u').data[%i,0,0]"%(j,vals[j-1])) # Gather data for 2N+1 'u' points
                    data[i-N,:] = ML_data[1:,ml_cnt]
                    ml_cnt += 1
                ml_nu[N:-N,0,0] = model.predict( data )[:,0]
                #Adjust min(ml_nu) to 0 to approximate correct bulk viscosity away from shock
                ml_nu[N:-N,0,0] = ml_nu[N:-N,0,0] - np.min(ml_nu[N:-N,0,0])
                pysim.var('ml_nu').data = ml_nu #Save variable data to simulation
                pysim.parse(":nu:  =  :ml_nu:") #Plug ml_nu values back into simulation
        pysim.cycle += 1 #Increment cycle
        time += dt
        dt = min(dt_max,0.3-time)

u_data = pysim.var('u').data #Get velocity data
u_data = u_data.flatten()
if TEST:
    nu_data = pysim.var('ml_nu').data #Get NN-AV variable data
    nu_data = nu_data.flatten()
else:
    nu_data = pysim.var('nu').data #Get OP-AV variable data
    nu_data = nu_data.flatten()

if not os.path.exists('Data'): #Make data directory if not there
    os.makedirs('Data')

if TRAIN:
    ML_data_trim = ML_data[:,:ml_cnt] #Trim training data, removing extra zero columns
    np.savetxt("Data/N%i_data.txt"%N,ML_data_trim.T,fmt="%.4e",newline="\n",delimiter=" ") #Save training data
if BASE:
    np.save("Data/burgers_base_velocity.npy",u_data) #Save base velocity data
    np.save("Data/burgers_base_nu.npy",nu_data) #Save base nu data
elif RESOLVED:
    x = np.linspace(0,1,num=200)
    xp = np.linspace(0,1,num=pysim.nx)
    fp_u = u_data
    interpPoints_u = np.interp(x,xp,fp_u)
    np.save("Data/burgers_res_velocity.npy",interpPoints_u) #Save resolved velocity data
    fp_nu = nu_data
    interpPoints_nu = np.interp(x,xp,fp_nu)
    np.save("Data/burgers_res_nu.npy",interpPoints_nu) #Save resolved velocity data
if TEST:
    np.save("Data/burgers_active_velocity_N%i.npy"%N,u_data) #Save NN model velocity data
    np.save("Data/burgers_active_nu_N%i.npy"%N,nu_data) #Save NN model nu data
