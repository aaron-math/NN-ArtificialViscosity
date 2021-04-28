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
    Npts = 10000
else:
    Npts = 200

if (not RESOLVED) and (not TEST):
    BASE = True
else:
    BASE = False

"""Initialize simulation"""
if BASE:
    title = "shu_base"
elif RESOLVED:
    title = "shu_res"
elif Test:
    title = "shu_%s"%Model

## Define a mesh

def getShu(Npts,viz=False):

    L = 10.0
    Lp = L * (Npts-1.0) / Npts
    imesh = "xdom = (-Lp/2.0, Lp/2.0, Npts)"
    imesh = imesh.replace('Lp',str(Lp))
    imesh = imesh.replace('Npts',str(Npts))

    # Initialize a simulation object on a mesh
    pysim = pyrandaSim('sod',imesh)
    pysim.addPackage( pyrandaBC(pysim) )
    pysim.addPackage( pyrandaTimestep(pysim) )

    # Define the equations of motion
    from equation_library import euler_1d
    eom = euler_1d
    eom += """
# Apply constant BCs
bc.const(['u'],['xn'],0.0)
bc.const(['u'],['x1'],2.629369)
bc.const(['rho'],['x1'],3.857143)
bc.const(['p'],['x1'],10.33333)
bc.const(['Et'],['x1'],39.166585)
:rhou: = :rho:*:u:
:cs:  = sqrt( :p: / :rho: * :gamma: )
:dt: = dt.courant(:u:,0.0,0.0,:cs:)
:dt: = numpy.minimum(:dt:,0.2 * dt.diff(:beta:,:rho:))
"""

    # Add the EOM to the solver
    pysim.EOM(eom)


    # Initial conditions Shu-Osher test problem
    ic = """
    :gamma: = 1.4
    eps = 2.0e-1
    :tmp: = (meshx+4.0)/%s
    :dum: = tanh(:tmp:)
    :dum: = (:dum:+1.0)/2.0
    :rho: = 3.857143 + :dum:*(1.0+eps*sin(5.0*meshx) - 3.857143)
    :u: = 2.629369*(1.0-:dum:)
    :p: = 10.33333 + :dum:*(1.0-10.33333)
    :rhou: = :rho: * :u:
    :Et:  = :p:/(:gamma: -1.0) + .5*:rho:*:u:*:u:
    """
    if TEST:
        ic += """
        :ml_beta: = :beta:   * 0.0"""

    # Set the initial conditions
    pysim.setIC(ic % pysim.dx)
    #model = tf.keras.models.load_model('DimLess_Model/my_model')
    if TEST: #Load NN model if testing
        import tensorflow as tf
        import tensorflow.keras
        model = tf.keras.models.load_model('%s/my_model'%Model)


    # Write a time loop
    time = 0.0

    # Approx a max dt and stopping time
    CFL = 0.5

    dt = pysim.variables['dt'].data * CFL * .01

    # Mesh for viz on master
    x = pysim.mesh.coords[0].data
    xx =  pysim.PyMPI.zbar( x )

    # Start time loop
    cnt = 1
    viz_freq = 50
    pvar = 'rho'
    #viz = True
    N = 3
    maxSamples = (pysim.nx * 100 * (pysim.nx - 2*N)) * 5 #RK4 adjustment
    ML_data = np.zeros((2*N+2,maxSamples))
    ml_cnt = 0
    tt = 1.8
    while tt > time:
        if RESOLVED:
            time = pysim.rk4(time,dt)
        else:
            # Update the EOM and get next dt
            phi = None
            #time = pysim.rk4(time,dt)
            dt = min( dt*1.1, pysim.variables['dt'].data * CFL )
            dt = min(dt, (tt - time) )
            for rkstep in range(5):
                phi = pysim.rk4_step(dt,rkstep,phi)
                if TEST:
                    beta_tmp = pysim.var('beta').data / pysim.var('rho').data / pysim.var('cs').data / pysim.dx #Remove dimensions
                    u_tmp = pysim.var('u').data / pysim.var('cs').data #Remove dimensions
                    #ml_beta = pysim.var('ml_beta').data
                    data = np.zeros( ( (pysim.nx-(2 * N)),2*N+1) )
                    for i in range(N,pysim.nx-N):
                        ML_data[0,ml_cnt] = pysim.var("beta").data[i,0,0] #Clean up
                        vals = [r for r in  range(i-N,i+N+1)]
                        for j in range(1,2*N+1+1): ##Fix to u instead of rho
                            exec("ML_data[%i,ml_cnt] = u_tmp[%i,0,0] / pysim.var('cs').data[%i,0,0]"%(j,vals[j-1],vals[j-1])) # Gather data for 2N+1 'u' points
                        data[i-N,:] = ML_data[1:,ml_cnt]
                        ### data = data / v_s
                        ml_cnt += 1
                    beta_tmp[N:-N,0,0] = model.predict( data )[:,0]
                    mask = beta_tmp < 0
                    beta_tmp[mask] = 0
                    rho_tmp = pysim.var('rho').data
                    dx = pysim.dx
                    cs_tmp = pysim.var('cs').data
                    beta_tmp = beta_tmp * rho_tmp * dx * cs_tmp
                    pysim.var('ml_beta').data = beta_tmp
                    #pysim.parse(":beta:  =  :ml_beta:")
                    pysim.parse(":tau:  =  :ml_beta: * :div:")
            time += dt

            # Print some output
            pysim.iprint("%s -- %s" % (cnt,time)  )
            cnt += 1
            v = pysim.PyMPI.zbar( pysim.variables[pvar].data )
            if viz:
                if (pysim.PyMPI.master and (cnt%viz_freq == 0)) and True:
                    #raw_input('Poop')
                    plt.figure(1)
                    plt.clf()
                    plt.plot(xx[:,0],v[:,0] ,'k.-')
                    plt.title(pvar)
                    plt.pause(.001)

    return [pysim,xx[:,0],v[:,0]]


[pysim,x,rho] = getShu(Npts)

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

if BASE:
    np.save("Data/shu_base_density.npy",rho_data) #Save base density data
    np.save("Data/shu_base_av.npy",beta_data) #Save base beta data
elif RESOLVED:
    x = np.linspace(pysim.meshOptions['x1'][0],pysim.meshOptions['xn'][0],num=200)
    xp = np.linspace(pysim.meshOptions['x1'][0],pysim.meshOptions['xn'][0],num=pysim.nx)
    fp_rho = rho_data
    interpPoints_rho = np.interp(x,xp,fp_rho)
    np.save("Data/shu_res_density.npy",interpPoints_rho) #Save resolved density data
    fp_beta = beta_data
    interpPoints_beta = np.interp(x,xp,fp_beta)
    np.save("Data/shu_res_av.npy",interpPoints_beta) #Save resolved density data
if TEST:
    np.save("Data/shu_active_density_N%i.npy"%N,rho_data) #Save NN model density data
    np.save("Data/shu_active_av_N%i.npy"%N,beta_data) #Save NN model beta data
