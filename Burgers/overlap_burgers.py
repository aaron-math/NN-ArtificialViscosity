import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--N', help='Number of points before/after in domain', default=3, type=int)
parser.add_argument('-s','--save', help='Save plots',action='store_true')
args = parser.parse_args()

N = args.N
SAVE = args.SAVE

nuActive = np.load("Data/nuActive_adjust.npy")
nuBase = np.load("Data/nuBase.npy")
velActive = np.load('Data/burgers_active_velocity_N%i.npy'%N)
velBase = np.load("Data/burgers_base_velocity.npy")

dom = np.linspace(0.0,1.0,200)

fig, ax1 = plt.subplots()

ax1.set_xlabel(r'$x$')
ax1.set_ylabel('Velocity',color='red')
ax1.plot(dom, velActive, '--',color='green',linewidth=3,label="NN-AV")
ax1.plot(dom, velBase, color='red',label="AV Operator")
ax1.tick_params(axis='y')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Bulk Viscosity',color='blue')
ax2.plot(dom, nuActive, '--',color='orange',label="NN-AV")
ax2.plot(dom, nuBase, color='blue',label="AV Operator")
ax2.tick_params(axis='y')
ax2.legend(loc=6)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
if SAVE:
    plt.savefig("/Users/aaronlarsen/Desktop/LLNL/Final_Presentations/Data/burger_overlap_adjust.png",dpi=640,bbox_inches='tight',transparent=True)
else:
    plt.show()
