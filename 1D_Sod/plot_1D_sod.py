import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--N', help='Number of points before/after in domain', default=3, type=int)
parser.add_argument('-w','--write', help='Write norms to file',action='store_true')
parser.add_argument('-s','--save', help='Save plots',action='store_true')
args = parser.parse_args()

N = args.N
WRITE = args.write
SAVE = args.save

HighRes = np.load('Data/sod_res_density.npy') #Already interpolated, 200 Points
ml_data = np.load('sod_active_density_N%i.npy'%N) #NN model data
base_data = np.load('sod_base_density.npy') #Base model data

NumVals = len(HighRes)

dom = np.linspace(0,6,200)
plt.plot(dom,HighRes,linewidth=3,label="Resolved Solution")
plt.plot(dom,ml_data,'--',linewidth=2,label="NN-AV")
plt.plot(dom,short,'--',linewidth=2,label="OP-AV")
plt.legend()
plt.title(r"1D Sod Shock Tube at $t=1.5$")
plt.xlabel(r"$x$")
plt.ylabel("Density")

if not os.path.exists('Results'):
    os.makedirs('Results')

if SAVE:
    plt.savefig("Results/sod_density.png",dpi=600,bbox_inches='tight',transparent=True)
else:
    plt.show()
plt.clf()

ml_data_av = np.load("Data/sod_active_av_N%i.npy"%N)
base_data_av = np.load("Data/sod_base_av.npy")

dom = np.linspace(0,6,200)
plt.plot(dom,ml_data_av,linewidth=2,label="NN-AV")
plt.plot(dom,base_data_av,linewidth=2,label="OP-AV")
plt.legend()
plt.xlabel(r"$x$")
plt.ylabel("Artificial Viscosity")

if SAVE:
    plt.savefig("Results/sod_av.png",dpi=600,bbox_inches='tight',transparent=True)
else:
    plt.show()
plt.clf()


"""Calculate Norms"""
L1 = []
L2 = []
L_inf = []

HighSum = np.sum(HighRes)

dom = [1,2]
ticks = ["AV","NN-AV"]

"""L1"""
L1.append(la.norm((HighRes-base_data)/HighSum,ord=1))
L1.append(la.norm((HighRes-ml_data)/HighSum,ord=1))
"""L2"""
L2.append(la.norm((HighRes-base_data)/HighSum,ord=2))
L2.append(la.norm((HighRes-ml_data)/HighSum,ord=2))
"""L_inf"""
L_inf.append(la.norm((HighRes-base_data)/HighSum,ord=np.inf))
L_inf.append(la.norm((HighRes-ml_data)/HighSum,ord=np.inf))

plt.plot(dom,L1,label=r"$L_1$")
plt.plot(dom,L2,label=r"$L_2$")
plt.plot(dom,L_inf,label=r"$L_\infty$")
plt.yscale('log')
plt.xticks(dom,ticks)
plt.title("1D Sod Shock Tube Relative Error")plt.legend()
plt.ylabel('Relative Error')
if SAVE:
    plt.savefig("Results/sod_error.png",dpi=600,bbox_inches='tight',transparent=True)
else:
    plt.show()
plt.clf()

print('\t Base \t\t NN-AV \t\t RK4')
print('L1\t',"{:e}".format(L1[0]),'\t',"{:e}".format(L1[1]),'\t',"{:e}".format(L1[2]))
print('L2\t',"{:e}".format(L2[0]),'\t',"{:e}".format(L2[1]),'\t',"{:e}".format(L2[2]))
print('L_inf\t',"{:e}".format(L_inf[0]),'\t',"{:e}".format(L_inf[1]),'\t',"{:e}".format(L_inf[2]))

if WRITE:
    with open('Results/sod_error.txt','w') as outfile:
        outfile.write('\t Base \t\t NN-AV\n')
        outfile.write('L1\t'+"{:e}".format(L1[0])+'\t'+"{:e}".format(L1[1])+'\n')
        outfile.write('L2\t'+"{:e}".format(L2[0])+'\t'+"{:e}".format(L2[1])+'\n')
        outfile.write('L_inf\t'+"{:e}".format(L_inf[0])+'\t'+"{:e}".format(L_inf[1]))
