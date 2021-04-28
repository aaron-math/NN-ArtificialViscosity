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

HighRes = np.load('Data/triple_point_res_density.npy') #Already interpolated, 200 Points
ml_data = np.load('Data/triple_point_active_density_N%i.npy'%N) #NN model data
base_data = np.load('Data/triple_point_base_density.npy') #Base model data

Nx1,y1,z1 = HighRes.shape
x2,y2,z2 = SYM.shape

xlim = 7.0
ylim = 3.0


Xfirst = np.zeros((x2,y2,1))
Yfirst = np.zeros((x2,y2,1))
Xfirst_1 = np.zeros((x2,y1,1))
Yfirst_1 = np.zeros((x1,y2,1))


for i in range(y1):
    x = np.linspace(0,xlim,num=x2)
    xp = np.linspace(0,xlim,num=x1)
    fp = HighRes[:,i,0]
    Xfirst_1[:,i,0] = np.interp(x,xp,fp)
for i in range(x1):
    x = np.linspace(0,ylim,num=y2)
    xp = np.linspace(0,ylim,num=y1)
    fp = HighRes[i,:,0]
    Yfirst_1[i,:,0] = np.interp(x,xp,fp)


for i in range(y2):
    x = np.linspace(0,xlim,num=x2)
    xp = np.linspace(0,xlim,num=x1)
    fp = Yfirst_1[:,i,0]
    Yfirst[:,i,0] = np.interp(x,xp,fp)
for i in range(x2):
    x = np.linspace(0,xlim,num=y2)
    xp = np.linspace(0,xlim,num=y1)
    fp = Xfirst_1[i,:,0]
    Xfirst[i,:,0] = np.interp(x,xp,fp)

xx1 = np.linspace(0,xlim,num=x1)
yy1 = np.linspace(0,ylim,num=y1)
xx2 = np.linspace(0,xlim,num=x2)
yy2 = np.linspace(0,ylim,num=y2)

if not os.path.exists('Results'):
    os.makedirs('Results')

plt.clf()
plt.axis('equal')
plt.contourf( xx2,yy2,np.log(base_data[:,:,0].T) ,64 , cmap=cm.jet)
plt.xticks([],[])
plt.yticks([],[])
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
if SAVE:
    plt.savefig("Results/triple_point_base_density.png",dpi=600,bbox_inches='tight',transparent=True)
else:
    plt.show()
plt.clf()

plt.clf()
plt.axis('equal')
plt.contourf( xx2,yy2,np.log(ml_data[:,:,0]) ,64 , cmap=cm.jet)
plt.xticks([],[])
plt.yticks([],[])
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
if SAVE:
    plt.savefig("Results/triple_point_active_density.png",dpi=600,bbox_inches='tight',transparent=True)
else:
    plt.show()
plt.clf()


ml_data_av = np.load("Data/triple_point_active_av_N%i.npy"%N)
base_data_av = np.load("Data/triple_point_base_av.npy")

plt.clf()
plt.axis('equal')
plt.contourf( xx2,yy2,base_data_av[:,:,0],64 , cmap=cm.jet)
plt.xticks([],[])
plt.yticks([],[])
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
if SAVE:
    plt.savefig("Results/triple_point_base_av.png",dpi=600,bbox_inches='tight',transparent=True)
else:
    plt.show()
plt.clf()

plt.clf()
plt.axis('equal')
plt.contourf( xx2,yy2,ml_data_av[:,:,0] ,64 , cmap=cm.jet)
plt.xticks([],[])
plt.yticks([],[])
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
if SAVE:
    plt.savefig("Results/triple_point_active_av.png",dpi=600,bbox_inches='tight',transparent=True)
else:
    plt.show()
plt.clf()

"""Calculate Norms"""
L1 = []
L2 = []
L_inf = []

HighSumX = np.sum(Xfirst[:,:,0])
HighSumY = np.sum(Yfirst[:,:,0])


L1.append(la.norm((base_data[:,:,0]-Xfirst[:,:,0])/HighSumX,ord=1))
L1.append(la.norm((ml_data[:,:,0]-Xfirst[:,:,0])/HighSumX,ord=1))
L2.append(la.norm((base_data[:,:,0]-Xfirst[:,:,0])/HighSumX,ord=2))
L2.append(la.norm((ml_data[:,:,0]-Xfirst[:,:,0])/HighSumX,ord=2))
L_inf.append(la.norm((base_data[:,:,0]-Xfirst[:,:,0])/HighSumX,ord=np.inf))
L_inf.append(la.norm((ml_data[:,:,0]-Xfirst[:,:,0])/HighSumX,ord=np.inf))

dom = [1,2]
ticks = ["AV","NN-AV"]

#plt.plot(dom,L1,linewidth=3,label=r"$L_1$")
plt.plot(dom,L2,linewidth=3,label=r"$L_2$")
plt.plot(dom,L_inf,linewidth=3,label=r"$L_\infty$")
plt.yscale('log')
plt.xticks(dom,ticks,fontsize=12)
plt.title("triple_point Relative Error",fontsize=18)
plt.legend()
plt.ylabel('Relative Error',fontsize=14)
if SAVE:
    plt.savefig("Results/triple_point_error.png",dpi=600,bbox_inches='tight',transparent=True)
else:
    plt.show()
plt.clf()


print('\t Base \t\t NN-AV')
print('L1\t',"{:e}".format(L1[0]),'\t',"{:e}".format(L1[1]))
print('L2\t',"{:e}".format(L2[0]),'\t',"{:e}".format(L2[1]))
print('L_inf\t',"{:e}".format(L_inf[0]),'\t',"{:e}".format(L_inf[1]))

if WRITE:
    with open('Results/triple_point_error.txt','w') as outfile:
        outfile.write('\t Base \t\t NN-AV\n')
        outfile.write('L1\t'+"{:e}".format(L1[0])+'\t'+"{:e}".format(L1[1])+'\n')
        outfile.write('L2\t'+"{:e}".format(L2[0])+'\t'+"{:e}".format(L2[1])+'\n')
        outfile.write('L_inf\t'+"{:e}".format(L_inf[0])+'\t'+"{:e}".format(L_inf[1]))
