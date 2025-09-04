import torch
import random
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import math
from utils import *


def seed_torch(seed):
    """ Seed initialization """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch(42)

# File initializations
file_path = './'
file_name = 'losses.txt'

N0 = 200
N_b = 200
N_f = 5000

nx, nt = (200,200)
x = np.linspace(0, 2*math.pi, nx)
t = np.linspace(0, 1, nt)

xv, tv = np.meshgrid(x,t)

x = np.reshape(xv, (-1,1))
t = np.reshape(tv, (-1,1))

beta = 100
exact_u = np.sin(xv - beta*tv)
Exact_u = np.reshape(exact_u, (-1,1))

X_exact = np.hstack((x, t, Exact_u))

lb = np.array([0.0])
ub = np.array([2*math.pi])
lftb = np.array([0.0])
rb = np.array([1.0])

n_blocks = 2

model = PhysicsInformedNN(N0, N_b, N_f, X_exact, beta, lb, ub, lftb, rb, n_blocks)

model.train()
np.savetxt('losses_%d.txt' % seed, model.d, fmt='%.10f %.10f %.10f')
torch.save(model.dnn.state_dict(), "wb_%d.pt" % seed)

X, T = xv, tv

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact_u.flatten()[:,None]

u_pred = model.predict(X_star[:, 0:1], X_star[:, 1:2])


error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

print('Error u: %e' % (error_u))

U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')


fig_1 = plt.figure(1, figsize=(15, 3.5),dpi=300)
plt.subplot(1, 3, 1)
h = plt.imshow(exact_u.T, interpolation='nearest', cmap='jet',
            extent=[lftb[0], rb[0], lb[0], ub[0]],
            origin='lower', aspect='auto')

c = plt.colorbar()
plt.xlabel(r'$t$',fontdict = {'fontsize': 14})
plt.ylabel(r'$x$',fontdict = {'fontsize': 14})
plt.title("Exact $u(x,t)$",fontdict = {'fontsize': 14})

plt.subplot(1, 3, 2)
h = plt.imshow(U_pred.T, interpolation='nearest', cmap='jet',
            extent=[lftb[0], rb[0], lb[0], ub[0]],
            origin='lower', aspect='auto')

c = plt.colorbar()
plt.xlabel(r'$t$',fontdict = {'fontsize': 14})
plt.ylabel(r'$x$',fontdict = {'fontsize': 14})
plt.title("Predicted $u(x,t)$",fontdict = {'fontsize': 14})

plt.subplot(1, 3, 3)
h = plt.imshow(abs(U_pred - exact_u).T, interpolation='nearest', cmap='jet',
            extent=[lftb[0], rb[0], lb[0], ub[0]],
            origin='lower', aspect='auto')

c = plt.colorbar()
plt.xlabel(r'$t$',fontdict = {'fontsize': 14})
plt.ylabel(r'$x$',fontdict = {'fontsize': 14})
plt.title("Absolute error",fontdict = {'fontsize': 14})
#plt.tight_layout()
plt.show()
