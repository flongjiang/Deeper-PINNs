import torch
import random
import scipy.io
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
from utils import *


def seed_torch(seed):
    """ Seed initialization """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# Exact solution boundaries
data = scipy.io.loadmat('./AC.mat')
Exact = data['uu']
Exact_sol = np.real(Exact)
t_sol = data['tt'].flatten()[:,None]
x_sol = data['x'].flatten()[:,None]

N0 = 256
N_b = 256
N_f = 5000

# Definition
dimx = 256*2
dimt = 201
tm = np.linspace(t_sol.min(), t_sol.max(), dimt)[:, None]
xm = np.linspace(x_sol.min(), x_sol.max(), dimx)[:, None]
X, T = np.meshgrid(xm, tm)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
lb = X_star.min(0)
ub = X_star.max(0)
n_blocks = 1

seed_torch(42)

model = PhysicsInformedNN(Exact_sol, x_sol, t_sol, N_b, N0, N_f, lb, ub, n_blocks)

model.train()
np.savetxt('losses.txt', model.d, fmt='%.10f %.10f %.10f')
torch.save(model.dnn.state_dict(), "wb.pt")


model.dnn.load_state_dict(torch.load('./wb.pt' % seed))

# Prediction
Exact = Exact_sol
X, T = np.meshgrid(x_sol, t_sol)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_pred = model.predict(X.flatten()[:,None], T.flatten()[:,None])
U_pred = np.reshape(u_pred, (Exact.shape[1], Exact.shape[0])).T
l2_rel = np.linalg.norm(Exact.flatten() - U_pred.flatten()) / np.linalg.norm(Exact.flatten(), 2)
print('L2:', l2_rel)


fig, ax = plt.subplots(dpi=300)

h = plt.imshow(U_pred, interpolation='nearest', cmap='jet',
            extent=[0.0, 1.0, -1.0, 1.0],
            origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

plt.legend(frameon=False, loc = 'best')

plt.show()


#####################################################
plt.figure(1, figsize=(20, 3),dpi=300)
plt.subplot(1, 3, 1)
h = plt.imshow(Exact, interpolation='nearest', cmap='jet',
            extent=[0.0, 1.0, -1.0, 1.0],
            origin='lower', aspect='auto')

c = plt.colorbar()
plt.xlabel(r'$t$',fontdict = {'fontsize': 14})
plt.ylabel(r'$x$',fontdict = {'fontsize': 14})
plt.title("Exact $u(x,t)$",fontdict = {'fontsize': 14})

plt.subplot(1, 3, 2)
h = plt.imshow(U_pred, interpolation='nearest', cmap='jet',
            extent=[0.0, 1.0, -1.0, 1.0],
            origin='lower', aspect='auto')

c = plt.colorbar()
plt.xlabel(r'$t$',fontdict = {'fontsize': 14})
plt.ylabel(r'$x$',fontdict = {'fontsize': 14})
plt.title("Predicted $u(x,t)$",fontdict = {'fontsize': 14})

plt.subplot(1, 3, 3)
h = plt.imshow(abs(U_pred - Exact), interpolation='nearest', cmap='jet',
            extent=[0.0, 1.0, -1.0, 1.0],
            origin='lower', aspect='auto')

c = plt.colorbar()
plt.xlabel(r'$t$',fontdict = {'fontsize': 14})
plt.ylabel(r'$x$',fontdict = {'fontsize': 14})
plt.title("Absolute error",fontdict = {'fontsize': 14})
# plt.tight_layout()
plt.show()
