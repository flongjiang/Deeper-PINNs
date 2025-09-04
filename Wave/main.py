import torch
import random
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
from plotting import newfig
from utils import *

def seed_torch(seed):
    """ Seed initialization """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch(42)

N0 = 400
N_b = 400
N_f = 5000

nx, nt = (200,200)
x = np.linspace(0, ub[0], nx)
t = np.linspace(0, rb[0], nt)

xv, tv = np.meshgrid(x,t)

x = np.reshape(xv, (-1,1))
t = np.reshape(tv, (-1,1))

c = 1
Exact_u = np.sin(np.pi*xv)*np.cos(c*np.pi*tv)
Exact_u = np.reshape(Exact_u, (-1,1))

X_exact = np.hstack((x, t, Exact_u))

n_blocks = 1
    
model = PhysicsInformedNN(N0, N_b, N_f, X_exact, c, n_blocks)

model.train()
np.savetxt('losses.txt', model.d, fmt='%.10f %.10f %.10f')
torch.save(model.dnn.state_dict(), "wb.pt")

X, T = xv, tv

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact_u.flatten()[:,None]

u_pred = model.predict(X_star[:, 0:1], X_star[:, 1:2])


error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

print('Error u: %e' % (error_u))

U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

fig, ax = newfig(1.3, 1.0)
ax.axis('off')
plt.figure(dpi=300)

fig, ax = plt.subplots(dpi=300)

ec = plt.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
            extent=[lftb[0], rb[0],lb[0], ub[0]],
            origin='lower', aspect='auto')


ax.autoscale_view()
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
cbar = plt.colorbar(ec)
cbar.set_label('$u(x,t)$')
plt.title("Predicted $u(x,t)$",fontdict = {'fontsize': 14})
plt.show()
