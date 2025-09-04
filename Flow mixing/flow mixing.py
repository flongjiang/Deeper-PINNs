import torch
import numpy as np
from pyDOE import lhs
from torch.optim import lr_scheduler
import random
import time
import matplotlib.pyplot as plt
from collections import OrderedDict

def seed_torch(seed):
    """ Seed initialization """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_torch(42)

if torch.cuda.is_available():
    """ Cuda support """
    print('cuda available')
    device = torch.device('cuda')
else:
    print('cuda not avail')
    device = torch.device('cpu')

def tonp(tensor):
    """ Torch to Numpy """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def grad(u, x):
    """ Get grad """
    gradient = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]
    return gradient


class star_block(torch.nn.Module):
    def __init__(self, width):
        super(star_block, self).__init__()
        self.activation = torch.nn.Tanh()
        self.layer1 = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer1.weight)
        self.layer2 = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer2.weight)
       
        self.layer1s = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer1s.weight)
        self.layer2s = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer2s.weight)
    
    def forward(self, inputs):
        
        H = inputs
        x1 = self.activation(self.layer1(H))
        x1s = self.activation(self.layer1s(H))
        x1 = x1*x1s
        
        x2 = self.activation(self.layer2(x1))
        x2s = self.activation(self.layer2s(x1))
        x2 = x2*x2s + H
        
        return x2


class DNN(torch.nn.Module):
    def __init__(self, layers, n_blocks):
        super(DNN, self).__init__()
        self.activation = torch.nn.Tanh()
        self.layer1 = torch.nn.Linear(layers[0], layers[1], bias=True)
        torch.nn.init.xavier_normal_(self.layer1.weight)
        
        layer_list = list()
        
        for i in range(n_blocks):
            layer_list.append(('star_block_%d' % i, star_block(layers[1])))
        layerDict = OrderedDict(layer_list)
        # Deploy layers
        self.features = torch.nn.Sequential(layerDict)
        
        self.out_layer = torch.nn.Linear(layers[1], layers[2])
        torch.nn.init.xavier_normal_(self.out_layer.weight)
        
        
        
    def forward(self, inputs):
        x1 = self.activation(self.layer1(inputs))

        x2 = self.features(x1)
        
        out = self.out_layer(x2)
        return out
    
class Trainer:
    def __init__(self, model, lb, ub, t_max, N_f, N_0,\
                      Exact_sol, X_sol, Y_sol, t_sol, \
                      optimizer, save_every):
        
        self.model = model.to(device)
        
        
        self.lb, self.ub, self.t_max = np.array(lb), np.array(ub), t_max
        self.N_f, self.N_0 = N_f, N_0
        self.x_sol = X_sol
        self.y_sol = Y_sol
        self.t_sol = t_sol
        self.Exact_sol = Exact_sol
        
        
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = model
        self.exec_time = 0
        self.iter = []; self.l2 = [];  self.l = []
        self.loss = None
        self.print_step = 100
    
    def net_r(self, X):
        """ Residual calculation """
        x = torch.tensor(X[:,0:1], requires_grad=True).float().to(device)
        y = torch.tensor(X[:,1:2], requires_grad=True).float().to(device)
        t = torch.tensor(X[:,2:3], requires_grad=True).float().to(device)
        
        r = torch.sqrt(x**2 + y**2)
        vt = torch.tanh(r)*(1/torch.cosh(r))**2 
        vt_max = torch.max(vt)
        a = -vt/vt_max*(y/r)
        b = vt/vt_max*(x/r)
        
        u = self.model(torch.cat([x, y, t], dim=1)) 
        u_x = grad(u, x)
        u_y = grad(u, y)
        u_t = grad(u, t)
        
        res = u_t + a*u_x + b*u_y
        return res

    def _run_epoch(self):
        self.optimizer.zero_grad()
        
        Xf = self.lb + (self.ub - self.lb)*lhs(2, self.N_f)
        tf = self.t_max*lhs(1, self.N_f)
        X_f = np.concatenate((Xf, tf), axis=1)
        
        X0 = self.lb + (self.ub - self.lb)*lhs(2, self.N_0)
        t0 = 0*lhs(1, self.N_0)
        X_0 = np.concatenate((X0, t0), axis=1)
        X_ic = torch.tensor(X_0, requires_grad=True).float().to(device)
        self.u_ic = self.model(X_ic)
        
        self.loss_ic = torch.mean((-torch.tanh(X_ic[:,1:2]/2) - self.u_ic)**2)
        
        self.res = self.net_r(X_f) # training points
        self.loss_r = torch.mean(self.res**2)
        
        self.loss = self.loss_r + self.loss_ic
        
        self.loss.backward()
        
        self.optimizer.step()

    def _save_checkpoint(self, epoch):
        if (epoch+1) % 5000 == 0:
            PATH = "checkpoint_%d.pt" % (epoch)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.loss,
                        }, PATH)

    def train(self, max_epochs, scheduler, step_size):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('all trainable parameters: %d' % params)
        for epoch in range(max_epochs):
            start_time = time.time()
            self._run_epoch()
            end_time = time.time()
            self.exec_time = end_time - start_time
            if (epoch+1) % step_size == 0:
                scheduler.step()
                
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
                
            if epoch % self.print_step == 0:
                
                u_p = self.predict(np.concatenate((self.x_sol, self.y_sol, self.t_sol), axis=1))
                
                # Relative error
                error = np.linalg.norm(u_p - self.Exact_sol, 2) / np.linalg.norm(self.Exact_sol, 2)
                
                print('Iter %d, loss: %.3e, L2: %.3e, t/iter: %.1e' % 
                      (epoch, self.loss.item(), error, self.exec_time))
                
                self.iter.append(epoch)
                self.l2.append(error)
                self.l.append(self.loss.item())
                
                # Stack them into a 2D array.
                self.d = np.column_stack((np.array(self.iter), np.array(self.l2), np.array(self.l)))
                np.savetxt('losses.txt', self.d, fmt='%d %.10f %.10f')
                
                
                
    def predict(self, X_p):
        with torch.no_grad():
            x = torch.tensor(X_p).float().to(device)
            self.model.eval()
            u= self.model(x)
            u = tonp(u)
        return u
            

def load_train_objs(layers, n_blocks):
    model = DNN(layers, n_blocks)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)
    total_epochs = 60000
    step_size = 1000
    save_every = 1000
    return model, optimizer, scheduler, step_size, total_epochs, save_every



hidden = 128
layers = [3] + [hidden] + [1]
n_blocks = 2
N_f = 20000
N_0 = 4000

lb = [-4.0, -4.0]
ub = [4.0, 4.0]
t_max = 4

nx, ny = (100,100)
x = np.linspace(lb[0], ub[0], nx)
y = np.linspace(lb[1], ub[1], ny)
xv, yv = np.meshgrid(x,y)
x_sol = np.reshape(xv, (-1,1))
y_sol = np.reshape(yv, (-1,1))

nt = 50
t = np.linspace(0, t_max, nt)

X_sol = np.tile(x_sol, nt).T.flatten()[:,None]
Y_sol = np.tile(y_sol, nt).T.flatten()[:,None]
t_sol = np.repeat(t, nx*ny)[:,None]

r = np.sqrt(X_sol**2 + Y_sol**2)
vt = np.tanh(r)*(1/np.cosh(r))**2 
vt_max = np.max(vt)
w = (1/r)*(vt/vt_max)
Exact_sol = -np.tanh((Y_sol/2)*np.cos(w*t_sol) - (X_sol/2)*np.sin(w*t_sol))
#
    
    
model, optimizer, scheduler, step_size, total_epochs, save_every = load_train_objs(layers, n_blocks)

trainer = Trainer(model, lb, ub, t_max, N_f, N_0,\
                  Exact_sol, X_sol, Y_sol, t_sol, \
                  optimizer, save_every)
    
trainer.train(total_epochs, scheduler, step_size)

torch.save(model.state_dict(), "wb_%d.pt")
np.savetxt('losses.txt', trainer.d, fmt='%d %.10f %.10f')

model.load_state_dict(torch.load('./wb.pt'))

u_pred = trainer.predict(np.concatenate((X_sol,Y_sol, t_sol), axis=1))
error_u = np.linalg.norm(u_pred - Exact_sol, 2) / np.linalg.norm(Exact_sol, 2)
print('Error u: %e' % (error_u))

###################### 0s ############################################
nx, ny = (500,500)
x = np.linspace(lb[0], ub[0], nx)
y = np.linspace(lb[1], ub[1], ny)
xv, yv = np.meshgrid(x,y)
x_p = np.reshape(xv, (-1,1))
y_p = np.reshape(yv, (-1,1))

u_pred = trainer.predict(np.concatenate((x_p, y_p, 0*x_p), axis=1))
U_pred_0 = np.reshape(u_pred, (nx,ny))

r = np.sqrt(x_p**2 + y_p**2)
vt = np.tanh(r)*(1/np.cosh(r))**2 
vt_max = np.max(vt)
w = (1/r)*(vt/vt_max)
exact_0 = -np.tanh((y_p/2)*np.cos(w*(0)) - (x_p/2)*np.sin(w*(0)))
Exact_0 = np.reshape(exact_0, (nx,ny))

plt.figure(1, figsize=(19, 4),dpi=300)
plt.subplot(1, 3, 1)
h = plt.imshow(Exact_0, interpolation='nearest', cmap='jet',
            extent=[-4.0, 4.0, -4.0, 4.0],
            origin='lower', aspect='auto')

c = plt.colorbar()
plt.xlabel(r'$x$',fontdict = {'fontsize': 14})
plt.ylabel(r'$y$',fontdict = {'fontsize': 14})
plt.title("Exact $u(x,y)$",fontdict = {'fontsize': 14})

plt.subplot(1, 3, 2)
h = plt.imshow(U_pred_0, interpolation='nearest', cmap='jet',
            extent=[-4.0, 4.0, -4.0, 4.0],
            origin='lower', aspect='auto')

c = plt.colorbar()
plt.xlabel(r'$x$',fontdict = {'fontsize': 14})
plt.ylabel(r'$y$',fontdict = {'fontsize': 14})
plt.title("Predicted $u(x,y)$",fontdict = {'fontsize': 14})

plt.subplot(1, 3, 3)
h = plt.imshow(abs(U_pred_0 - Exact_0), interpolation='nearest', cmap='jet',
            extent=[-4.0, 4.0, -4.0, 4.0],
            origin='lower', aspect='auto')

c = plt.colorbar()
plt.xlabel(r'$x$',fontdict = {'fontsize': 14})
plt.ylabel(r'$y$',fontdict = {'fontsize': 14})
plt.title("Absolute error",fontdict = {'fontsize': 14})
# plt.tight_layout()
plt.show()

###################### 5s ############################################
u_pred = trainer.predict(np.concatenate((x_p, y_p, 0*x_p + 5), axis=1))
X_star = np.hstack((x_p.flatten()[:,None], y_p.flatten()[:,None]))
U_pred_5 = np.reshape(u_pred, (nx,ny))

r = np.sqrt(x_p**2 + y_p**2)
vt = np.tanh(r)*(1/np.cosh(r))**2 
vt_max = np.max(vt)
w = (1/r)*(vt/vt_max)
exact_5 = -np.tanh((y_p/2)*np.cos(w*(5)) - (x_p/2)*np.sin(w*(5)))
Exact_5 = np.reshape(exact_5, (nx,ny))

plt.figure(1, figsize=(19, 4),dpi=300)
plt.subplot(1, 3, 1)
h = plt.imshow(Exact_5, interpolation='nearest', cmap='jet',
            extent=[-4.0, 4.0, -4.0, 4.0],
            origin='lower', aspect='auto')

c = plt.colorbar()
plt.xlabel(r'$x$',fontdict = {'fontsize': 14})
plt.ylabel(r'$y$',fontdict = {'fontsize': 14})
plt.title("Exact $u(x,y)$",fontdict = {'fontsize': 14})

plt.subplot(1, 3, 2)
h = plt.imshow(U_pred_5, interpolation='nearest', cmap='jet',
            extent=[-4.0, 4.0, -4.0, 4.0],
            origin='lower', aspect='auto')

c = plt.colorbar()
plt.xlabel(r'$x$',fontdict = {'fontsize': 14})
plt.ylabel(r'$y$',fontdict = {'fontsize': 14})
plt.title("Predicted $u(x,y)$",fontdict = {'fontsize': 14})

plt.subplot(1, 3, 3)
h = plt.imshow(abs(U_pred_5 - Exact_5), interpolation='nearest', cmap='jet',
            extent=[-4.0, 4.0, -4.0, 4.0],
            origin='lower', aspect='auto')

c = plt.colorbar()
plt.xlabel(r'$x$',fontdict = {'fontsize': 14})
plt.ylabel(r'$y$',fontdict = {'fontsize': 14})
plt.title("Absolute error",fontdict = {'fontsize': 14})
# plt.tight_layout()
plt.show()

###################### 10s ############################################
u_pred = trainer.predict(np.concatenate((x_p, y_p, 0*x_p + 10), axis=1))
X_star = np.hstack((x_p.flatten()[:,None], y_p.flatten()[:,None]))
U_pred_10 = np.reshape(u_pred, (nx,ny))

r = np.sqrt(x_p**2 + y_p**2)
vt = np.tanh(r)*(1/np.cosh(r))**2 
vt_max = np.max(vt)
w = (1/r)*(vt/vt_max)
exact_10 = -np.tanh((y_p/2)*np.cos(w*(10)) - (x_p/2)*np.sin(w*(10)))
Exact_10 = np.reshape(exact_10, (nx,ny))

plt.figure(1, figsize=(19, 4),dpi=300)
plt.subplot(1, 3, 1)
h = plt.imshow(Exact_10, interpolation='nearest', cmap='jet',
            extent=[-4.0, 4.0, -4.0, 4.0],
            origin='lower', aspect='auto')

c = plt.colorbar()
plt.xlabel(r'$x$',fontdict = {'fontsize': 14})
plt.ylabel(r'$y$',fontdict = {'fontsize': 14})
plt.title("Exact $u(x,y)$",fontdict = {'fontsize': 14})

plt.subplot(1, 3, 2)
h = plt.imshow(U_pred_10, interpolation='nearest', cmap='jet',
            extent=[-4.0, 4.0, -4.0, 4.0],
            origin='lower', aspect='auto')

c = plt.colorbar()
plt.xlabel(r'$x$',fontdict = {'fontsize': 14})
plt.ylabel(r'$y$',fontdict = {'fontsize': 14})
plt.title("Predicted $u(x,y)$",fontdict = {'fontsize': 14})

plt.subplot(1, 3, 3)
h = plt.imshow(abs(U_pred_10 - Exact_10), interpolation='nearest', cmap='jet',
            extent=[-4.0, 4.0, -4.0, 4.0],
            origin='lower', aspect='auto')

c = plt.colorbar()
plt.xlabel(r'$x$',fontdict = {'fontsize': 14})
plt.ylabel(r'$y$',fontdict = {'fontsize': 14})
plt.title("Absolute error",fontdict = {'fontsize': 14})
# plt.tight_layout()
plt.show()

