import os
import sys
import time
import torch
import random
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from pyDOE import lhs
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from collections import OrderedDict
from functools import partial
import math
from plotting import newfig

mpl.rcParams.update(mpl.rcParamsDefault)
np.set_printoptions(threshold=sys.maxsize)
plt.rcParams['figure.max_open_warning'] = 4

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

def input_encoding(x, t):
    
    Px = 2.0 * np.pi
    wx = 2.0 * np.pi / Px
    wx = torch.tensor(wx, dtype=torch.float32).to(device)
    
    out = torch.cat((t, torch.cos(wx * x), torch.sin(wx * x)), 1)
    return out

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
        
        self.layer1s = torch.nn.Linear(layers[0], layers[1], bias=True)
        torch.nn.init.xavier_normal_(self.layer1s.weight)
        
        layer_list = list()
        
        for i in range(n_blocks):
            layer_list.append(('star_block_%d' % i, star_block(layers[1])))
        layerDict = OrderedDict(layer_list)
        # Deploy layers
        self.features = torch.nn.Sequential(layerDict)
        
        self.out_layer = torch.nn.Linear(layers[1], layers[2])
        torch.nn.init.xavier_normal_(self.out_layer.weight)
        
    def forward(self, inputs):
        H = inputs
        H = input_encoding(inputs[:,0:1], inputs[:,1:2])
        
        x1 = self.activation(self.layer1(H))
        x1s = self.activation(self.layer1s(H))
        x1 = x1*x1s
        
        x2 = self.features(x1)
        
        out = self.out_layer(x2)
        return out
    
class PhysicsInformedNN():         
    """ PINN Class """
    
    def __init__(self, N0, N_b, N_f, X_exact, beta, lb, ub, lftb, rb, n_blocks):
        
        # Initialization
        self.iter = 0
        self.exec_time = 0
        self.print_step = 100
        self.beta = beta
        self.epochs = 120000
        self.it = []; self.l2 = []; self.l = []
        self.loss = None
        
        self.lb = lb 
        self.ub = ub 
        self.lftb = lftb 
        self.rb = rb
        
        self.N0 = N0
        self.N_b = N_b
        self.N_f = N_f
        # Data
        
        self.x_tst = torch.tensor(X_exact[:, 0:1], requires_grad=True).float().to(device) # test points
        self.t_tst = torch.tensor(X_exact[:, 1:2], requires_grad=True).float().to(device)
        self.Exact = X_exact[:, 2:3]
        
        self.dnn = DNN(layers, n_blocks).to(device)
        
        # Optimizer (1st ord)
        # self.optimizer =torch.optim.SGD(self.dnn.parameters(), lr=0.001)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3, betas=(0.99, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, verbose=True)
        self.step_size = 3000#2500
        
    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_r(self, x, t):
        """ Residual calculation """
        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_x = grad(u, x)
        f_u = u_t + self.beta*u_x
        return f_u
    
    
    def loss_func(self):
        """ Loss function """
        x_0 = 2*math.pi*lhs(1, self.N0)
        self.x_0 = torch.tensor(x_0, requires_grad=True).float().to(device)
        self.t_0 = torch.tensor(np.zeros_like(x_0), requires_grad=True).float().to(device)
        
        t_b = lhs(1, self.N_b)
        self.x_lb = torch.tensor(np.zeros_like(t_b) + self.lb[0], requires_grad=True).float().to(device)
        self.t_lb = torch.tensor(t_b, requires_grad=True).float().to(device)
        self.x_ub = torch.tensor(np.zeros_like(t_b) + self.ub[0], requires_grad=True).float().to(device)
        self.t_ub = torch.tensor(t_b, requires_grad=True).float().to(device)
        
        x_f = self.lb + (self.ub-self.lb)*lhs(1, self.N_f)
        t_f = self.lftb + (self.rb - self.lftb)*lhs(1, self.N_f)
        self.x_f = torch.tensor(x_f, requires_grad=True).float().to(device) #training points
        self.t_f = torch.tensor(t_f, requires_grad=True).float().to(device)
        
        self.optimizer.zero_grad()
        # Predictions
        self.u_l = self.net_u(self.x_lb, self.t_lb)
        self.u_u = self.net_u(self.x_ub, self.t_ub)
        self.u_0 = self.net_u(self.x_0, self.t_0)
        self.r_pred = self.net_r(self.x_f, self.t_f) # training points
        
        loss_r = torch.mean(self.r_pred**2)
            
        loss_0 = torch.mean((self.u_0 - torch.sin(self.x_0))**2)
        

        # Loss calculation
        self.loss = loss_r + 1000*loss_0
        self.loss.backward()
        self.iter += 1

        if self.iter % self.print_step == 0:
           
            with torch.no_grad():
                # Grid prediction (for relative L2)
                res = self.net_u(self.x_tst, self.t_tst)
                sol = tonp(res)

                # L2 calculation
                l2_rel = np.linalg.norm(self.Exact - sol, 2) / np.linalg.norm(self.Exact, 2)
                print('Iter %d, Loss: %.3e, Rel_L2: %.3e, t/iter: %.1e' % 
                      (self.iter, self.loss.item(),  l2_rel, self.exec_time))
                
                self.it.append(self.iter)
                self.l.append(self.loss.item())
                self.l2.append(l2_rel)
                d = np.column_stack((np.array(self.it), np.array(self.l2), np.array(self.l)))
                np.savetxt('losses.txt', d, fmt='%.10f %.10f %.10f')
        # Optimizer step
        self.optimizer.step()
                
    def train(self):
        """ Train model """
        model_parameters = filter(lambda p: p.requires_grad, self.dnn.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('all trainable parameters: %d' % params)
        
        
        self.dnn.train()
        for epoch in range(self.epochs):
            start_time = time.time()
            self.loss_func()
            end_time = time.time()
            self.exec_time = end_time - start_time
            if (epoch+1) % self.step_size == 0:
                self.scheduler.step()
                
            if (epoch+1) % 5000 == 0:
                PATH = "checkpoint_%d.pt" % (epoch)
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.dnn.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': self.loss,
                            }, PATH)
        
        # Write data
        a = np.array(self.it)
        b = np.array(self.l2)
        c = np.array(self.l)
        # Stack them into a 2D array.
        self.d = np.column_stack((a, b, c))
        # Write to a txt file
        
    def predict(self, X, T):
        x = torch.tensor(X).float().to(device)
        t = torch.tensor(T).float().to(device)
        self.dnn.eval()
        u = self.net_u(x, t)
        u = tonp(u)
        return u

hidden = 100
layers = [3] + [hidden] + [1]

