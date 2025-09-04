import time
import torch
import numpy as np
from pyDOE import lhs
from torch.optim import lr_scheduler
from collections import OrderedDict


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
        # self.activation = torch.nn.SiLU()
        self.layer1 = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer1.weight)
        self.layer2 = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer2.weight)
        
        self.layer1s = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer1s.weight)
        self.layer2s = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer2s.weight)
        
        self.layer1t = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer1t.weight)
        self.layer2t = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer2t.weight)
        
        
    def forward(self, inputs):
        H = inputs
        x1 = self.activation(self.layer1(H))
        x1s = self.activation(self.layer1s(H))
        x1t = self.activation(self.layer1t(H))
        x1 = x1*x1s*x1t
        
        x2 = self.activation(self.layer2(x1))
        x2s = self.activation(self.layer2s(x1))
        x2t = self.activation(self.layer2t(x1))
        x2 = x2*x2s*x2t + H
        
        return x2
        

class DNN(torch.nn.Module):
    def __init__(self, layers, n_blocks):
        super(DNN, self).__init__()
        self.activation = torch.nn.Tanh()
        
        layer_list = list()
        
        for i in range(n_blocks):
            layer_list.append(('star_block_%d' % i, star_block(layers[1])))
        layerDict = OrderedDict(layer_list)
        # Deploy layers
        self.features = torch.nn.Sequential(layerDict)
        
        self.out_layer = torch.nn.Linear(layers[1], layers[2])
        torch.nn.init.xavier_normal_(self.out_layer.weight)
        
        self.W = torch.nn.Parameter(1*torch.randn(layers[0], layers[1] // 2), requires_grad=True)
        
        
    def forward(self, inputs):
        H = inputs
        x1 =  torch.cat([torch.sin(torch.matmul(H, self.W)),
                          torch.cos(torch.matmul(H, self.W))], dim=1)
        
        out = self.features(x1)
        
        out = self.out_layer(out)
        return out
    
class PhysicsInformedNN():
    """ PINN Class """
    
    def __init__(self, N0, N_b, N_f, X_exact, c, n_blocks):
        
        # Initialization
        self.iter = 0
        self.exec_time = 0
        self.print_step = 100
        self.epoch = 30000
        self.it = []; self.l2 = []; self.l = []
        self.l_ini = []; self.l_g = []; self.l_bound = []
        self.loss, self.losses = None, []
        
        self.c = c
        self.N0 = N0
        self.N_b = N_b
        self.N_f = N_f
        # Data
        self.X_exact = X_exact
        self.x_tst = torch.tensor(self.X_exact[:, 0:1], requires_grad=True).float().to(device) # test points
        self.t_tst = torch.tensor(self.X_exact[:, 1:2], requires_grad=True).float().to(device)
        self.Exact = self.X_exact[:, 2:3]
        
        self.dnn = DNN(layers, n_blocks).to(device)
        
        # Optimizer (1st ord)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, verbose=True)
        self.step_size = 500
        
    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_r(self, x, t):
        """ Residual calculation """
        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_tt = grad(u_t, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        f_u = u_tt - self.c**2*u_xx
        return f_u
    
    
    def loss_func(self):
        """ Loss function """
        
        x_0 = ub[0]*lhs(1, self.N0)
        self.x_0 = torch.tensor(x_0, requires_grad=True).float().to(device)
        self.t_0 = torch.tensor(np.zeros_like(x_0), requires_grad=True).float().to(device)
        
        t_b = rb[0]*lhs(1, self.N_b)
        self.x_lb = torch.tensor(np.zeros_like(t_b) + lb[0], requires_grad=True).float().to(device)
        self.t_lb = torch.tensor(t_b, requires_grad=True).float().to(device)
        self.x_ub = torch.tensor(np.zeros_like(t_b) + ub[0], requires_grad=True).float().to(device)
        self.t_ub = torch.tensor(t_b, requires_grad=True).float().to(device)
        
        x_f = lb + (ub-lb)*lhs(1, self.N_f)
        t_f = lftb + (rb - lftb)*lhs(1, self.N_f)
        self.x_f = torch.tensor(x_f, requires_grad=True).float().to(device) #training points
        self.t_f = torch.tensor(t_f, requires_grad=True).float().to(device)
        
        self.optimizer.zero_grad()
        # Predictions
        self.u_l = self.net_u(self.x_lb, self.t_lb)
        self.u_u = self.net_u(self.x_ub, self.t_ub)
        self.u_0 = self.net_u(self.x_0, self.t_0)
        self.r_pred = self.net_r(self.x_f, self.t_f) # training points
        
        self.loss_r = torch.mean(self.r_pred**2)
            
        self.loss_0 = torch.mean((self.u_0 - torch.sin(torch.pi*self.x_0))**2 + (grad(self.u_0, self.t_0))**2)
        
        self.loss_b = torch.mean((self.u_l)** 2 + (self.u_u)**2)

        # Loss calculation
        self.loss = self.loss_r  + self.loss_0 + self.loss_b
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
                # print(tonp(self.dnn.W), tonp(self.dnn.star_block25.W2))
                
                self.it.append(self.iter)
                self.l.append(self.loss.item())
                self.l2.append(l2_rel)
                self.l_ini.append(self.loss_0.item())
                
                a = np.array(self.it)
                b = np.array(self.l2)
                c = np.array(self.l)
                d = np.array(self.l_ini)
                # Stack them into a 2D array.
                e = np.column_stack((a, b, c, d))
                np.savetxt('losses.txt', e, fmt='%.10f %.10f %.10f %.10f')
                
        # Optimizer step
        self.optimizer.step()
        self.losses.append(self.loss.item())
        
    def train(self):
        """ Train model """
        model_parameters = filter(lambda p: p.requires_grad, self.dnn.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('all trainable parameters: %d' % params)
        
        
        self.dnn.train()
        for epoch in range(self.epoch):
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
        with torch.no_grad():
            x = torch.tensor(X).float().to(device)
            t = torch.tensor(T).float().to(device)
            self.dnn.eval()
            u = self.net_u(x, t)
            u = tonp(u)
        return u

hidden = 100
layers = [2] + [hidden] + [1]

lb = np.array([0.0]) # low x
ub = np.array([1.0]) # up x
lftb = np.array([0.0]) # left t
rb = np.array([10.0]) # right t
