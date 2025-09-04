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


def input_encoding(x, t):
    L = 2
    w = 2.0 * np.pi / L
    w = torch.tensor(w, dtype=torch.float32).to(device)
    out = torch.cat((t, torch.cos(w * x), torch.sin(w * x)), 1)
    return out

class star_block(torch.nn.Module):
    def __init__(self, width):
        super(star_block, self).__init__()
        self.activation = torch.nn.Tanh()
        self.layer1_1 = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer1_1.weight)
        self.layer2_1 = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer2_1.weight)
        
        self.layer1_2 = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer1_2.weight)
        self.layer2_2 = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer2_2.weight)
        
        self.layer1_3 = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer1_3.weight)
        self.layer2_3 = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer2_3.weight)
        
        self.layer1_4 = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer1_4.weight)
        self.layer2_4 = torch.nn.Linear(width, width, bias=True)
        torch.nn.init.xavier_normal_(self.layer2_4.weight)
        
        
    def forward(self, inputs):
        H = inputs
        x1_1 = self.activation(self.layer1_1(H))
        x1_2 = self.activation(self.layer1_2(H))
        x1_3 = self.activation(self.layer1_3(H))
        x1_4 = self.activation(self.layer1_4(H))
        x1 = x1_1*x1_2*x1_3*x1_4
        
        x2_1 = self.activation(self.layer2_1(x1))
        x2_2 = self.activation(self.layer2_2(x1))
        x2_3 = self.activation(self.layer2_3(x1))
        x2_4 = self.activation(self.layer2_4(x1))
        x2 = x2_1*x2_2*x2_3*x2_4 + H#
        
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
        
        self.W = torch.nn.Parameter(2*torch.randn(layers[0], layers[1] // 2))
        
        
    def forward(self, inputs):
        H = inputs
        H = input_encoding(inputs[:,0:1], inputs[:,1:2])
        x1 =  torch.cat([torch.sin(torch.matmul(H, self.W)),
                          torch.cos(torch.matmul(H, self.W))], dim=1) 
        
        
        x2 = self.features(x1)
        
        out = self.out_layer(x2)
        return out
    
class PhysicsInformedNN():
    """ PINN Class """
    
    def __init__(self, Exact_sol, x_sol, t_sol, N_b, N0, N_f, lb, ub, n_blocks):
        
        # Initialization
        self.iter = 0
        self.exec_time = 0
        self.print_step = 500
        self.total_epoch = 300000
        self.it = []; self.l2 = []; self.l = []
        self.loss = None
        
        self.N0 = N0
        self.N_b = N_b
        self.N_f = N_f
        
        self.lb = lb
        self.ub = ub
        
        # Intermediate results
        self.Exact = Exact_sol
        X, T = np.meshgrid(x_sol, t_sol)
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        self.xx = torch.tensor(X_star[:, 0:1]).float().to(device)
        self.tt = torch.tensor(X_star[:, 1:2]).float().to(device)
        
        self.dnn = DNN(layers, n_blocks).to(device)
        
            
        # Optimizer
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3, betas=(0.99, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, verbose=True)
        self.step_size = 5000
        
        
    def net_u(self, x, t):
        """ Get the velocities """
        
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_r(self, x, t):
        """ Residual calculation """
        
        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        f = u_t - 0.0001*u_xx + 5.0*u*u*u - 5.0*u
        return f

    def loss_func(self):
        """ Loss function """
        
        # Predictions
        x_0 = -2*lhs(1, self.N0) + 1
        
        self.x_0 = torch.tensor(x_0, requires_grad=True).float().to(device)
        self.t_0 = torch.tensor(np.zeros_like(x_0), requires_grad=True).float().to(device)
        
        
        # Collocation points
        X_f = self.lb + (self.ub-self.lb)*lhs(2, self.N_f)
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        
        self.optimizer.zero_grad()
        self.u_0 = self.net_u(self.x_0, self.t_0)
        self.r_pred = self.net_r(self.x_f, self.t_f)

        loss_r = torch.mean(self.r_pred**2)
        loss_0 = torch.mean((self.u_0 - self.x_0**2*torch.cos(torch.pi*self.x_0)) ** 2)
        

        # Loss calculation
        self.loss = loss_r + 100*loss_0
        self.loss.backward()
        self.iter += 1

        if self.iter % self.print_step == 0:
            with torch.no_grad():
                # Grid prediction (for relative L2)
                res = self.net_u(self.xx, self.tt)
                sol = tonp(res)
                sol = np.reshape(sol, (self.Exact.shape[1], self.Exact.shape[0])).T

                # L2 calculation
                l2_rel = np.linalg.norm(self.Exact.flatten() - sol.flatten(), 2) / np.linalg.norm(self.Exact.flatten(), 2)
                print('Iter %d, loss: %.3e, Rel_L2: %.3e, t/iter: %.1e' % 
                      (self.iter, self.loss.item(), l2_rel, self.exec_time))
                print()
                
                self.it.append(self.iter)
                self.l2.append(l2_rel)
                self.l.append(self.loss.item())
                
                # Stack them into a 2D array.
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
        for epoch in range(self.total_epoch):
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
    
    def predict(self, x, t):
        x = torch.tensor(x).float().to(device)
        t = torch.tensor(t).float().to(device)
        self.dnn.eval()
        u = self.net_u(x, t)
        u = tonp(u)
        return u
    
hidden = 512
layers = [3] + [hidden] + [1]