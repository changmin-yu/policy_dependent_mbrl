# -*- coding: utf-8 -*-
"""
Created on Mon May 31 21:30:47 2021

@author: yuwang5
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import TweedieRegressor

# SEED = 10
# np.random.seed(SEED)
# torch.random.manual_seed(SEED)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0.2)



class DS_Learn(nn.Module):
    def __init__(self, inputs_dim, hidden_dim, outputs_dim):
        super(DS_Learn, self).__init__()

        self.linear1 = nn.Linear(inputs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, outputs_dim)

        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    
class data_generation:
    def __init__(self, left, right, n):
        self.left = np.array(left).reshape([1,-1])
        self.right = np.array(right).reshape([1,-1])
        self.n = n
        self.classes = self.left.shape[1]


    def generate(self,):
        
        tmp = []
        for i in range(self.classes):
             tmp.append(np.random.uniform(self.left[0,i], self.right[0,i], [self.n//self.classes, 1]) )
            
        
        return  np.concatenate(tmp, 0)
    
n=20
a = np.concatenate((
    np.random.uniform(-0.5,-0.4,size=n),
    np.random.uniform(-0.9,-0.8,size=n),
    np.random.uniform(-0.1,0.2,size=n),
    np.random.uniform(-0.3,-0.2,size=n),
    np.random.uniform(0.5,0.6,size=n),
    np.random.uniform(0.9,1.0,size=n),
    np.random.uniform(0.,0.1,size=n),
    np.random.uniform(0.2,0.3,size=n),
    np.random.uniform(-0.6,-0.6,size=n),
    ))

x = np.array(list(range(len(a))))
x = x/ max(x)

x = x*4 -2


z1 = np.polyfit(x, a, 90) 
p1 = np.poly1d(z1)
aa = p1(x)
plt.plot(x,a, 'b', label='true')
plt.plot(x,aa, 'r', label='polyfit')
plt.legend()
plt.savefig('polyfit.png')



configs =  [
    {'num_data': 1000, 'left' : [-2], 'right' : [2], 'hidden' : 4},    
    {'num_data': 2000, 'left' : [-2], 'right' : [2], 'hidden' : 4}, 
    {'num_data': 3000, 'left' : [-2], 'right' : [2], 'hidden' : 4}, 
    {'num_data': 1000, 'left' : [-2,], 'right' : [-1], 'hidden' : 4},    
    {'num_data': 1000, 'left' : [-2,], 'right' : [0], 'hidden' : 4}, 
    {'num_data': 2000, 'left' : [-2,], 'right' : [0], 'hidden' : 4}, 
    {'num_data': 1000, 'left' : [-2, 1], 'right' : [-1,2], 'hidden' : 4},     
    {'num_data': 1000, 'left' : [-2, 1], 'right' : [0,2], 'hidden' : 4},
    {'num_data': 1000, 'left' : [-2, 0], 'right' : [0,2], 'hidden' : 4},
    {'num_data': 1000, 'left' : [-2, 0], 'right' : [0,2], 'hidden' : 4},
    {'num_data': 1000, 'left' : [-2, 0], 'right' : [0,2], 'hidden' : 4},
    {'num_data': 1000, 'left' : [-2, 0], 'right' : [0,2], 'hidden' : 4},
    {'num_data': 1000, 'left' : [-2, 0], 'right' : [0,2], 'hidden' : 4},
    {'num_data': 1000, 'left' : [-2, 0], 'right' : [0,2], 'hidden' : 4},
    {'num_data': 1000, 'left' : [-2, 0], 'right' : [0,2], 'hidden' : 4},
    {'num_data': 1000, 'left' : [-2, 0], 'right' : [0,2], 'hidden' : 4},
    {'num_data': 1000, 'left' : [-2, 0], 'right' : [0,2], 'hidden' : 4},
    {'num_data': 1000, 'left' : [-2, 0], 'right' : [0,2], 'hidden' : 4},  
    {'num_data': 2000, 'left' : [-2, 1], 'right' : [-1,2], 'hidden' : 4},  
]

def train_config(cfg, batch_size, num_epoch, polyfit, model, test, test_low, test_high, test_n, model_kwargs, impl='torch'):
    num_data = cfg['num_data']
    left = cfg['left']
    right = cfg['right']
    
    dg = data_generation(left, right, num_data)
    x = dg.generate()
    y = polyfit(x)
    
    if impl == 'torch':
        ds_x = torch.FloatTensor(x)
        ds_y = torch.FloatTensor(y)
    else:
        ds_x = x
        ds_y = y
         
    print(model_kwargs)
    mdl = model(**model_kwargs)
    
    if impl == 'torch':
        
        optim = torch.optim.SGD(mdl.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        
        for i in range(num_data//batch_size * num_epoch):
            idx = np.random.randint(0, num_data, batch_size)
            out = mdl(ds_x[idx, :])
            loss = loss_fn(out, ds_y[idx, :])
            optim.zero_grad()
            loss.backward()
            optim.step()
            if i % 1000 == 0:
                print(f'training iter: {i} | loss: {loss.item()}')
        
        if test:
            test_x_np = np.linspace(test_low, test_high, test_n)
            test_y_np = polyfit(test_x_np)
            
            test_x = torch.FloatTensor(test_x_np).reshape(-1, 1)
            # test_y = torch.FloatTensor(test_y_np).reshape(-1, 1)
            
            test_pred_y = mdl(test_x)
            for i in range(4):
                start = int(test_n/4*i)
                end = int(test_n/4*(i+1))
                test_loss_i = np.mean((test_pred_y.data.numpy()[start:end] - test_y_np[start:end])**2)
                print(f'test low: {test_low}, test high: {test_high}, {i+1}/4 test, test loss: {test_loss_i}')
    
    elif impl == 'sklearn':
        mdl.fit(ds_x, ds_y)
        if test:
            test_x_np = np.linspace(test_low, test_high, test_n)
            test_y_np = polyfit(test_x_np)
            
            test_pred_y = mdl.predict(test_x_np.reshape(-1, 1))
            
            for i in range(4):
                start = int(test_n/4*i)
                end = int(test_n/4*(i+1))
                test_loss_i = np.mean((test_pred_y[start:end] - test_y_np[start:end])**2)
                print(f'test low: {test_low}, test high: {test_high}, {i+1}/4 test, test loss: {test_loss_i}')
            

# print(configs[3])
# train_config(cfg=configs[3], batch_size=30, num_epoch=200, polyfit=p1, model=DS_Learn, test=True, test_low=-2, test_high=-1, test_n=2000, 
#              model_kwargs={'inputs_dim': 1, 'hidden_dim': 15, 'outputs_dim': 1})

# print(configs[-1])
# train_config(cfg=configs[-1], batch_size=30, num_epoch=200, polyfit=p1, model=DS_Learn, test=True, test_low=-2, test_high=-1, test_n=2000, 
#              model_kwargs={'inputs_dim': 1, 'hidden_dim': 15, 'outputs_dim': 1})

print(configs[3])
train_config(cfg=configs[3], batch_size=30, num_epoch=200, polyfit=p1, model=TweedieRegressor, test=True, test_low=-2, test_high=-1, test_n=2000, 
             model_kwargs={'power': 0.0, 'alpha': 0.0}, impl='sklearn')

print(configs[-1])
train_config(cfg=configs[-1], batch_size=30, num_epoch=200, polyfit=p1, model=TweedieRegressor, test=True, test_low=-2, test_high=-1, test_n=2000, 
             model_kwargs={'power': 0.0, 'alpha': 0.0}, impl='sklearn')

# config = configs[0]

# num_data = config['num_data'] 
# left = config['left'] 
# right = config['right'] 


# dg =  data_generation(left, right, num_data)
# x = dg.generate()
# y = p1(x)
    
# ds_x = torch.FloatTensor(x)
# ds_y = torch.FloatTensor(y)

# model = DS_Learn(1,30,1)

# optim = torch.optim.SGD(model.parameters(), lr=0.01)
# Loss = torch.nn.MSELoss()

# num_data = ds_x.shape[0]

# for ii in range(num_data//30 * 500):

#     idx = np.random.randint(0, num_data, 30)
#     out = model(ds_x[idx,:])
#     loss = Loss(out, ds_y[idx,:])
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
#     if ii % 1000 == 0:
#         print(loss.item())
        
      
        
# test_data_x_np = np.linspace(-2, 2, 2000) 

# test_data_y_np = p1(test_data_x_np)

# test_data_x = torch.FloatTensor(test_data_x_np).reshape(-1,1)
# test_data_y = torch.FloatTensor(test_data_y_np).reshape(-1,1)

# pred_y = model(test_data_x)

# num_test_data = test_data_x_np.shape[0]
# loss_all = []


# for i in range(4):
#     start =  int(num_test_data/ 4 *i)
#     end = int(num_test_data/4*(i+1))
#     loss_all.append(np.mean((pred_y.data.numpy()[start:end]-test_data_y_np[start:end])**2))
    



# plt.plot(test_data_x_np, test_data_y_np,'.')
# plt.plot(test_data_x_np, pred_y.data, '.')
# plt.ylim(-1.5,1.5)






