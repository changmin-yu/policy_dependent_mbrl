import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:0')

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0.2)


class DS_Gen(nn.Module):
    def __init__(self, inputs_dim, hidden_dim, outputs_dim):
        super(DS_Gen, self).__init__()

        self.linear1 = nn.Linear(inputs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, outputs_dim)

        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x




def pre_def_data(x):
    x = x.cpu().numpy()
    return torch.FloatTensor(np.concatenate([np.mean(np.sin(20*x)*1,1, keepdims=True), np.mean(-np.exp(x**2)/10 - np.sin(x)/10 - x**2 ,1 , keepdims=True)],1)).to(device)
    # return np.array([np.mean(np.exp(x)/10 + np.sin(x)/5 + x**4), np.mean(-np.exp(x**2)/5 - np.sin(x)/10 - x**2  )])


# pre_data_x = np.array([ [-0.25,-0.25,-0.25,-0.25],
#                     [-0.15,-0.15,-0.15,-0.15],
#                     [-0.05,-0.05,-0.05,-0.05],
#                     [0.05, 0.05, 0.05, 0.05],
#                     [0.15, 0.15, 0.15, 0.15],
#                     [0.25, 0.25, 0.25, 0.25], ])

# pre_data_y = np.array([pre_def_data(x) for x in pre_data_x])



# pre_data_x = torch.FloatTensor(pre_data_x).to(device)
# pre_data_y = torch.FloatTensor(pre_data_y).to(device)

# ds_y_gen = DS_Gen(4,20,2).to(device)
# optim = torch.optim.SGD(ds_y_gen.parameters(), lr=0.1)
# Loss = torch.nn.MSELoss()
# for i in range(50):
#     out = ds_y_gen(pre_data_x)
#     loss = Loss(out, pre_data_y)
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
#     print(loss.item())




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
    def __init__(self, mu, sigma, n):
        self.mu = np.array(mu).reshape([1,-1])
        self.sigma = np.array(sigma).reshape([1,-1])
        self.n = n
        self.dimension = self.mu.shape[1]


    def generate(self,):
        return np.random.randn(self.n, self.dimension) * self.sigma + self.mu


configs =  [
    {'num_data': 100, 'sigma_factor' : 2, 'mu_factor' : 5, 'hidden' : 4},
    {'num_data': 100, 'sigma_factor' : 0.5, 'mu_factor' : 5, 'hidden' : 4},
    {'num_data': 100, 'sigma_factor' : 1, 'mu_factor' : 5, 'hidden' : 4},
    {'num_data': 100, 'sigma_factor' : 2, 'mu_factor' : 3, 'hidden' : 4},
    {'num_data': 100, 'sigma_factor' : 2, 'mu_factor' : 8, 'hidden' : 4},
    {'num_data': 100, 'sigma_factor' : 2, 'mu_factor' : 5, 'hidden' : 2},
    {'num_data': 100, 'sigma_factor' : 2, 'mu_factor' : 5, 'hidden' : 8},
    # {'num_data': 100, 'sigma_factor' : 2, 'mu_factor' : 5, 'hidden' : 4},
    # {'num_data': 100, 'sigma_factor' : 2, 'mu_factor' : 5, 'hidden' : 4},
    # {'num_data': 100, 'sigma_factor' : 2, 'mu_factor' : 5, 'hidden' : 4},
    # {'num_data': 100, 'sigma_factor' : 2, 'mu_factor' : 5, 'hidden' : 4},
    # {'num_data': 100, 'sigma_factor' : 2, 'mu_factor' : 5, 'hidden' : 4},
    # {'num_data': 100, 'sigma_factor' : 2, 'mu_factor' : 5, 'hidden' : 4},
    # {'num_data': 100, 'sigma_factor' : 2, 'mu_factor' : 5, 'hidden' : 4},
]

for config in configs:
    with torch.no_grad():
        ds_y_gen  = pre_def_data
        num_data = config['num_data'] 
        sigma_factor = config['sigma_factor'] 
        mu_factor = config['mu_factor'] 

        ds_0_x = torch.FloatTensor(data_generation(np.array([-0.1, -0.2, -0.1, -0.2])*mu_factor, np.array([0.2,0.2,0.2,0.2])*sigma_factor, num_data).generate()).to(device)
        ds_00_x = torch.FloatTensor(data_generation(np.array([-0.1, -0.2, -0.1, -0.2])*mu_factor, np.array([0.2,0.2,0.2,0.2])*sigma_factor, num_data).generate()).to(device)
        
        ds_2_x = torch.FloatTensor(data_generation(np.array([0.2, 0.3, 0.2, 0.3])*mu_factor, np.array([0.2,0.2,0.2,0.2])*sigma_factor, num_data).generate()).to(device)
        
        ds_4_x = torch.FloatTensor(data_generation(np.array([-0.2, -0.2, -0.2, -0.2])*mu_factor, np.array([0.2,0.2,0.2,0.2])*sigma_factor, num_data).generate()).to(device)
        
        ds_6_x = torch.FloatTensor(data_generation(np.array([0.3, 0.3, 0.3, 0.3])*mu_factor, np.array([0.2,0.2,0.2,0.2])*sigma_factor, num_data).generate()).to(device)
        

        sigma_factor = 2
        ds_1_x = torch.FloatTensor(data_generation(np.array([-0.1, -0.2, -0.1, -0.2])*mu_factor, np.array([0.2,0.2,0.2,0.2])*sigma_factor, num_data).generate()).to(device)
        ds_3_x = torch.FloatTensor(data_generation(np.array([0.2, 0.3, 0.2, 0.3])*mu_factor, np.array([0.2,0.2,0.2,0.2])*sigma_factor, num_data).generate()).to(device)
        ds_5_x = torch.FloatTensor(data_generation(np.array([-0.2, -0.2, -0.2, -0.2])*mu_factor, np.array([0.2,0.2,0.2,0.2])*sigma_factor, num_data).generate()).to(device)
        ds_7_x = torch.FloatTensor(data_generation(np.array([0.3, 0.3, 0.3, 0.3])*mu_factor, np.array([0.2,0.2,0.2,0.2])*sigma_factor, num_data).generate()).to(device)
        # sigma_factor = 1

        ds_8_x = torch.cat([ds_0_x, ds_00_x ],0)
        ds_9_x = torch.cat([ds_0_x, ds_2_x ],0)
        ds_10_x = torch.cat([ds_0_x, ds_4_x ],0)


    
        




        ds_0_y =    ds_y_gen(ds_0_x)
        ds_1_y =    ds_y_gen(ds_1_x)
        ds_2_y =    ds_y_gen(ds_2_x)
        ds_3_y =    ds_y_gen(ds_3_x)
        ds_4_y =    ds_y_gen(ds_4_x)
        ds_5_y =    ds_y_gen(ds_5_x)
        ds_6_y =    ds_y_gen(ds_6_x)
        ds_7_y =    ds_y_gen(ds_7_x)
        ds_8_y =    ds_y_gen(ds_8_x)
        ds_9_y =    ds_y_gen(ds_9_x)
        ds_10_y =   ds_y_gen(ds_10_x)

    


    all_data_x = [ds_0_x, ds_1_x, ds_2_x, ds_3_x, ds_4_x, ds_5_x, ds_6_x, ds_7_x, ds_8_x, ds_9_x, ds_10_x,]
    all_data_y = [ds_0_y, ds_1_y, ds_2_y, ds_3_y, ds_4_y, ds_5_y, ds_6_y, ds_7_y, ds_8_y, ds_9_y, ds_10_y,]



    res = {}


    for kk in range(11):
        print(kk)
        model = DS_Learn(4,config['hidden'],2).to(device)

        optim = torch.optim.SGD(model.parameters(), lr=0.02)
        Loss = torch.nn.MSELoss()
        data_x = all_data_x[kk]  
        data_y = all_data_y[kk]




        res[f'ds_{kk}'] = {'train':{}, 'test':{}}

        
        for ii in range(num_data//10 * 500):
            num_data = data_x.shape[0]
            idx = np.random.randint(0, num_data, 10)
            out = model(data_x[idx,:])
            loss = Loss(out, data_y[idx,:])
            optim.zero_grad()
            loss.backward()
            optim.step()
            # print(loss.item())

            if ii % 100 == 0:
                for jj in range(11):
                    test_data_x = all_data_x[jj]
                    test_data_y = all_data_y[jj]            
                    out = model(test_data_x)
                    loss = Loss(out, test_data_y)
                    if f'test_ds_{jj}' not in res[f'ds_{kk}']['test']:
                        res[f'ds_{kk}']['test'][f'test_ds_{jj}'] = []
                    res[f'ds_{kk}']['test'][f'test_ds_{jj}'].append([loss.item()])
            


    pd_res = {}

    for key, value in res.items():
        for key1, value1 in value['test'].items():
            if key  not in pd_res:
                pd_res[key] = {}
            if key1 not in pd_res[key]:
                pd_res[key][key1] = {}
            pd_res[key][key1] = value1[-1][0]
    import pandas as pd

    df = pd.DataFrame(pd_res)

    name = ''
    for key, value in config.items():
        name += f'{key}-{value}-'

    df.to_csv(f'{name}res.csv')
