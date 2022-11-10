import argparse
import os
import itertools
import torch
from torch import nn
import numpy as np

from buffer.trajectory_replay_buffer import ReplayBuffer
from mbpo_dynamics_model import EnsembleDynamicsModel
from envs import make_dmc_env
from model.network import PolicyEmbeddingGRU, PolicyEmbedding_VAE

os.environ['MUJOCO_GL'] = 'egl'

def get_args():
    parser = argparse.ArgumentParser(description='Training MBPO dynamics model with all cached trajectory data using extended one-hot encoding')
    parser.add_argument('--env', default='quadruped_run', type=str, 
                        help='task name')
    parser.add_argument('--device', default='cuda', type=str, 
                        help='device')
    parser.add_argument('--seed', default=1, type=int, 
                        help='random seed')
    parser.add_argument('--model-num-networks', default=7, type=int, 
                        help='number of networks in the ensemble dynamics model')
    parser.add_argument('--model-num-elites', default=5, type=int, 
                        help='number of elite networks')
    parser.add_argument('--model-hidden-size', default=200, type=int, 
                        help='hidden size for dynamics model')
    parser.add_argument('--model-use-decay', default=False, type=bool, 
                        help='Weight decay in learning the dynamics model')
    parser.add_argument('--model-predict-done', default=False, type=bool, 
                        help='dynamics model predict done signal')
    parser.add_argument('--actor-index', default=1e5, type=float, 
                        help='index of the cached actor for generating trajectory data')
    parser.add_argument('--buffer-capacity', default=1e6, type=float, 
                        help='replay buffer capacity')
    parser.add_argument('--save-policy-trajectory-prefix', default='logs', type=str, 
                        help='prefix for saving the policy generated trajectory data')
    parser.add_argument('--batch-size', default=128, type=int, 
                        help='batch size for model learning')
    parser.add_argument('--num-epochs', default=5, type=int, 
                        help='number of epochs for model learning')
    parser.add_argument('--max-epochs-since-update', default=5, type=int, 
                        help='')
    parser.add_argument('--model-lr', default=1e-4, type=float, 
                        help='learning rate for training dynamics model')
    parser.add_argument("--one-hot-extension", default=True, type=bool, 
                        help='using one hot extension for training on MBPO dynamics model')
    parser.add_argument('--post-fix', default='', type=str, 
                        help='post fix of logdir')
    parser.add_argument('--one-hot-size', default=19, type=int, 
                        help='number of classes for one-hot encoding')
    parser.add_argument('--gru-num-layers', default=1, type=int, 
                        help='number of GRU hidden layers')
    parser.add_argument('--pretraining-batch-size', default=64, type=int, 
                        help='batch size for pretraining policy embedding GRU')
    parser.add_argument('--gru-lr', default=1e-4, type=float, 
                        help='learning rate for pretraining GRU policy embedding')
    parser.add_argument('--gru-num-epochs', default=500, type=int, 
                        help='number of epochs for training gru policy embedding')
    parser.add_argument('--valid-ratio', default=0.1, type=float, 
                        help='validation ratio for pretraining')
    parser.add_argument('--gru-hidden-dim', default=32, type=int, 
                        help='number of hidden units of GRU')
    parser.add_argument('--predict-reward', default=False, type=bool, 
                        help='whether or not let the GRU to predict reward as well')
    parser.add_argument('--trajectory-len', default=1000, type=int, 
                        help='length of the trajectory data')
    parser.add_argument('--vae-latent-dim', default=64, type=int, 
                        help='latent dimension of VAE embedding of Policy Embedding (GRU hiddens)')
    parser.add_argument('--vae-lr', default=5e-4, type=float, 
                        help='learning rate for training vae')
    parser.add_argument('--vae-num-epochs', default=20, type=int, 
                        help='number of training epochs for PE_VAE')
    return parser.parse_args()

def compute_policy_embedding(trajectories):
    num_trajectory = len(trajectories)
    # for i in range(num_trajectory):


def pretraining_policy_embedding(cfg):
    env = make_dmc_env(cfg)
    device = torch.device(cfg.device)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    input_dim = obs_dim + action_dim
    output_dim = obs_dim+1 if cfg.predict_reward else obs_dim
    
    PE_GRU = PolicyEmbeddingGRU(input_dim, cfg.gru_hidden_dim, output_dim, cfg.gru_num_layers).to(device)
    PE_GRU_criterion = nn.MSELoss()
    PE_GRU_optimiser = torch.optim.Adam(PE_GRU.parameters(), lr=cfg.gru_lr)
    
    buffer_logdir = os.path.join(cfg.save_policy_trajectory_prefix, cfg.env+'_'+cfg.post_fix, 'trajectories')
    cached_trajectory_fnames = os.listdir(buffer_logdir)
    
    print('start loading buffer trajectories...')
    buffer_dict = {str(f[:-3]): torch.load(os.path.join(buffer_logdir, f)) for f in cached_trajectory_fnames}
    
    obs = np.concatenate([buffer_dict[k].obses[:100] for k in buffer_dict.keys()], axis=0)
    next_obs = np.concatenate([buffer_dict[k].next_obses[:100] for k in buffer_dict.keys()], axis=0)
    action = np.concatenate([buffer_dict[k].actions[:100] for k in buffer_dict.keys()], axis=0)
    # reward = np.concatenate([buffer_dict[k].rewards[:100] for k in buffer_dict.keys()], axis=0)
    print(f'finished loading dataset, number of trajectory samples: {len(obs)}')
    
    num_samples, seqlen = obs.shape[0], obs.shape[1]
    
    random_ind = np.random.permutation(obs.shape[0])
    obs = obs[random_ind]
    action = action[random_ind]
    # reward = reward[random_ind]
    next_obs = next_obs[random_ind]
    
    inp = np.concatenate((obs, action), axis=-1)
    target = next_obs
    num_valid = int(cfg.valid_ratio * num_samples)
    num_train = num_samples = num_valid
    
    train_input = inp[num_valid:]
    train_target = target[num_valid:]
    valid_input = inp[:num_valid]
    valid_target = target[:num_valid]
    valid_input = torch.from_numpy(valid_input).float().to(device)
    valid_target = torch.from_numpy(valid_target).float().to(device)
    
    for epoch in range(cfg.gru_num_epochs):
        train_idx = np.random.permutation(train_input.shape[0])
        losses = []
        valid_losses_list = []
        for start_ind in np.arange(0, train_input.shape[0], cfg.pretraining_batch_size):
            idx = train_idx[start_ind:min(start_ind+cfg.pretraining_batch_size, train_input.shape[0])]
            inp = torch.from_numpy(train_input[idx]).float().to(device)
            tar = torch.from_numpy(train_target[idx]).float().to(device)
            
            pred_tar = PE_GRU(inp)
            loss = PE_GRU_criterion(tar, pred_tar)
            
            PE_GRU_optimiser.zero_grad()
            loss.backward()
            PE_GRU_optimiser.step()
            
            losses.append(loss)
        print(f'epoch: {epoch} | training loss: {torch.mean(torch.tensor(losses)):.2f}')

        with torch.no_grad():
            valid_pred_tar = PE_GRU(valid_input)
            valid_loss = PE_GRU_criterion(valid_pred_tar, valid_target)
            valid_loss = valid_loss.detach().cpu().numpy()
            print(f'epoch: {epoch} | valid loss: {valid_loss:.2f}')
    
    save_gru_logdir = os.path.join(cfg.save_policy_trajectory_prefix, cfg.env+'_'+cfg.post_fix, 'PE_GRU')
    if not os.path.exists(save_gru_logdir):
        os.makedirs(save_gru_logdir)
    torch.save(PE_GRU, os.path.join(save_gru_logdir, 'pretrained_PE_GRU.pt'))
    
    '''
    Version @ 09/11/2021:
        instead of concatenating all the hidden states across all timepoints, we maintain a polyak average of
        the hidden states:
        mu_{t+1} = (1-r) * mu_{t} + r * h_{t+1}
        where mu_{t} is the policy representation at time point t, r is the update ratio, h_{t} is the hidden 
        state of the GRU at time point t
    '''
    # # start training PE_VAE
    # PE_VAE = PolicyEmbedding_VAE(cfg.trajectory_len, cfg.gru_hidden_dim, cfg.vae_latent_dim, device)
    # PE_VAE_criterion = nn.MSELoss()
    # PE_VAE_optimiser = torch.optim.Adam(PE_VAE.parameters(), lr=cfg.vae_lr)
    # for epoch in range(cfg.vae_num_epochs):
    #     train_idx = np.random.permutation(num_train)
    #     vae_losses = []
    #     for start_ind in np.arange(0, num_train, cfg.pretraining_batch_size):
    #         with torch.no_grad():
    #             batch_size = min(cfg.pretraining_batch_size, num_train-start_ind)
    #             idx = train_idx[start_ind:min(start_ind+cfg.pretraining_batch_size, train_input.shape[0])]
    #             inp = torch.from_numpy(train_input[idx]).float().to(device)
            
    #             _, hidden_list = PE_GRU(inp, return_hidden=True)
    #             vae_inp = hidden_list.transpose(1, 2).view(seqlen, batch_size, -1).transpose(0, 1)
    #         pred_recon = PE_VAE(vae_inp)
    #         vae_loss = PE_VAE_criterion(pred_recon, vae_inp)
            
    #         PE_VAE_optimiser.zero_grad()
    #         vae_loss.backward()
    #         PE_VAE_optimiser.step()
            
    #         vae_losses.append(vae_loss)
    #     print(f'epoch: {epoch} | training loss: {torch.mean(torch.tensor(vae_losses)):.2f}')
        
    #     with torch.no_grad():
    #         _, valid_hidden = PE_GRU(valid_input, return_hidden=True)
    #         valid_vae_inp = valid_hidden.transpose(1, 2).view(seqlen, num_valid, -1).transpose(0, 1)
    #         valid_out = PE_VAE(valid_vae_inp)
    #         valid_loss = PE_VAE_criterion(valid_out, valid_vae_inp)
    #         valid_loss = valid_loss.detach().cpu().numpy()
    #         print(f'epoch: {epoch} | valid loss: {valid_loss:.2f}')
    # save_PE_VAE_logdir = os.path.join(cfg.save_policy_trajectory_prefix, cfg.env+'_'+cfg.post_fix, 'PE_VAE')
    # if not os.path.exists(save_PE_VAE_logdir):
    #     os.makedirs(save_PE_VAE_logdir)
    # torch.save(PE_VAE, os.path.join(save_PE_VAE_logdir, 'pretrained_PE_VAE.pt'))


if __name__=='__main__':
    cfg = get_args()
    cfg.post_fix = 'traj_buffer'
    pretraining_policy_embedding(cfg)