import argparse
import os
import re
import torch
import numpy as np

from buffer.replay_buffer import ReplayBuffer
from mbpo_dynamics_model import EnsembleDynamicsModel
from envs import make_dmc_env
from model.GRU_PE import compute_policy_embedding
from model.network import PolicyEmbedding_VAE, PolicyEmbeddingGRU
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYDEVD_WARN_EVALUATION_TIMEOUT'] = "500"

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
    parser.add_argument('--save-policy-trajectory-prefix', default='cached_policy_gen_data', type=str, 
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
    parser.add_argument('--logdir', default='logs', type=str, 
                        help='high-level dir for cached models')
    return parser.parse_args()

def train_GRU_PE_dynamics_model(cfg):
    env = make_dmc_env(cfg)
    device = torch.device(cfg.device)
    
    # initialise pretrained PE networks
    pretrained_PE_GRU = torch.load(os.path.join('logs', cfg.env+'_'+cfg.post_fix, 'PE_GRU', 'pretrained_PE_GRU.pt'))
    # pretrained_PE_VAE = torch.load(os.path.join('logs', cfg.env+'_'+cfg.post_fix, 'PE_VAE', 'pretrained_PE_VAE.pt'))
    assert isinstance(pretrained_PE_GRU, PolicyEmbeddingGRU)
    # assert isinstance(pretrained_PE_VAE, PolicyEmbedding_VAE)
    pretrained_PE_GRU.to(device)
    # pretrained_PE_VAE.to(device)
    
    # compute policy embeddings
    trajectory_logdir = os.path.join('logs', cfg.env+'_'+cfg.post_fix, 'trajectories')
    trajectory_fnames = os.listdir(trajectory_logdir)
    PE_dict = {}
    for f in trajectory_fnames:
        f_buffer = torch.load(os.path.join(trajectory_logdir, f))
        f_PE = []
        for i in range(0, len(f_buffer), 10):
            obs_batch = f_buffer.obses[i:min(i+10, len(f_buffer))]
            action_batch = f_buffer.actions[i:min(i+10, len(f_buffer))]
            f_PE.append(compute_policy_embedding(obs_batch, action_batch, pretrained_PE_GRU, device))# , pretrained_PE_VAE))
        f_PE = torch.cat(f_PE, dim=0).mean(dim=0).view(1, -1)
        PE_dict[int(f[:-3])] = f_PE.cpu().detach().numpy()
        print(f'Policy Embedding for trajectory no.{int(f[:-3])} done!')
    PE_dim = f_PE.shape[-1]
    f_buffer = None
    f_PE = None
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    dynamics_model = EnsembleDynamicsModel(network_size=cfg.model_num_networks, elite_size=cfg.model_num_elites, state_size=obs_dim, 
                                           action_size=action_dim, reward_size=1, hidden_size=cfg.model_hidden_size, use_decay=cfg.model_use_decay, 
                                           predict_done=cfg.model_predict_done, learning_rate=cfg.model_lr, PE_size=PE_dim)
    
    print('start loading buffer trajectories')
    buffer_dict = {int(f[:-3]): torch.load(os.path.join(trajectory_logdir, f)) for f in trajectory_fnames}
    
    obs = np.concatenate([buffer_dict[k].obses for k in buffer_dict.keys()], axis=0)
    action = np.concatenate([np.concatenate([buffer_dict[k].actions, np.tile(PE_dict[k][None, ...], \
        [buffer_dict[k].actions.shape[0], buffer_dict[k].actions.shape[1], 1])], axis=-1) for k in buffer_dict.keys()], axis=0)
    reward = np.concatenate([buffer_dict[k].rewards for k in buffer_dict.keys()], axis=0)
    next_obs = np.concatenate([buffer_dict[k].next_obses for k in buffer_dict.keys()], axis=0)
    print(f'number of samples: {len(obs)}')
    
    buffer_dict = None
    # obs, action, reward, next_obs = replay_buffer.obses, replay_buffer.actions, replay_buffer.rewards, replay_buffer.next_obses
    
    obs = obs.reshape(-1, obs.shape[-1])
    action = action.reshape(-1, action.shape[-1])
    reward = reward.reshape(-1, reward.shape[-1])
    next_obs = next_obs.reshape(-1, next_obs.shape[-1])
    
    valid_idx = (np.abs(reward) < 1e2).flatten()
    obs = obs[valid_idx]
    action = action[valid_idx]
    reward = reward[valid_idx]
    next_obs = next_obs[valid_idx]
    
    random_ind = np.random.permutation(obs.shape[0])[:int(1e6)]
    obs = obs[random_ind]
    action = action[random_ind]
    reward = reward[random_ind]
    next_obs = next_obs[random_ind]
    print(f'number of randomly subsampled samples: {len(obs)}')
    
    inputs = np.concatenate((obs, action), axis=-1)
    delta_obs = next_obs - obs
    labels = np.concatenate((reward, delta_obs), axis=-1)
    
    print('start training')
    train_loss, holdout_mse_loss = dynamics_model.train(inputs, labels, batch_size=cfg.batch_size, holdout_ratio=0.2, 
                                                        max_epochs_since_update=cfg.max_epochs_since_update)
    print('finished training')
    save_model_logdir = os.path.join(cfg.save_policy_trajectory_prefix, cfg.env+cfg.post_fix, 'trained_PE_GRU_extended_models')
    if not os.path.exists(save_model_logdir):
        os.makedirs(save_model_logdir)
    torch.save(dynamics_model, os.path.join(save_model_logdir, 'PE_GRU_extended_model.pt'))


if __name__=='__main__':
    cfg = get_args()
    cfg.post_fix = 'traj_buffer'
    # for index in np.arange(100000, 2000000, 100000):
    #     cfg.actor_index = index
    #     print(f'actor index: {cfg.actor_index}')
    train_GRU_PE_dynamics_model(cfg)