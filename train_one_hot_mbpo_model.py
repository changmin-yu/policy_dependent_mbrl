import argparse
import os
import torch
import numpy as np

from buffer.replay_buffer import ReplayBuffer
from mbpo_dynamics_model import EnsembleDynamicsModel
from envs import make_dmc_env

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
    return parser.parse_args()

def to_one_hot_1e5(ind):
    out = np.zeros((1, 19))
    ind = ind // 100000
    out[0, int(ind-1)] = 1
    return out

def train_unified_ensemble_dynamics_model(cfg):
    env = make_dmc_env(cfg)
    device = cfg.device
    
    # random_state = np.random.RandomState(cfg.seed)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    dynamics_model = EnsembleDynamicsModel(network_size=cfg.model_num_networks, elite_size=cfg.model_num_elites, state_size=obs_dim, 
                                           action_size=action_dim, reward_size=1, hidden_size=cfg.model_hidden_size, use_decay=cfg.model_use_decay, 
                                           predict_done=cfg.model_predict_done, learning_rate=cfg.model_lr, one_hot_size=cfg.one_hot_size)
    
    # replay_buffer = ReplayBuffer(obs_dim, action_dim, cfg.buffer_capacity, device)
    
    buffer_logdir = os.path.join(cfg.save_policy_trajectory_prefix, cfg.env, 'trajectories')# , str(int(cfg.actor_index))+'.pt')
    cached_trajectory_fnames = os.listdir(buffer_logdir)
    if not os.path.exists(buffer_logdir):
        raise NotImplementedError('cached buffer not found')
    
    print('start loading buffer trajectories')
    buffer_dict = {str(f): torch.load(os.path.join(buffer_logdir, f)) for f in cached_trajectory_fnames}
    
    # replay_buffer = torch.load(buffer_logdir)
    # buffer_length = len(replay_buffer)
    
    obs = np.concatenate([buffer_dict[k].obses for k in buffer_dict.keys()], axis=0)
    action = np.concatenate([np.concatenate([buffer_dict[k].actions, np.repeat(to_one_hot_1e5(int(k[:-3])), len(buffer_dict[k]), axis=0)], axis=-1) for k in buffer_dict.keys()], axis=0)
    reward = np.concatenate([buffer_dict[k].rewards for k in buffer_dict.keys()], axis=0)
    next_obs = np.concatenate([buffer_dict[k].next_obses for k in buffer_dict.keys()], axis=0)
    print(f'number of samples: {len(obs)}')
    # obs, action, reward, next_obs = replay_buffer.obses, replay_buffer.actions, replay_buffer.rewards, replay_buffer.next_obses
    
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
    save_model_logdir = os.path.join(cfg.save_policy_trajectory_prefix, cfg.env+cfg.post_fix, 'trained_models')
    if not os.path.exists(save_model_logdir):
        os.makedirs(save_model_logdir)
    torch.save(dynamics_model, os.path.join(save_model_logdir, 'unified_model.pt'))


if __name__=='__main__':
    cfg = get_args()
    # for index in np.arange(100000, 2000000, 100000):
    #     cfg.actor_index = index
    #     print(f'actor index: {cfg.actor_index}')
    train_unified_ensemble_dynamics_model(cfg)