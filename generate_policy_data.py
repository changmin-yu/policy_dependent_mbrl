import os
import argparse
import time
import torch
import dmc2gym
import numpy as np

from agent.sac import SACAgent
from agent.actor import DiagGaussianActor
from agent.critic import DoubleQCritic
from buffer.replay_buffer import ReplayBuffer
import utils
from envs import make_dmc_env

os.environ["MUJOCO_GL"] = "egl"


def get_args():
    parser = argparse.ArgumentParser(description='SAC configs')
    parser.add_argument('--env', default='quadruped_run', type=str, 
                        help='task name')
    parser.add_argument('--num-collection-steps', default=1e5, type=float, 
                        help='name of collection steps')
    parser.add_argument('--replay-buffer-capacity', default=1e5, type=float, 
                        help='replay buffer capacity')
    parser.add_argument('--actor-index', default=1e5, type=float, 
                        help='index for the policy for data collection')
    parser.add_argument('--save-policy-trajectory-prefix', default='cached_policy_gen_data', type=str, 
                        help='prefix for saving the policy generated trajectory data')
    parser.add_argument('--max-episode-steps', default=1000, type=int, 
                        help='maximum number of steps per episode')
    parser.add_argument('--device', default='cuda', type=str, 
                        help='device')
    parser.add_argument('--seed', default=1, type=int, 
                        help='random seed')
    parser.add_argument('--agent-name', default='sac', type=str, 
                        help='agent name')
    parser.add_argument('--discount', default=0.99, type=float, 
                        help='discounting factor')
    parser.add_argument('--init-temperature', default=0.1, type=float, 
                        help='initial temperature parameter')
    parser.add_argument('--alpha-lr', default=1e-4, type=float, 
                        help='temperature parameter learnin rate')
    parser.add_argument('--alpha-betas', default=[0.9, 0.999], type=list, 
                        help='betas for Adam optimiser for temprature parameter')
    parser.add_argument('--actor-lr', default=1e-4, type=float, 
                        help='learning rate for actor learning')
    parser.add_argument('--actor-betas', default=[0.9, 0.999], type=list, 
                        help='betas for Adam optimiser for actor parameters')
    parser.add_argument('--actor-update-frequency', default=1, type=int, 
                        help='frequency for updating actor parameters')
    parser.add_argument('--critic-lr', default=1e-4, type=float, 
                        help='learning rate for critic learning')
    parser.add_argument('--critic-betas', default=[0.9, 0.999], type=list, 
                        help='betas for Adam optimiser for critic parameters')
    parser.add_argument('--critic-tau', default=0.05, type=float, 
                        help='soft updating the target critic model')
    parser.add_argument('--critic-target-update-frequency', default=2, type=int, 
                        help='frequency for updating the target critic model')
    parser.add_argument('--batch-size', default=1024, type=int, 
                        help='batch size')
    parser.add_argument('--learnable-temperature', default=True, type=bool, 
                        help='learnable alpha parameter')
    parser.add_argument('--post-fix', default='', type=str, 
                        help='post fix of logdir')
    parser.add_argument('--verbose', default=True, type=bool, 
                        help='printing progress in actor collecting data')
    return parser.parse_args()


class actor_collect_data:
    def __init__(self, cfg):
        self.workdir = os.getcwd()
        print(f'workspace: {self.workdir}')
        self.cfg = cfg
        self.env = make_dmc_env(cfg)
        self.device = cfg.device
            
        self.cached_actor_logdir = os.path.join(self.workdir, 'logs', cfg.env, 'actor', 'params', str(int(cfg.actor_index))+'.pt')
        if not os.path.exists(self.cached_actor_logdir):
            raise NotImplementedError('Cached actor pt file not found')
        
        GaussianActor_cfg = {'obs_dim': self.env.observation_space.shape[0], 
                             'action_dim': self.env.action_space.shape[0], 
                             'hidden_dim': 1024, 
                             'hidden_depth': 2, 
                             'log_std_bounds': [-5, 2]}
        action_range = [float(self.env.action_space.low.min()), 
                        float(self.env.action_space.high.max())]
        self.action_range = action_range
    
        self.cached_actor = DiagGaussianActor(**GaussianActor_cfg).to(self.device)
        
        checkpoint = torch.load(self.cached_actor_logdir)
        self.cached_actor.load_state_dict(checkpoint)
        self.cached_actor.eval() # no batchnorm or dropout, hence probably not so necessary
        
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)
        
        self.save_buffer_dir = os.path.join(cfg.save_policy_trajectory_prefix, cfg.env+cfg.post_fix, 'trajectories')
        
        self.step = 0
    
    def generate_data(self, save=True):
        if not os.path.exists(self.save_buffer_dir):
            os.makedirs(self.save_buffer_dir)
        episode, episode_reward, done = 0, 0, True
        episode_step = 0
        start_time = time.time()
        while self.step < self.cfg.num_collection_steps:
            if done or episode_step >= self.env._max_episode_steps:
                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
            
            obs_torch = torch.FloatTensor(obs).to(self.device)
            obs_torch = obs_torch.unsqueeze(0)
            dist = self.cached_actor(obs_torch)
            action = dist.mean
            action = action.clamp(*self.action_range)
            assert action.ndim == 2 and action.shape[0] == 1
            action = utils.to_np(action[0])
            
            next_obs, reward, done, _ = self.env.step(action)
            
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            
            episode_reward += reward
            
            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            
            if self.cfg.verbose and self.step % 10000 == 0:
                print(f'step: {self.step} | time span: {time.time() - start_time:.2f}')
                start_time = time.time()
        
        if save:
            torch.save(self.replay_buffer, os.path.join(self.save_buffer_dir, str(int(self.cfg.actor_index))+'.pt'))
            print('replay buffer saved')
        return self.replay_buffer
        
    
if __name__=='__main__':
    cfg = get_args()
    cfg.post_fix = '100k'
    for index in np.arange(100000, 2000000, 100000):
        cfg.actor_index = index
        print(f'actor index: {cfg.actor_index}')
        data_collector = actor_collect_data(cfg)
        data_collector.generate_data()



