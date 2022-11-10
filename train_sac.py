#!/usr/bin/env python3
import numpy as np
from numpy.core.fromnumeric import argsort
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import argparse

from video import VideoRecorder
from logger import Logger
from buffer.replay_buffer import ReplayBuffer
import utils

import dmc2gym

from agent.sac import SACAgent
from agent.actor import DiagGaussianActor
from agent.critic import DoubleQCritic

os.environ["MUJOCO_GL"] = "osmesa"


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


def get_args():
    parser = argparse.ArgumentParser(description='SAC configs')
    parser.add_argument('--env', default='cheetah_run', type=str, 
                        help='task name')
    parser.add_argument('--experiment', default='test_exp', type=str, 
                        help='experiment name')
    parser.add_argument('--num-train-steps', default=1e6, type=float, 
                        help='name of training steps')
    parser.add_argument('--replay-buffer-capacity', default=1e6, type=float, 
                        help='replay buffer capacity')
    parser.add_argument('--num-seed-steps', default=5000, type=int, 
                        help='number of initial exploration steps')
    parser.add_argument('--eval-frequency', default=10000, type=int, 
                        help='evaluation frequency')
    parser.add_argument('--num-eval-episodes', default=10, type=int, 
                        help='number of evaluation episodes')
    parser.add_argument('--device', default='cuda', type=str, 
                        help='device')
    parser.add_argument('--log-frequency', default=10000, type=int, 
                        help='log frequency')
    parser.add_argument('--log-save-tb', default=True, type=bool, 
                        help='log save tensorboard')
    parser.add_argument('--save-video', default=False, type=bool, 
                        help='log save video')
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
    parser.add_argument('--save-frequency', default=100000, type=int, 
                        help='model parameters save frequency')
    return parser.parse_args()

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        
        self.save_actor_path = os.path.join(self.work_dir, 'logs', self.cfg.env, 'actor', 'params')
        self.save_critic_path = os.path.join(self.work_dir, 'logs', self.cfg.env, 'critic', 'params')
        self.save_alpha_path = os.path.join(self.work_dir, 'logs', self.cfg.env, 'alpha', 'params')
        self.save_actor_optim_path = os.path.join(self.work_dir, 'logs', self.cfg.env, 'actor', 'optim')
        self.save_critic_optim_path = os.path.join(self.work_dir, 'logs', self.cfg.env, 'critic', 'optim')
        self.save_alpha_optim_path = os.path.join(self.work_dir, 'logs', self.cfg.env, 'alpha', 'optim')
        
        if not os.path.exists(self.save_actor_path):
            os.makedirs(self.save_actor_path)
        if not os.path.exists(self.save_critic_path):
            os.makedirs(self.save_critic_path)
        if not os.path.exists(self.save_alpha_path):
            os.makedirs(self.save_alpha_path)
        if not os.path.exists(self.save_actor_optim_path):
            os.makedirs(self.save_actor_optim_path)
        if not os.path.exists(self.save_critic_optim_path):
            os.makedirs(self.save_critic_optim_path)
        if not os.path.exists(self.save_alpha_optim_path):
            os.makedirs(self.save_alpha_optim_path)

        self.logger = Logger(os.path.join(self.work_dir, 'logs', self.cfg.env),
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent_name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_env(cfg)
        
        action_range = [float(self.env.action_space.low.min()), 
                        float(self.env.action_space.high.max())]
        
        DoubleQCritic_cfg = {'obs_dim': self.env.observation_space.shape[0], 
                             'action_dim': self.env.action_space.shape[0], 
                             'hidden_dim': 1024, 
                             'hidden_depth': 2}
        GaussianActor_cfg = {'obs_dim': self.env.observation_space.shape[0], 
                             'action_dim': self.env.action_space.shape[0], 
                             'hidden_dim': 1024, 
                             'hidden_depth': 2, 
                             'log_std_bounds': [-5, 2]}
        
        self.agent = SACAgent(obs_dim=self.env.observation_space.shape[0], action_dim=self.env.action_space.shape[0], action_range=action_range, 
                              device=self.cfg.device, critic_model=DoubleQCritic, critic_cfg=DoubleQCritic_cfg, actor_model=DiagGaussianActor, 
                              actor_cfg=GaussianActor_cfg, discount=self.cfg.discount, init_temperature=self.cfg.init_temperature, alpha_lr=self.cfg.alpha_lr, 
                              alpha_betas=self.cfg.alpha_betas, actor_lr=self.cfg.actor_lr, actor_betas=self.cfg.actor_betas, 
                              actor_update_frequency=self.cfg.actor_update_frequency, critic_lr=self.cfg.critic_lr, critic_betas=self.cfg.critic_betas, 
                              critic_tau=self.cfg.critic_tau, critic_target_update_frequency=self.cfg.critic_target_update_frequency, 
                              batch_size=self.cfg.batch_size, learnable_temperature=self.cfg.learnable_temperature)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            # self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                # self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            # self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                # if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    if self.step > 0 and self.step % self.cfg.save_frequency == 0:
                        torch.save(self.agent.actor.state_dict(), os.path.join(self.save_actor_path, str(self.step)+'.pt'))
                        torch.save(self.agent.critic.state_dict(), os.path.join(self.save_critic_path, str(self.step)+'.pt'))
                        torch.save(self.agent.actor_optimizer.state_dict(), os.path.join(self.save_actor_optim_path, str(self.step)+'.pt'))
                        torch.save(self.agent.critic_optimizer.state_dict(), os.path.join(self.save_critic_optim_path, str(self.step)+'.pt'))
                        torch.save(self.agent.alpha, os.path.join(self.save_alpha_path, str(self.step)+'.pt'))
                        torch.save(self.agent.log_alpha_optimizer.state_dict(), os.path.join(self.save_alpha_optim_path, str(self.step)+'.pt'))
                        print(f'step: {self.step} | all models saved')

                self.logger.log('train/episode_reward', episode_reward, self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    cfg = get_args()
    main(cfg)
