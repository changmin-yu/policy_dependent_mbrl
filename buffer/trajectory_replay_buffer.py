import numpy as np
import torch


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, maxlen):
        self.capacity = capacity
        self.device = device
        self.maxlen = maxlen

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, maxlen, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, maxlen, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, maxlen, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, maxlen, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, maxlen, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, maxlen, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        trajectory_len = obs.shape[0]
        trajectory_len = min(trajectory_len, self.maxlen)
        np.copyto(self.obses[self.idx, :trajectory_len, :], obs)
        np.copyto(self.actions[self.idx, :trajectory_len, :], action)
        np.copyto(self.rewards[self.idx, :trajectory_len, :], reward)
        np.copyto(self.next_obses[self.idx, :trajectory_len, :], next_obs)
        np.copyto(self.not_dones[self.idx, :trajectory_len, :], 1-done)
        np.copyto(self.not_dones_no_max[self.idx, :trajectory_len, :], 1-done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
    
    @property
    def transition_buffer(self):
        obses = np.reshape(self.obses, -1, self.obses.shape[-1])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        next_obses = self.next_obses.reshape(-1, self.next_obses.shape[-1])
        rewards = self.rewards.reshape(-1, self.rewards.shape[-1])
        not_dones = self.not_dones.reshape(-1, self.not_dones.shape[-1])
        not_dones_no_max = self.not_dones_no_max.reshape(-1, self.not_dones_no_max.shape[-1])
        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max