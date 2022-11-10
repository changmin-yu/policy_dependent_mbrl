import argparse
import os
import numpy as np
import torch

from mbpo_dynamics_model import EnsembleDynamicsModel
from envs import make_dmc_env
from buffer.trajectory_replay_buffer import ReplayBuffer
import utils
from generate_trajectory_data import actor_collect_data
from model.network import PolicyEmbeddingGRU, PolicyEmbedding_VAE
from model.GRU_PE import compute_policy_embedding

os.environ['MUJOCO_GL'] = 'egl'

# CUDA_VISIBLE_DEVICES=0 python evaluate_trained_GRU_PE_extended_model.py --post-fix traj_buffer

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate trained MBPO dynamics model with extended policy embeddings given GRU features')
    parser.add_argument('--env', default='quadruped_run', type=str, 
                        help='task name')
    parser.add_argument('--device', default='cuda', type=str, 
                        help='device')
    parser.add_argument('--seed', default=1, type=int, 
                        help='random seed')
    parser.add_argument('--trajectory-index', default=1e5, type=float, 
                        help='trajectory index')
    parser.add_argument('--actor-index', default=1e5, type=float, 
                        help='trajectory index')
    parser.add_argument('--model-name', default='PE_GRU_extended_model', type=str, 
                        help='name prefix of the saved model filename')
    parser.add_argument('--batch-size', default=1024, type=int, 
                        help='batch size for model learning')
    parser.add_argument('--logdir-prefix', default='cached_policy_gen_data', type=str, 
                        help='logdir prefix')
    parser.add_argument('--num-collection-steps', default=5e4, type=float, 
                        help='number of evaluation transition tuples')
    parser.add_argument('--replay-buffer-capacity', default=5e4, type=float,
                        help='replay buffer capacity')
    parser.add_argument('--predict-done', default=False, type=bool, 
                        help='dynamics model predict done signal')
    parser.add_argument('--inc-var-loss', default=True, type=bool, 
                        help='including variance for computing MSE loss')
    parser.add_argument('--post-fix', default='', type=str, 
                        help='post fix for logdir')
    parser.add_argument('--save-policy-trajectory-prefix', default='cached_policy_gen_data', type=str, 
                        help='prefix for saving the policy generated trajectory data')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='printing progress in actor collecting data')
    parser.add_argument('--trajectory-maxlen', default=1000, type=int, 
                        help='max length of the buffer entries')
    parser.add_argument('--replay-buffer-capacity', default=25, type=int, 
                        help='number of trajectories (capacity) of the replay buffer')
    return parser.parse_args()

def main(cfg):
    utils.set_seed_everywhere(cfg.seed)
    
    pretrained_PE_GRU_fname = os.path.join('logs', cfg.env+'_'+cfg.post_fix, 'PE_GRU', 'pretrained_PE_GRU.pt')
    pretrained_PE_VAE_fname = os.path.join('logs', cfg.env+'_'+cfg.post_fix, 'PE_VAE', 'pretrained_PE_VAE.pt')
    
    pretrained_PE_GRU = torch.load(pretrained_PE_GRU_fname)
    pretrained_PE_VAE = torch.load(pretrained_PE_VAE_fname)
    
    assert isinstance(pretrained_PE_GRU, PolicyEmbeddingGRU)
    assert isinstance(pretrained_PE_VAE, PolicyEmbedding_VAE)
    
    env = make_dmc_env(cfg)
    device = torch.device(cfg.device)
    
    trained_model_fname = os.path.join(cfg.logdir_prefix, cfg.env+'_'+cfg.post_fix, 'trained_PE_GRU_extended_models', cfg.model_name+'.pt')    
    trained_model = torch.load(trained_model_fname)
       
    assert isinstance(trained_model, EnsembleDynamicsModel)
   
    ensemble_model = trained_model.ensemble_model
    
    network_size = trained_model.network_size
    
    dataCollector = actor_collect_data(cfg)
    eval_num = int(cfg.trajecotry_maxlen * cfg.replay_buffer_capacity)
    cfg.num_collection_steps = eval_num
    eval_buffer = dataCollector.generate_data(save=False)
    
    print(f'actor {cfg.actor_index} data collected')
    eval_obs, eval_action, eval_reward, eval_next_obs, _, _ = eval_buffer.sample(eval_num)
    
    eval_obs = eval_obs.cpu().numpy() # (N, L, H)
    eval_action = eval_action.cpu().numpy()
    eval_reward = eval_reward.cpu().numpy()
    eval_next_obs = eval_next_obs.cpu().numpy()
    
    policy_embeddings = compute_policy_embedding(eval_obs, eval_action, pretrained_PE_GRU, pretrained_PE_VAE)
    
    eval_inputs = np.concatenate((eval_obs.reshape(-1, eval_obs.shape[-1]), eval_action.reshape(-1, eval_action.shape[-1]), np.repeat(policy_embeddings, eval_obs.shape[0], axis=0)), axis=-1)
    delta_obs = (eval_next_obs - eval_obs).reshape(-1, eval_obs.shape[-1])
    eval_labels = np.concatenate((eval_reward.reshape(-1, 1), delta_obs), axis=-1)
    eval_labels = torch.from_numpy(eval_labels).float().to(device)
    eval_labels = eval_labels[None, :, :].repeat([network_size, 1, 1])
    
    eval_inputs = trained_model.scaler.transform(eval_inputs)
    ensemble_mean, ensemble_var = [], []
    
    if cfg.predict_done:
        ensemble_done = []
    loss_list = []
    
    batch_counter = 0
    with torch.no_grad():
        for i in range(0, eval_num, cfg.batch_size):
            input = torch.from_numpy(eval_inputs[i:min(i + cfg.batch_size, eval_num)]).float().to(device)
            b_mean, b_var, b_done = ensemble_model(input[None, :, :].repeat([network_size, 1, 1]), ret_log_var=True)
            if cfg.predict_done:
                ensemble_done.append(b_done.detach().cpu().numpy())
            
            labels = eval_labels[:, i:min(i+cfg.batch_size, eval_num), :]
            
            inv_var = torch.exp(-b_var)
            if cfg.inc_var_loss:
                mse_loss = torch.mean(torch.mean(torch.pow(b_mean-labels, 2)*inv_var, dim=-1), dim=-1)
            else:
                mse_loss = torch.mean(torch.mean(torch.pow(b_mean-labels, 2), dim=-1), dim=-1)
            batch_loss = torch.sum(mse_loss)
            batch_counter += 1
            loss_list.append(batch_loss.cpu().numpy())
    mean_eval_loss = np.mean(np.array(loss_list)[:-1])
    print(f'Trajectory index: {int(cfg.actor_index)} | Model index: {cfg.model_name} | Average loss: {mean_eval_loss:.2f}')
        

if __name__=='__main__':
    cfg = get_args()
    for actor_ind in np.arange(1e5, 2e6, 3e5):
            cfg.actor_index = actor_ind
            main(cfg)
