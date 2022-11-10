import torch
from torch import nn
from model.network import PolicyEmbeddingGRU, PolicyEmbedding_VAE

def compute_policy_embedding(state_batch, action_batch, trained_PE_GRU, device):#, trained_PE_VAE): 
    #, reward_batch=None, predict_reward=False):
    # num_samples, seqlen = state_batch.size(0), state_batch.size(1)
    
    trained_PE_GRU.eval()
    # trained_PE_VAE.eval()
    
    gru_inp = torch.cat((torch.FloatTensor(state_batch), torch.FloatTensor(action_batch)), dim=-1).to(device)
    _, gru_hidden = trained_PE_GRU(gru_inp, return_hidden=True)
    # if not predict_reward:
    #     assert reward_batch is not None, "reward labels not available!"
    
    # vae_inp = gru_hidden.transpose(1, 2).view(seqlen, num_samples, -1).transpose(0, 1)
    # _, embed_mean, _ = trained_PE_VAE(vae_inp)
    # policy_embedding = torch.mean(embed_mean, dim=0)
    # policy_embedding = torch.mean(gru_hidden, dim=0)
    return gru_hidden