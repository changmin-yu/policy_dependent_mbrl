import torch
from torch import nn
from torch.nn.modules.linear import Linear
from envs import make_dmc_env

class PolicyEmbeddingGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, bidirectional=False, repr_ratio=0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, bidirectional=bidirectional)
        self.linear_proj = nn.Linear(hidden_dim, output_dim)
        self.repr_ratio = repr_ratio
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)

    def forward(self, x, hidden=None, return_hidden=False):
        x = x.permute(1, 0, 2) # (B, L, D) -> (L, B, D)
        
        seq_len, batch_size = x.size(0), x.size(1)
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        if not return_hidden:
            y, _ = self.gru(x, hidden)
            out = self.linear_proj(y.view(-1, y.size(-1)))
            out = out.view(seq_len, batch_size, -1).transpose(0, 1)
            return out
        output_list = []
        hidden_repr = torch.zeros_like(hidden)
        for i in range(seq_len):
            y, hidden = self.gru(x[i][None, ...], hidden)
            y = self.linear_proj(y)
            output_list.append(y)
            # if return_hidden:
            #     hidden_list.append(hidden)
            hidden_repr = hidden_repr * (1-self.repr_ratio) + self.repr_ratio * hidden
        output_list = torch.cat(output_list, dim=0).transpose(0, 1)
        if self.num_layers == 1 and not self.bidirectional:
            hidden_repr = hidden_repr.squeeze(0)
        # hidden_list = torch.cat(hidden_list, dim=0)
        # return (output_list, hidden_list)
        return (output_list, hidden_repr)
    
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class FCEncoder(nn.Module):
    def __init__(self, seqlen, feature_dim, latent_dim, hidden_dims=[256, 256], encoder_dropout=[0.2, 0.2]):
        super().__init__()
        self.network = nn.ModuleList()
        self.network.append(nn.Flatten())
        
        self.latent_dim = latent_dim

        input_dim = seqlen * feature_dim

        for d0, d1, d2 in zip([input_dim]+hidden_dims[:-1], hidden_dims, encoder_dropout):
            self.network.extend([nn.Linear(d0, d1), nn.ReLU()])
            if d2 != 0.0:
                self.network.extend([nn.Dropout(d2)])
                
        self.network.extend([nn.Linear(hidden_dims[-1], latent_dim*2)])
        self.network = nn.Sequential(*self.network)
    
    def forward(self, x):
        x = self.network(x)
        mean, log_var = torch.split(x, self.latent_dim, dim=-1)
        return mean, log_var
    
class FCDecoder(nn.Module):
    def __init__(self, latent_dim, seqlen, feature_dim, hidden_dims=[256, 256], decoder_dropout=[0.2, 0.2]):
        super().__init__()
        self.network = nn.ModuleList()
        in_dim = latent_dim
        
        for d0, d1, d2 in zip([latent_dim]+hidden_dims[:-1], hidden_dims, decoder_dropout):
            self.network.extend([nn.Linear(d0, d1), nn.ReLU(), nn.Dropout(d2)])
        
        self.network.extend([nn.Linear(hidden_dims[-1], seqlen*feature_dim), View((-1, seqlen, feature_dim))])
        self.network = nn.Sequential(*self.network)
    
    def forward(self, x, conditions=None):
        x = self.network(x)
        return x
    
class PolicyEmbedding_VAE(nn.Module):
    def __init__(self, seqlen, feature_dim, latent_dim=64, device=torch.device('cuda'), 
                 encoder_kwargs={}, decoder_kawrgs={}):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        
        self.encoder = FCEncoder(seqlen, feature_dim, latent_dim, **encoder_kwargs).to(device)
        self.decoder = FCDecoder(latent_dim, seqlen, feature_dim, **decoder_kawrgs).to(device)
        
    def forward(self, x):
        post_mean, post_logvar = self.encoder(x)
        z = self.reparameterise(post_mean, post_logvar)
        x_decoded = self.decoder(z)
        return x_decoded, post_mean, post_logvar
    
    def reparameterise(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(self.device)
        samples = self.decoder(z)
        return samples
    
    def generate(self, x, **kwargs):
        return self.forward(x)[0]