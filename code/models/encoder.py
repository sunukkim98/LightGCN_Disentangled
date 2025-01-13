import torch
import torch.nn.functional as F
import torch.nn as nn
from .dlgconv import DLGConv

class InitDisenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_factors, act_fn):
        """
        Build a initial disentangling layer

        Args:
            in_dim: input features dimension (d_in)
            out_dim: output embedding dimension (d_0)
            num_factors: number of factors (K)
            act_fn: torch activation function
        """
        super(InitDisenLayer, self).__init__()
        self.d_in = in_dim
        self.d_0 = out_dim
        self.K = num_factors
        self.act_fn = act_fn
        self.setup_layers()

    def setup_layers(self):
        """
        Set up layers 
        """
        self.disen_weights = nn.Parameter(torch.randn(self.K, self.d_0, self.d_in//self.K))
        self.disen_bias = nn.Parameter(torch.zeros(1, self.K, self.d_in//self.K))
        torch.nn.init.xavier_uniform_(self.disen_weights)

    def forward(self, X):
        """
        Disentangle input features into K vectors
        
        The einsum function used in forward operates as follows:
            (Input Features (N, d_in) X FC Weights (K, d_in, d_0)) + FC biases (1, d_in, d_0)
                -> Initial disentangled embedding (N, K, d_0)
        It is equivalent to pass input features into K fully-connected layers
        
        Args:
            X: input node features

        Returns:
            f_0: Initial disentangled node embedding
        """
        f_0 = torch.einsum("ij,kjl->ikl", X, self.disen_weights) + self.disen_bias
        # # 임베딩 변환 과정 제거
        # if self.K == 1:
        #     f_0 = X.unsqueeze(1)
        # else:
        #     N = X.size(0)
        #     f_0 = X.view(N, self.K, -1)

        f_0 = F.normalize(self.act_fn(f_0))
        f_0 = self.act_fn(f_0)
        return f_0
    
class Encoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_factors,
        num_layers,
        aggr_type,
        act_fn
    ):
        """
        Build a Encoder

        Args:
            in_dim: input feature dimension (d_in)
            out_dim: output embedding dimension (d_out)
            num_factors: number of factors (K)
            num_layers: number of layers (L)
            aggr_type: aggregation type ['sum', 'mean', 'max', 'attn']
            act_fn: torch activation function
        """

        super(Encoder, self).__init__()
        self.d_in = in_dim
        self.d_out = out_dim
        self.K = num_factors
        self.L = num_layers
        self.act_fn = act_fn
        self.aggr_type = aggr_type

        self.setup_layers()


    def setup_layers(self):
        """
        Set up layers for Encoder
        """
        self.init_disen = InitDisenLayer(self.d_in, self.d_out, self.K, self.act_fn)
        self.conv_layers = nn.ModuleList()
        for _ in range(self.L):
            self.conv_layers.append(DLGConv(self.d_out, self.d_out, self.K, self.act_fn, self.aggr_type))


    def forward(self, X, edges):
        """
        Generate disentangled node representation
        Args:
            X: input node features
            edges: collection of edge lists

        Returns:
            Z: disentangled node embeddings
        """
        f_0 = self.init_disen(X)

        f_emb = [f_0]
        f_l = f_0
        for ldn_conv in self.conv_layers:
            f_l = ldn_conv(f_l, edges)
            f_emb.append(f_l)
        
        f_emb = torch.stack(f_emb, dim=1)
        f_out = torch.mean(f_emb, dim=1)

        _Z = f_emb
        Z = f_out
        return Z, _Z# (N, K, d_out/K)