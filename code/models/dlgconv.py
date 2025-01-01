import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min
from torch_scatter.composite import scatter_softmax

class DLGConv(nn.Module):
    def __init__(self, 
                 in_dim,
                 out_dim,
                 
                 num_factors,
                 act_fn,
                 aggr_type):
        """
        Build a DLGConv layer
        
        Args:
            in_dim: input embedding dimension (d_(l-1))
            out_dim: output embedding dimension (d_l)
            num_factors: number of factors (K)
            act_fn: torch activation function 
            aggr_type: aggregation type ['sum', 'mean', 'max', 'attn']
        """
        super(DLGConv, self).__init__()
        
        # 입력 파라미터 설정
        self.d_in = in_dim
        self.d_out = out_dim
        self.K = num_factors
        self.num_neigh_type = 1 # 무방향 그래프이므로 이웃 타입의 수는 1
        self.act_fn = act_fn
        self.aggr_type = aggr_type

        # 레이어 초기화 함수 호출
        self.setup_layers()

    ######################################v_0구현시 생략######################################
    def setup_layers(self):
        if self.aggr_type == 'attn':
            self.disen_attn_weights = nn.ModuleList()
            for _ in range(self.num_neigh_type):
                disen_attn_w = nn.Parameter(torch.empty(self.K, 2*self.d_in//self.K))
                torch.nn.init.xavier_uniform_(disen_attn_w)
                self.disen_attn_weights.append(disen_attn_w)
        
        elif self.aggr_type == 'max':
            self.disen_max_weights = nn.ParameterList()
            for _ in range(self.num_neigh_type):
                disen_max_w = nn.Parameter(torch.empty(self.K, self.d_in//self.K, self.d_in//self.K))
                torch.nn.init.xavier_uniform_(disen_max_w)
                self.disen_max_weights.append(disen_max_w)
            
        # 노드 임베딩 업데이트를 위한 가중치와 편향 초기화
        self.disen_update_weights = nn.Parameter(torch.empty(self.K, (self.num_neigh_type+1)*self.d_in//self.K, self.d_out//self.K))
        self.disen_update_bias = nn.Parameter(torch.zeros(1, self.K, self.d_in//self.K))
        torch.nn.init.xavier_uniform_(self.disen_update_weights)
    #########################################################################################
    
    def forward(self, f_in, edges):
        """
        For each factor, aggregate the neighbors' embedding and update the anode embeddings using aggregated messages and before layer embedding
        
        Args:
            f_in: disentangled node embeddings of before layer
            edges: collection of edge lists
        Returns:
            f_out: aggregated disentangled node embeddings
        """
        
        # m_agg = [] # 집계된 메시지를 저장할 리스트
        # m_agg.append(f_in) # 이전 임베딩 추가
        
        # for neigh_type_idx, edges in enumerate(edges_each_type):
        #     m = self.aggregate(f_in, edges, neigh_type_idx=neigh_type_idx)
        #     m_agg.append(m)

        # 무방향 그래프에서는 이웃 타입이 1개이므로, 반복문 생략
        m = self.aggregate(f_in, edges)
        # m_agg.append(m)

        # f_out = self.update(m_agg) # v_0에서는 update 생략
        # f_out = self.normalize(m_agg) # 정규화 수행
        f_out = F.normalize(m)
        print("dlgconv f_out.shape: ", f_out.shape)
        return f_out

    def aggregate(self, f_in, edges):
        """
        Aggregate messsages for each factor by considering neighbor type and aggregator type
        torch_scatter: https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
        
        Args:
            f_in: disentangled node embeddings of before layer
            edges: edge list of neighbors
            
        Returns:
            m: aggregated meesages of neighbors
        """
        src, dst = edges[:, 0], edges[:, 1]
        
        out = f_in.new_zeros(f_in.shape)
        
        if self.aggr_type == 'sum':
            m = scatter_add(f_in[dst], src, dim=0, out=out)
            
        ######################################v_0구현시 생략######################################
        elif self.aggr_type == 'attn':
            f_edge = torch.concat([f_in[src], f_in[dst]], dim=2)
            score = F.leaky_relu(torch.einsum("ijk,jk->ij", f_edge, self.disen_attn_weights[neigh_type_idx])).unsqueeze(2) 
            norm = scatter_softmax(score, src, dim=0) 
            m = scatter_add(f_in[dst]*norm, src, dim=0, out=out)
            
        elif self.aggr_type == 'mean':
            m = scatter_mean(f_in[dst], src, dim=0, out=out)
            
        elif self.aggr_type == 'max':
            f_in_max = torch.einsum("ijk,jkl->ijl", f_in, self.disen_max_weights[neigh_type_idx])
            m = scatter_max(f_in_max[dst], src, dim=0, out=out)[0]
        #########################################################################################
        
        return m