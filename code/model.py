"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F
from models.encoder import Encoder
from models.decoder import PariwiseCorrelationDecoder


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def computer(self):
        return self.embedding_user.weight, self.embedding_item.weight
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        _users, _items = torch.split(embs, [self.num_users, self.num_items])
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items, _users, _items
    
    def getUsersRating(self, users):
        all_users, all_items, _, _ = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items, _, _ = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma

class DLightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(DLightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.K = self.config['num_factors']

        # self.factor_weights = nn.Parameter(torch.randn(self.K, self.latent_dim) * 0.1)
        # self.factor_bias = nn.Parameter(torch.zeros(self.K))

        # # Original embedding initialization
        # self.embedding_user = torch.nn.Embedding(
        #     num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # self.embedding_item = torch.nn.Embedding(
        #     num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        
        # Change embedding initialization for K factors
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim*self.K)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim*self.K)
        
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')

        # self.act_fn = self.config['act_fn']

        # Add learnable weight matrix Ws for factor correlation
        # self.Ws = nn.Parameter(torch.randn(self.K, self.K) * 0.1)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

    def reshape_embedding(self, embedding):
        # Reshape (N, K*dim) -> (N, K, dim)
        return embedding.view(embedding.shape[0], self.K, -1)

    def initial_disentangle(self, x):
        # x: input features
        factors = []
        for k in range(self.K):
            # FC layer for each factor
            factor = torch.matmul(x, self.factor_weights[k]) + self.factor_bias[k]
            # Apply activation (ReLU or tanh based on config)
            factor = self.act_fn(factor)
            # L2 normalize
            factor = F.normalize(factor, p=2, dim=-1)
            factors.append(factor)
        
        return torch.stack(factors, dim=1)  # Stack K factors

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        Disentangled Light Graph Convolution propagation
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]

        # # shape이 (N, dim)에서 (N, K, dim)으로 변경되어야 함
        # all_emb = all_emb.unsqueeze(1).expand(-1, self.K, -1)
        
        # # 초기 disentanglement 적용
        # all_emb = self.initial_disentangle(all_emb)  # shape: (N, K, dim)
        
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        # 각 factor별로 독립적인 propagation 수행
        final_embs = []
        layer_embs = []  # 각 레이어의 임베딩을 저장
        
        # for k in range(self.K):
        #     factor_emb = all_emb[:, k, :]  # k번째 factor 추출
        #     k_layer_embs = [factor_emb]
            
        #     for layer in range(self.n_layers):
        #         if self.A_split:
        #             temp_emb = []
        #             for f in range(len(g_droped)):
        #                 # neighborhood aggregation
        #                 aggregated = torch.sparse.mm(g_droped[f], factor_emb)
        #                 temp_emb.append(aggregated)
        #             factor_emb = torch.cat(temp_emb, dim=0)
        #         else:
        #             # neighborhood aggregation
        #             factor_emb = torch.sparse.mm(g_droped, factor_emb)
                    
        #         k_layer_embs.append(factor_emb)
                
        #     # Stack L layers for factor k
        #     k_stacked_embs = torch.stack(k_layer_embs, dim=1)  # (N, L+1, dim)
        #     layer_embs.append(k_stacked_embs)
            
        #     final_emb = torch.mean(k_stacked_embs, dim=1)  # final disentangled factor
        #     final_embs.append(final_emb)
        
        # # Combine all factors
        # layer_all_emb = torch.stack(layer_embs, dim=2)  # (N, L+1, K, dim)
        # final_all_emb = torch.stack(final_embs, dim=-1)  # (N, dim, K)
        
        # # Split users and items embeddings
        # users, items = torch.split(final_all_emb, [self.num_users, self.num_items])
        # users_layer_emb, items_layer_emb = torch.split(layer_all_emb, [self.num_users, self.num_items])
        
        # return users, items, users_layer_emb, items_layer_emb
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp = []
                    for k in range(self.K):
                        temp.append(torch.sparse.mm(g_droped[f], all_emb[:,:,k]))
                    temp_emb.append(torch.stack(temp, dim=1))
                all_emb = torch.cat(temp_emb, dim=0)
            else:
                temp = []
                for k in range(self.K):
                    temp.append(torch.sparse.mm(g_droped, all_emb[:,:,k])) 
                all_emb = torch.stack(temp, dim=1)
            embs.append(all_emb)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items, embs[:self.num_users], embs[self.num_items:]
    
    # def getUsersRating(self, users):
    #     """
    #     Calculate rating scores for given users with all items
    #     """
    #     all_users, all_items, _, _ = self.computer()
    #     users_emb = all_users[users]  # (batch_size, dim, K)
        
    #     # (batch_size, dim, K) x (n_items, dim, K) -> (batch_size, n_items, K)
    #     H_ui = torch.einsum('bdk,ndk->bnk', users_emb, all_items)
    #     weighted_H = H_ui * self.Ws.sum(-1)
    #     rating = weighted_H.sum(dim=-1)
        
    #     return self.f(rating)

    def getUsersRating(self, users):
        all_users, all_items, _, _ = self.computer()
        users_emb = all_users[users]
        rating = (users_emb.unsqueeze(1) * all_items.unsqueeze(0)).sum(dim=-1).sum(dim=-1)
        return self.f(rating)
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items, users_layer_emb, items_layer_emb = self.computer()
        
        users_emb = all_users[users]  
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        
        users_layer = users_layer_emb[users]
        pos_layer = items_layer_emb[pos_items] 
        neg_layer = items_layer_emb[neg_items]
        
        return users_emb, pos_emb, neg_emb, users_layer, pos_layer, neg_layer

    # def bpr_loss(self, users, pos, neg):
    #     users_emb, pos_emb, neg_emb, users_layer, pos_layer, neg_layer = self.getEmbedding(users.long(), pos.long(), neg.long())
        
    #     # Calculate pairwise correlations for positive and negative pairs
    #     pos_H_ui = torch.matmul(users_emb.transpose(1, 2), pos_emb)
    #     neg_H_ui = torch.matmul(users_emb.transpose(1, 2), neg_emb)
        
    #     # Calculate scores
    #     pos_scores = (pos_H_ui * self.Ws).sum(dim=[-2,-1])
    #     neg_scores = (neg_H_ui * self.Ws).sum(dim=[-2,-1])
        
    #     # BPR loss
    #     loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
    #     # L2 regularization
    #     reg_loss = (1/2)*(users_layer.norm(2).pow(2) + 
    #                         pos_layer.norm(2).pow(2) + 
    #                         neg_layer.norm(2).pow(2))/float(len(users))
        
    #     return loss, reg_loss

    def bpr_loss(self, users, pos, neg):
       (users_emb, pos_emb, neg_emb, 
       userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
       
       reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                      posEmb0.norm(2).pow(2) + 
                      negEmb0.norm(2).pow(2))/float(len(users))
       
       pos_scores = (users_emb * pos_emb).sum(dim=-1).sum(dim=-1)
       neg_scores = (users_emb * neg_emb).sum(dim=-1).sum(dim=-1)
       
       loss = torch.mean(F.softplus(neg_scores - pos_scores))
       return loss, reg_loss
       
    # def forward(self, users, items):
    #     """
    #     Calculate score for given user-item pairs using pairwise correlation decoder
    #     """
    #     # Get embeddings from computer()
    #     all_users, all_items, _, _ = self.computer()
        
    #     # Get specific user-item embeddings
    #     users_emb = all_users[users]  # shape: (batch_size, dim, K)
    #     items_emb = all_items[items]  # shape: (batch_size, dim, K)
        
    #     # Calculate pairwise correlations matrix (H_ui)
    #     # users_emb: (batch_size, dim, K), items_emb: (batch_size, dim, K)
    #     # H_ui: (batch_size, K, K)
    #     H_ui = torch.matmul(users_emb.transpose(1, 2), items_emb)
        
    #     # Calculate final scores using learnable weight matrix Ws
    #     # Ws: (K, K)
    #     # weighted_H: (batch_size, K, K)
    #     weighted_H = H_ui * self.Ws
        
    #     # Sum over both K dimensions to get final score
    #     # scores: (batch_size,)
    #     scores = weighted_H.sum(dim=[-2, -1])
        
    #     return scores
    def forward(self, users, items):
        all_users, all_items, _, _ = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        return (users_emb * items_emb).sum(dim=-1).sum(dim=-1)