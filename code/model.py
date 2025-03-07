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
        assert self.latent_dim % self.K == 0, "latent_dim must be divided by num_factors K"

        # # Settings for FC Layer
        # self.fc_users = nn.ModuleList([
        #     nn.Linear(self.latent_dim, self.latent_dim // self.K)
        #     for _ in range(self.K)
        # ])
        # self.fc_items = nn.ModuleList([
        #     nn.Linear(self.latent_dim, self.latent_dim // self.K)
        #     for _ in range(self.K)
        # ])
        # self.act_fn = self.config['act_fn']

        # Original embedding initialization
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        
        # # Settings for Score
        # self.W_s = nn.Parameter(torch.empty(self.K, self.K))
        
        if self.config['pretrain'] == 0:
            # nn.init.normal_(self.embedding_user.weight, std=0.1)
            # nn.init.normal_(self.embedding_item.weight, std=0.1)

            # xavier uniform 초기화 적용
            nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)

            # for fc in self.fc_users:
            #     # nn.init.normal_(fc.weight, std=0.1)
            #     nn.init.xavier_uniform_(fc.weight, gain=1)
            #     nn.init.zeros_(fc.bias)
            # for fc in self.fc_items:
            #     # nn.init.normal_(fc.weight, std=0.1)
            #     nn.init.xavier_uniform_(fc.weight, gain=1)
            #     nn.init.zeros_(fc.bias)

            # nn.init.ones_(self.W_s)
            
            world.cprint('use Xavier uniform initilizer')
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
    
    # def disentangle_embedding(self, emb, fc_layers):
    #     """
    #     Disentangle embedding into K factors
    #     """
    #     batch_size = emb.size(0)
    #     factors = []

    #     for k in range(self.K):
    #         factor_k = fc_layers[k](emb) # [num_nodes, latent_dim // K]
    #         factor_k = self.act_fn(factor_k)
    #         factors.append(factor_k)

    #     return torch.stack(factors, dim=1) # [num_nodes, K, latent_dim // K]
    
    def computer(self):
        """
        Disentangled Light Graph Convolution propagation
        """       
        users_emb = self.embedding_user.weight.view(self.num_users, self.K, -1) # shape: [num_users, K, recdim]
        items_emb = self.embedding_item.weight.view(self.num_items, self.K, -1) # shape: [num_items, K, recdim]

        # users_emb = self.disentangle_embedding(
        #     self.embedding_user.weight,
        #     self.fc_users
        # )
        # items_emb = self.disentangle_embedding(
        #     self.embedding_item.weight,
        #     self.fc_items
        # )
        
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        # 각 factor별로 독립적인 propagation 수행
        final_users = []
        final_items = []
        all_users = []
        all_items = []
        
        for k in range(self.K):
            factor_users = users_emb[:, k, :] # [num_users, dim / k]
            factor_items = items_emb[:, k, :] # [num_items, dim / k]
            factor_emb = torch.cat([factor_users, factor_items]) # [num_users + num_items, dim / k]

            embs = [factor_emb]

            # Propagation
            for layer in range(self.n_layers):
                if self.A_split:
                    temp_emb = []
                    for f in range(len(g_droped)):
                        temp_emb.append(torch.sparse.mm(g_droped[f], factor_emb))
                    side_emb = torch.cat(temp_emb, dim=0)
                    factor_emb = side_emb
                else:
                    factor_emb = torch.sparse.mm(g_droped, factor_emb)
                embs.append(factor_emb)

            embs = torch.stack(embs, dim=1) # [num_users + num_items, L+1, dim / k]
            light_out = torch.mean(embs, dim=1) # [num_users + num_items, dim / k]

            factor_users, factor_items = torch.split(light_out, [self.num_users, self.num_items])
            _factor_users, _factor_items = torch.split(embs, [self.num_users, self.num_items])

            final_users.append(factor_users)
            final_items.append(factor_items)
            all_users.append(_factor_users)
            all_items.append(_factor_items)
        
        users = torch.stack(final_users, dim=1) # [num_users, K, dim / K]
        items = torch.stack(final_items, dim=1) # [num_items, K, dim / K]
        _users = torch.stack(all_users, dim=2) # [num_users, n_layers + 1, K, dim / K]
        _items = torch.stack(all_items, dim=2) # [num_items, n_layers + 1, K, dim / K]
        
        return users, items, _users, _items
    
    def getUsersRating(self, users):
        all_users, all_items, _, _ = self.computer()
        users_emb = all_users[users.long()] # [batch_size, K, dim]
        items_emb = all_items # [num_items, K, dim]

        # rating = self.f(torch.matmul(users_emb, items_emb.t()))

        factor_ratings = []
        for k in range(self.K):
            score_k = torch.matmul(users_emb[:, k, :], items_emb[:, k, :].t()) # [batch_size, num_items]
            factor_ratings.append(score_k)

        rating = torch.stack(factor_ratings).sum(dim = 0) # [batch_size, num_items]
        rating = self.f(rating)

        # # Compute H_ui
        # H_ui = torch.einsum('bkd,nmd->bknm', users_emb, items_emb)
        # score = torch.einsum('bknm,km->bn', H_ui, self.W_s)
        # rating = self.f(score)

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

        pos_scores = []
        neg_scores = []
        for k in range(self.K):
            pos_k = torch.sum(torch.mul(users_emb[:, k, :], pos_emb[:, k, :]), dim=1) # [batch_size]
            neg_k = torch.sum(torch.mul(users_emb[:, k, :], neg_emb[:, k, :]), dim=1) # [batch_size]
            pos_scores.append(pos_k)
            neg_scores.append(neg_k)
        
        pos_scores = torch.stack(pos_scores).sum(dim=0) # [batch_size]
        neg_scores = torch.stack(neg_scores).sum(dim=0) # [batch_size]
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        # H_ui_pos = torch.matmul(users_emb, pos_emb.transpose(2, 1)) # [batch_size, K, K]

        # H_ui_neg = torch.matmul(users_emb, neg_emb.transpose(2, 1)) # [batch_size, K, K]

        # # Score calculation
        # pos_scores = torch.sum(H_ui_pos * self.W_s, dim=(1, 2)) # [batch_size]
        # neg_scores = torch.sum(H_ui_neg * self.W_s, dim=(1, 2)) # [batch_size]

        # loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users] # [batch_size, K, dim / K]
        items_emb = all_items[items] # [batch_size, K, dim / K]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)

        factor_scores = []
        for k in range(self.K):
            score_k = torch.sum(torch.mul(users_emb[:, k, :], items_emb[:, k, :]), dim=1)
            factor_scores.append(score_k)
        
        gamma = torch.stack(factor_scores).sum(dim=0)

        # H_ui = torch.matmul(users_emb, items_emb.transpose(2, 1)) # [batch_size, K, K]
        # gamma = torch.sum(H_ui * self.W_s, dim=(1, 2)) # [batch_size]
        return gamma