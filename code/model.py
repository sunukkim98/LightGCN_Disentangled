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
    def __init__(self, config: dict, dataset: BasicDataset):
        super(DLightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.n_factors = self.config['n_factors']  # 추가: 디스엔탱글된 요인의 수
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.__init_weight()
        self.Graph = self.dataset.getSparseGraph()

    def __init_weight(self):
        self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)
        
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initializer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretrained data')
        
        # 추가: 각 요인에 대한 선형 변환 레이어
        self.factor_k = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.n_factors)])
        # 추가: 페어와이즈 상관 디코더를 위한 가중치 행렬
        self.W_s = nn.Parameter(torch.Tensor(self.n_factors, self.n_factors))
        nn.init.xavier_uniform_(self.W_s)

    def __dropout(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def computer(self):
        # LightGCN의 원래 구조를 유지
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        
        if self.config['dropout']:
            if self.training:
                print("dropping")
                g_droped = self.__dropout(self.Graph, self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    # 추가: 디스엔탱글된 요인을 얻는 메서드
    def get_disentangled_factors(self, embeddings):
        factors = []
        for k in range(self.n_factors):
            f_k = F.relu(self.factor_k[k](embeddings))
            f_k = F.normalize(f_k, p=2, dim=1)
            factors.append(f_k)
        return factors

    # 변경: 페어와이즈 상관 디코더를 사용하는 forward 메서드
    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        
        user_factors = self.get_disentangled_factors(users_emb)
        item_factors = self.get_disentangled_factors(items_emb)
        
        H_ui = torch.zeros(users.size(0), self.n_factors, self.n_factors).to(users.device)
        for i in range(self.n_factors):
            for j in range(self.n_factors):
                H_ui[:, i, j] = torch.sum(user_factors[i] * item_factors[j], dim=1)
        
        scores = torch.sum(H_ui * self.W_s, dim=(1, 2))
        return scores

    # 변경: BPR 손실 계산 메서드
    def bpr_loss(self, users, pos, neg):
        pos_scores = self.forward(users, pos)
        neg_scores = self.forward(users, neg)
        
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        reg_loss = (1/2)*(self.embedding_user(users).norm(2).pow(2) + 
                          self.embedding_item(pos).norm(2).pow(2) +
                          self.embedding_item(neg).norm(2).pow(2))/float(len(users))
        
        return loss, reg_loss

    # # 추가: 판별 손실 계산 메서드
    # def get_discriminative_loss(self, embeddings):
    #     factors = self.get_disentangled_factors(embeddings)
    #     disc_loss = 0
    #     for k, factor in enumerate(factors):
    #         logits = F.linear(factor, self.factor_k[k].weight)
    #         labels = torch.full((factor.size(0),), k, dtype=torch.long, device=factor.device)
    #         disc_loss += F.cross_entropy(logits, labels)
    #     return disc_loss / self.n_factors

    # # 변경: 최종 손실 계산 메서드
    # def calc_loss(self, users, pos, neg):
    #     bpr_loss, reg_loss = self.bpr_loss(users, pos, neg)
    #     all_users, all_items = self.computer()
    #     # disc_loss_users = self.get_discriminative_loss(all_users[users])
    #     # disc_loss_items = self.get_discriminative_loss(all_items[pos])
        
    #     # loss = bpr_loss + self.config['lambda_reg'] * reg_loss + \
    #     #        self.config['lambda_disc'] * (disc_loss_users + disc_loss_items)
        
    #     loss = bpr_loss + self.config['lambda_reg'] * reg_loss
        
    #     return loss

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.forward(users, torch.arange(self.num_items).to(users.device))
        return rating