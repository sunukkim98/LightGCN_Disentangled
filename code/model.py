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
from models.encoder import InitDisenLayer
from models.encoder import Encoder
# from models.dlgconv import DLGConv
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self, 
                 config: dict, 
                 dataset):
        """
        LightGCN with Disentangled Embeddings

        Args:
            config: configuration dictionary
            dataset: dataset object
        """
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset

        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.n_layers = config['lightGCN_n_layers']
        self.keep_prob = config['keep_prob']
        self.A_split = config['A_split']

        self.encoder = Encoder(
            in_dim=self.latent_dim,
            out_dim=self.latent_dim,
            num_factors=config['num_factors'],
            num_layers=self.n_layers,
            aggr_type=config['aggr_type'],
            act_fn=torch.relu
        )
        self.decoder = PariwiseCorrelationDecoder(
            num_factors=config['num_factors'],
            out_dim=self.latent_dim,
            num_users=self.num_users,
            num_items=self.num_items
        )
        self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)

        if config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(config['item_emb']))

        self.Graph = dataset.getSparseGraph()
        self.edge_list = self.get_edge_list(self.Graph)

    def get_edge_list(self, sparse_adj):
        """
        Convert sparse adjacency matrix to edge list in format [[user1, item1], [user2, item2], ...]
        
        Args:
            sparse_adj (torch.sparse.FloatTensor): Sparse adjacency matrix in format [I, R; R^T, I]
            
        Returns:
            torch.Tensor: A tensor of shape (N, 2) containing [user_id, item_id] pairs
        """
        # Get indices from sparse tensor
        indices = sparse_adj.indices()
        
        # Matrix size
        n = sparse_adj.size(0)
        n_users = self.dataset.n_users # Number of users
        
        # Filter edges between users and items (top-right quadrant)
        mask = (indices[0] < n_users) & (indices[1] >= n_users)
        user_item_indices = indices[:, mask]
        
        # Convert item indices to start from 0
        users = user_item_indices[0]  # These are already 0-based
        items = user_item_indices[1] - n_users  # Subtract offset to make 0-based
        
        # Stack users and items to create edge list
        edge_list = torch.stack([users, items], dim=1)
        
        return edge_list

    def forward(self, users, items):
        """
        Forward pass to compute scores.
        """
        X = torch.cat([self.embedding_user.weight, self.embedding_item.weight], dim=0)
        Z = self.encoder(X, self.edge_list)
        scores = self.decoder(Z, users, items)
        return scores

    def getUsersRating(self, users):
        """
        Compute user-item scores for recommendation.
        """
        Z = self.get_embedding()
        all_users, all_items = torch.split(Z, [self.num_users, self.num_items], dim=0)
        users_emb = all_users[users]
        items_emb = all_items
        scores = self.decoder.calculate_score(users_emb, items_emb)
        return scores

    def getEmbedding(self, users, pos_items, neg_items):
        """
        Extract embeddings for users and items.
        """
        Z = self.get_embedding()
        all_users, all_items = torch.split(Z, [self.num_users, self.num_items], dim=0)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        """
        Calculate BPR loss for training.
        """
        Z = self.get_embedding()
        loss, reg_loss = self.decoder.bpr_loss(Z, users, pos, neg)
        return loss, reg_loss

    def get_embedding(self):
        """
        Compute embeddings for evaluation.
        """
        X = torch.cat([self.embedding_user.weight, self.embedding_item.weight], dim=0)
        Z = self.encoder.forward(X, self.edge_list)
        return Z
