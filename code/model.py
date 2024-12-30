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
                 config:dict, # 설정 정보(임베딩 차원, 레이어 수, 드롭아웃 확률 등)
                 dataset:BasicDataset): # 데이터셋 정보
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : BasicDataset = dataset
        self.__init_weight() # 가중치 초기화
        self.setup_layers() # disentangled graph convolutional encoder 설정

    def __init_weight(self):
        # 임베딩 초기화
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec'] # 임베딩 차원
        self.n_layers = self.config['lightGCN_n_layers'] # 레이어 수

        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        # 사용자와 아이템에 대한 임베딩 행렬 생성
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        

        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
            
            # 임베딩 초기값 설정
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1) # 평균 0, 표준편차 0.1인 정규분포로 초기화
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()

        # 희소 행렬로 저장된 그래프 구조 불러오기
        self.Graph = self.dataset.getSparseGraph()
        self.Edges = self.dataset.get_edges()
        print("self.Edges.shape: ", self.Edges.shape)
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

    def setup_layers(self):
        """
        Set up layers for Disentangled Graph Convolutional Encoder
        """
        self.init_disen = InitDisenLayer(self.latent_dim, self.latent_dim, 8, torch.relu)

        # print("save_txt")
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
    
    # neighbor aggregation 수행하는 함수
    def computer(self):
        """
        propagate methods for lightGCN
        """       

        # 초기 임베딩 결합
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        print('all_emb shape:', all_emb.shape)

        #   torch.split(all_emb , [self.num_users, self.num_items])

        # 초기 임베딩을 리스트에 추가하여 레이어별 결과를 저장
        embs = [all_emb]

        # 드롭아웃 여부에 따라 그래프를 수정
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph

        print(all_emb.shape)
        f_0 = self.init_disen(all_emb)
        print("***f_0 shape: ", f_0.shape)
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # 희소 행렬 곱셈으로 노드 임베딩을 업데이트
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())

        # 각 레이어의 임베딩을 평균 풀링으로 결합
        light_out = torch.mean(embs, dim=1)
        _users, _items = torch.split(embs, [self.num_users, self.num_items])

        # 최종 사용자 및 아이템 임베딩과 레이어별 임베딩 반환
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items, _users, _items
    
    # 예측 및 평가
    def getUsersRating(self, users):
        # if all_users is None or all_items is None:
        #     all_users, all_items, _, _ = self.computer()

        all_users, all_items, _, _ = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items

        # 사용자와 아이템 임베딩의 내적 계산 후 시그모이드 함수 적용
        rating = self.f(torch.matmul(users_emb, items_emb.t()))

        # 예측된 평가 점수 반환
        return rating
    
    # 훈련 중 임베딩을 추출하는 함수
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items, _, _ = self.computer()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    # 손실 함수 (BPR)
    def bpr_loss(self, users, pos, neg):
        # 사용자의 긍정적, 부정적 샘플의 임베딩 추출
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())

        # Normailization(L2 norm)
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    # 예측 함수
    def forward(self, users, items):
        # compute embedding
        all_users, all_items, _, _ = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
