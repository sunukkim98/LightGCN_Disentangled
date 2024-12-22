import torch
from torch import nn
from model import LightGCN  # LightGCN 코드가 포함된 파일에서 불러오기
from dataloader import BasicDataset  # 데이터셋 클래스를 정의한 파일에서 불러오기

# 설정 및 데이터셋 생성
config = {
    'latent_dim_rec': 64,       # 임베딩 차원
    'lightGCN_n_layers': 3,     # LightGCN 레이어 수
    'keep_prob': 0.8,           # 드롭아웃 확률
    'A_split': False,           # 희소 그래프 분할 여부
    'dropout': False,           # 드롭아웃 적용 여부
    'pretrain': 0               # 사전학습 사용 여부
}

# 가상의 사용자 및 아이템 수 설정
class FakeDataset(BasicDataset):
    def __init__(self, n_users, m_items):
        self.n_users = n_users
        self.m_items = m_items
        
    def getSparseGraph(self):
        # 임의의 희소 그래프 생성 (10x10 예제)
        indices = torch.LongTensor([[0, 1, 2], [2, 0, 1]])  # 예제 엣지
        values = torch.FloatTensor([1, 1, 1])  # 엣지 가중치
        size = torch.Size([10, 10])  # 그래프 크기
        graph = torch.sparse.FloatTensor(indices, values, size)
        return graph

dataset = FakeDataset(n_users=5, m_items=5)  # 사용자, 아이템 수 설정

# 모델 객체 생성
model = LightGCN(config, dataset)

# Embedding 크기 및 값 확인
users, items, _users, _items = model.computer()

print("Layer-wise embedding sizes:")
for i in range(len(_users)):  # 각 레이어별 임베딩 확인
    print(f"Layer {i}: {_users[i].shape}")
    print(_users[i])

print("\nFinal embedding sizes:")
print("Users:", users.shape)
print("Items:", items.shape)
