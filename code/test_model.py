import torch
from torch import nn
from model import LightGCN  # LightGCN 코드가 포함된 파일에서 불러오기
from dataloader import Loader  # 데이터셋 클래스를 정의한 파일에서 불러오기
import world

# 설정 및 데이터셋 생성
config = {
    'latent_dim_rec': 64,       # 임베딩 차원
    'lightGCN_n_layers': 3,     # LightGCN 레이어 수
    'keep_prob': 0.8,           # 드롭아웃 확률
    'A_split': False,           # 희소 그래프 분할 여부
    'dropout': False,           # 드롭아웃 적용 여부
    'pretrain': 0               # 사전학습 사용 여부
}

gowalla_path = "../data/gowalla"
gowalla_dataset = Loader(path=gowalla_path)

# 모델 객체 생성
model = LightGCN(config, gowalla_dataset)
model = model.to(world.device)

# Embedding 크기 및 값 확인
users, items, _users, _items = model.computer()

print("Layer-wise embedding sizes:")
for i in range(0, 5):  # 각 레이어별 임베딩 확인
    print(users[i].shape)

print("\nFinal embedding sizes:")
print("Users:", users.shape)
print("Items:", items.shape)
