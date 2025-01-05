import torch
import torch.nn as nn
import torch.nn.functional as F

class PariwiseCorrelationDecoder(nn.Module):
    def __init__(self, num_factors, out_dim, num_users, num_items):
        """
        Build a pairwise factor correlation decoder with BPR loss
        """
        super(PariwiseCorrelationDecoder, self).__init__()

        self.K = num_factors
        self.d_out = out_dim
        self.num_users = num_users
        self.num_items = num_items
        
        self.predictor = nn.Sequential(
            nn.Linear(self.K**2, 1, bias=False),
        )
        torch.nn.init.xavier_uniform_(self.predictor[0].weight)

    def getEmbedding(self, Z, users, pos_items, neg_items):
        """
        Extract embeddings for users and items
        """
        all_users, all_items = torch.split(Z, [self.num_users, self.num_items], dim=0)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        return users_emb, pos_emb, neg_emb
    
    def forward(self, users_emb, items_emb):
        """
        Calculate correlation score for user-item pairs
        """
        print("users_emb.shape: ", users_emb.shape)
        print("items_emb.shape: ", items_emb.shape)
        items_emb_t = torch.transpose(items_emb, 1, 2)
        H = torch.bmm(users_emb, items_emb_t)
        score = self.predictor(H.reshape(H.size(0), -1))
        return score.squeeze()
    
    def getUsersRating(self, users_emb, items_emb):
        """
        Calculate rating scores for all user-item pairs
        users_emb: [num_users, 8, 8]
        items_emb: [num_items, 8, 8]
        """
        num_users = users_emb.shape[0]  # batch_size로 사용
        num_items = items_emb.shape[0]
        
        all_scores = []
        
        for i in range(0, num_items, num_users):
            end_idx = min(i + num_users, num_items)
            items_batch = items_emb[i:end_idx]
            
            # 현재 배치의 실제 크기
            current_batch_size = end_idx - i
            
            # 현재 배치가 num_users보다 작은 경우 패딩
            if current_batch_size < num_users:
                # 마지막 배치를 num_users 크기에 맞게 패딩
                padding = items_emb[0:num_users-current_batch_size]
                items_batch = torch.cat([items_batch, padding], dim=0)
            
            # Transpose item embeddings for correlation calculation
            items_batch_t = items_batch.transpose(1, 2)
            
            # Calculate correlation matrix
            H = torch.bmm(users_emb, items_batch_t)
            
            # Reshape H to match predictor input dimension (self.K**2 = 64)
            score = self.predictor(H.reshape(num_users, -1))
            
            # 패딩된 부분 제거
            if current_batch_size < num_users:
                score = score[:, :current_batch_size]
                
            all_scores.append(score)
        
        # Concatenate all batch scores
        rating = torch.cat(all_scores, dim=1)
        return rating.squeeze()