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
        
        # Classifier for prediction
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
        items_emb_t = torch.transpose(items_emb, 1, 2)
        H = torch.bmm(users_emb, items_emb_t)
        score = self.predictor(H.reshape(H.size(0), -1))
        return score.squeeze()