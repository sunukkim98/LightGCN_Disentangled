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
        num_users = users_emb.shape[0]
        num_items = items_emb.shape[0]
        
        # Reshape users_emb to [num_users, 1, 8, 8] and expand to [num_users, num_items, 8, 8]
        users_emb_expanded = users_emb.unsqueeze(1)
        
        batch_size = num_users
        all_scores = []
        
        for i in range(0, num_items, batch_size):
            end_idx = min(i + batch_size, num_items)
            # Take a batch of items [batch_size, 8, 8]
            items_batch = items_emb[i:end_idx]
            
            # Transpose item embeddings [batch_size, 8, 8] -> [8, batch_size, 8]
            items_batch_t = torch.transpose(items_batch, 0, 1)
            items_batch_t = torch.transpose(items_batch_t, 1, 2)
            
            # Calculate batch scores using bmm
            # users_emb: [num_users, 8, 8], items_batch_t: [8, batch_size, 8]
            H = torch.bmm(users_emb, items_batch_t)
            
            # Get scores for the batch
            batch_scores = self.predictor(H.reshape(H.size(0), -1))
            all_scores.append(batch_scores)
        
        # Concatenate all batch scores
        rating = torch.cat(all_scores, dim=1)
        return rating.squeeze()