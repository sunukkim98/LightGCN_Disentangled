import os
import torch
import numpy as np
from scipy.sparse import csr_matrix
from dataloader import LastFM, Loader


# Path configurations
gowalla_path = "../data/gowalla"
amazon_path = "../data/amazon-book"
yelp_path = "../data/yelp2018"

print("\n" + "="*50 + "\n")

# Validate Gowalla dataset
gowalla_dataset = Loader(path=gowalla_path)

# Validate Amazon-books dataset
amazon_dataset = Loader(path=amazon_path)

# Validate yelp2018 dataset
yelp_dataset = Loader(path=yelp_path)