import os
import torch
import numpy as np
from scipy.sparse import csr_matrix
from dataloader import LastFM, Loader

if __name__ == "__main__":
    # Path configurations
    gowalla_path = "../data/gowalla"
    amazon_path = "../data/amazon-book"
    yelp_path = "../data/yelp2018"

    print("\n" + "="*50 + "\n")

    # Validate Gowalla dataset
    gowalla_dataset = Loader(path=gowalla_path)
    validate_dataset(gowalla_dataset)

    # Validate Amazon-books dataset
    amazon_dataset = Loader(path=amazon_path)
    validate_dataset(amazon_dataset)

    # Validate yelp2018 dataset
    yelp_dataset = Loader(path=yelp_path)
    validate_dataset(yelp_dataset)