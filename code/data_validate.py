import os
import torch
import numpy as np
from scipy.sparse import csr_matrix
from dataloader import LastFM, Loader

def validate_dataset(dataset):
    # print(f"Validating dataset: {dataset.}")

    # Validate number of users and items
    print("Number of users:", dataset.n_users)
    print("Number of items:", dataset.m_items)

    # Validate train data size
    print("Training data size:", dataset.trainDataSize)

    # Validate sparsity of the dataset
    sparsity = dataset.trainDataSize / (dataset.n_users * dataset.m_items)
    print(f"Sparsity: {sparsity:.4f}")

    # Validate test dictionary
    test_dict = dataset.testDict
    if test_dict:
        print(f"Number of users with test data: {len(test_dict)}")
    else:
        print("No test data available.")

    # Validate all positive items
    all_pos = dataset.allPos
    print(f"Number of users with positive items: {len(all_pos)}")

    # Validate sparse graph
    sparse_graph = dataset.getSparseGraph()
    if sparse_graph is not None:
        print("Sparse graph successfully generated.")
    else:
        print("Sparse graph generation failed.")

    print("Validation complete.")

if __name__ == "__main__":
    # Path configurations
    gowalla_path = "../data/gowalla"
    amazon_path = "../data/amazon-book"
    yelp_path = "../data/yelp2018"

    print("\n" + "="*50 + "\n")

    # Validate Gowalla dataset
    gowalla_dataset = Loader(path=gowalla_path)
    validate_dataset(gowalla_dataset)

    amazon_dataset = Loader(path=amazon_path)
    validate_dataset(amazon_dataset)

    yelp_dataset = Loader(path=yelp_path)
    validate_dataset(yelp_dataset)