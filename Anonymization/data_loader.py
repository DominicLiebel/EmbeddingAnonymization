# data_loader.py

import torch
import numpy as np

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_npz_files(file_path, normalize=True):
    data = np.load(file_path)
    embeddings = torch.from_numpy(data['embeddings']).float()  # Convert to float
    labels = torch.from_numpy(data['labels']).long()  # Assuming labels are integers

    if normalize:
        # Normalize the embeddings along each feature dimension
        mean = embeddings.mean(dim=0)
        std = embeddings.std(dim=0)
        embeddings = (embeddings - mean) / std

    return embeddings, labels
