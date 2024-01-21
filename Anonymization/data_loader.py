# data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from torch import from_numpy


def load_npz_files(file_path):
    data = np.load(file_path)
    embeddings = from_numpy(data['embeddings'])
    labels = data['labels']

    return embeddings, labels