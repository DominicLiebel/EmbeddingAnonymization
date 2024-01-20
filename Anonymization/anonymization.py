# anonymization.py

import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def anonymize_embeddings_random(embeddings, noise_factor=0.1):
    anonymized_embeddings = noise_factor * torch.randn_like(embeddings)
    return anonymized_embeddings


def anonymize_embeddings_laplace(embeddings, epsilon=0.1):
    """
    Anonymize embeddings using Laplace noise.

    Parameters:
    - embeddings: PyTorch tensor, the original embeddings
    - epsilon: float, scale parameter for Laplace distribution
    - device: str, device to place the noise tensor on ("cpu" or "cuda")

    Returns:
    - PyTorch tensor, anonymized embeddings
    """
    laplace_noise = torch.tensor(np.random.laplace(scale=epsilon, size=embeddings.shape),
                                 dtype=torch.float32)
    anonymized_embeddings = embeddings + laplace_noise
    return anonymized_embeddings


def anonymize_embeddings_dp(embeddings, epsilon=0.1, device="cpu"):
    anonymized_embeddings = (embeddings + torch.tensor(np.random.normal(scale=epsilon, size=embeddings.shape),
                                                       dtype=torch.float32)).to(device)
    return anonymized_embeddings


def anonymize_embeddings_permutation(embeddings):
    permutation = torch.randperm(embeddings.shape[1])
    return embeddings[:, permutation]


def anonymize_embeddings_hashing(embeddings, salt="secret_salt", percentage_hashed=100):
    """
    Anonymize embeddings using hashing.

    Parameters:
    - embeddings (torch.Tensor): Input embeddings to be anonymized.
    - salt (str): Salt value for the hash function.
    - percentage_hashed (int): Percentage of embeddings to be anonymized.

    Returns:
    - torch.Tensor: Anonymized embeddings.
    """
    # Validate the percentage_hashed value
    if not (0 <= percentage_hashed <= 100):
        raise ValueError("percentage_hashed should be between 0 and 100 (inclusive).")

    # Calculate the number of embeddings to be hashed
    num_hashed = int((percentage_hashed / 100) * len(embeddings))

    # Apply the hash function to the selected embeddings
    hashed_embeddings = torch.tensor(np.vectorize(hash)(embeddings.cpu().numpy().astype(str) + salt)[:num_hashed], dtype=torch.long)

    # Concatenate the original embeddings for the remaining percentage
    anonymized_embeddings = torch.cat([hashed_embeddings, embeddings[:len(embeddings) - num_hashed]])

    return anonymized_embeddings


def anonymize_embeddings_pca(embeddings, n_components=2):
    pca = PCA(n_components=n_components)
    return torch.tensor(pca.fit_transform(embeddings.cpu().numpy()), dtype=torch.float32)


def anonymize_embeddings_density_based(embeddings, eps=0.5, min_samples=5, noise_scale=0.01, device="cpu"):
    """
    Anonymize embeddings using density-based clustering.

    Parameters:
    - embeddings: PyTorch tensor or NumPy array, the original embeddings
    - eps: float, maximum distance between two samples for one to be considered as in the neighborhood of the other
    - min_samples: int, the number of samples in a neighborhood for a point to be considered as a core point
    - noise_scale: float, scale parameter for Laplace noise
    - device: str, device to place the noise tensor on ("cpu" or "cuda")

    Returns:
    - PyTorch tensor, anonymized embeddings
    """
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    # Perform density-based clustering using DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)

    # Assign a cluster label to each data point
    cluster_labels = db.labels_

    # Generate Laplace noise
    laplace_noise = np.random.laplace(scale=noise_scale, size=embeddings.shape)

    # Add noise to each cluster separately
    unique_labels = np.unique(cluster_labels)
    anonymized_embeddings = embeddings.copy()
    for label in unique_labels:
        cluster_indices = (cluster_labels == label)
        anonymized_embeddings[cluster_indices] += laplace_noise[cluster_indices]
    return anonymized_embeddings


def anonymize_embeddings(embeddings, args):
    if args.method == 'density_based':
        return anonymize_embeddings_density_based(embeddings, args.eps, args.min_samples,
                                                  args.noise_scale,)
    elif args.method == 'dp':
        return anonymize_embeddings_dp(embeddings, args.eps)
    elif args.method == 'laplace':
        return anonymize_embeddings_laplace(embeddings, args.eps)
    elif args.method == 'hashing':
        return anonymize_embeddings_hashing(embeddings, (1-args.noise_scale))
    else:
        raise ValueError("Unsupported anonymization method")
