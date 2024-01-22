# anonymization.py

import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

#TODO: https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor

def anonymize_embeddings_random(embeddings, noise_factor=0.1):
    anonymized_embeddings = noise_factor * torch.randn_like(embeddings)
    return anonymized_embeddings


def anonymize_embeddings_laplace(embeddings, noise_scale=0.1):
    """
    Anonymize embeddings using Laplace noise.

    Parameters:
    - embeddings: PyTorch tensor, the original embeddings
    - noise_scale: float, scale parameter for Laplace distribution

    Returns:
    - PyTorch tensor, anonymized embeddings
    """
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    laplace_noise = torch.tensor(np.random.laplace(scale=noise_scale, size=embeddings.shape),
                                 dtype=torch.float32)

    # Add Laplace noise to each element in the embeddings tensor
    anonymized_embeddings = embeddings.copy()
    for i in range(embeddings.shape[0]):
        anonymized_embeddings[i] += laplace_noise[i]

    return anonymized_embeddings


def anonymize_embeddings_dp(embeddings, epsilon=0.1, device="cpu"):
    anonymized_embeddings = (embeddings + torch.tensor(np.random.normal(scale=epsilon, size=embeddings.shape),
                                                       dtype=torch.float32)).to(device)
    return anonymized_embeddings


def anonymize_embeddings_permutation(embeddings):
    permutation = torch.randperm(embeddings.shape[1])
    return embeddings[:, permutation]


def anonymize_embeddings_hashing(embeddings, salt="secret_salt"):
    hashed_embeddings = torch.tensor(np.vectorize(hash)(embeddings.cpu().numpy().astype(str) + salt), dtype=torch.long)
    return hashed_embeddings


def anonymize_embeddings_pca(embeddings, n_components=2):
    pca = PCA(n_components=n_components)
    return torch.tensor(pca.fit_transform(embeddings.cpu().numpy()), dtype=torch.float32)


def get_cluster_edges(cluster_embeddings):
    cluster_embeddings_np = cluster_embeddings.numpy()
    min_values = np.min(cluster_embeddings_np, axis=0)
    max_values = np.max(cluster_embeddings_np, axis=0)
    return min_values, max_values


def find_nearest_cluster(embedding, cluster_edges):
    distances = [np.linalg.norm(embedding - ((min_values + max_values) / 2)) for min_values, max_values in cluster_edges]
    nearest_cluster_idx = np.argmin(distances)
    return nearest_cluster_idx


def anonymize_embeddings_cluster_creator(train_embeddings, eps, min_samples):
    # Create a DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit the model to the original embeddings
    labels = dbscan.fit_predict(train_embeddings.numpy())  # Convert to NumPy array

    silhouette = silhouette_score(train_embeddings, labels)

    print(f"Epsilon: {eps}, Silhouette Score: {silhouette}")

    # Get the unique cluster labels
    unique_labels = np.unique(labels)
    print("Number of clusters:", len(unique_labels))

    # Initialize an array to store the cluster edges
    cluster_edges = []

    # Calculate the edges for each cluster
    for label in unique_labels:
        cluster_mask = (labels == label)
        cluster_embeddings = train_embeddings[cluster_mask]
        min_values, max_values = get_cluster_edges(cluster_embeddings)
        min_values_tensor = torch.tensor(min_values, dtype=torch.float32)
        max_values_tensor = torch.tensor(max_values, dtype=torch.float32)
        cluster_edges.append((min_values_tensor, max_values_tensor))

    return cluster_edges


#TODO: Add kNN anonymization method
#TODO: Cluster embeddings by label, then try cluster anonymization?
def anonymize_embeddings_cluster(cluster_edges, embeddings, noise_scale):
    anonymized_embeddings = []
    found_count = 0  # Counter for embeddings not found in any cluster
    not_found_count = 0  # Counter for embeddings not found in any cluster

    for embedding in embeddings:

        for cluster_edge in cluster_edges:
            min_values, max_values = cluster_edge
            # Check if all elements in embedding are greater than or equal to min_values
            condition_min = torch.all(embedding >= min_values)

            # Check if all elements in embedding are less than or equal to max_values
            condition_max = torch.all(embedding <= max_values)

            if condition_min and condition_max:
                # Test embedding is within the cluster, use cluster coordinates
                centroid = (min_values + max_values) / 2
                anonymized_embeddings.append(centroid)
                found_count += 1
                break

        if not condition_min and not condition_max:
            nearest_cluster_idx = find_nearest_cluster(embedding, cluster_edges)
            nearest_centroid = (cluster_edges[nearest_cluster_idx][0] + cluster_edges[nearest_cluster_idx][1]) / 2
            anonymized_embeddings.append(torch.tensor(nearest_centroid, dtype=torch.float32))
            not_found_count += 1

            #laplace_noise_vector = np.random.laplace(scale=noise_scale, size=embeddings.shape)
            #anonymized_embeddings.append(embedding + laplace_noise_vector)

    #TODO: Found in corectly labeled cluster
    # Add label to clusters?
    print(f"Embeddings found in clusters: {found_count}")
    print(f"Embeddings not found in clusters: {not_found_count}")

    anonymized_embeddings = torch.stack(anonymized_embeddings)


    return anonymized_embeddings