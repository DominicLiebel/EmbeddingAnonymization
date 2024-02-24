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


def anonymize_embeddings_uniform(embeddings, epsilon=0.1, device="cpu"):
    """
    Anonymize embeddings using Differential Privacy (DP) with uniform noise.

    This function adds uniform noise to the embeddings to protect privacy while
    preserving utility. The amount of noise is controlled by the `epsilon` parameter,
    which determines the privacy-utility trade-off.

    Parameters:
    - embeddings: PyTorch tensor, the original embeddings.
    - epsilon: float, the parameter controlling the privacy guarantee of DP.
        Lower values provide stronger privacy but introduce more noise.
    - device: str, the device on which the computation will be performed (e.g., "cpu" or "cuda").

    Returns:
    - PyTorch tensor, the anonymized embeddings.

    Raises:
    - ValueError: If `epsilon` is negative.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    # Generate uniform noise and move it to the specified device
    noise = torch.rand_like(embeddings) * epsilon * 2 - epsilon  # Scale to [-epsilon, epsilon]
    anonymized_embeddings = embeddings + noise.to(device)

    return anonymized_embeddings

def anonymize_embeddings_gaussian(embeddings, epsilon=0.1, device="cpu"):
    """
    Anonymize embeddings using Differential Privacy (DP) with Gaussian noise.

    This function adds Gaussian noise to the embeddings to protect privacy while
    preserving utility. The amount of noise is controlled by the `epsilon` parameter,
    which determines the privacy-utility trade-off.

    Parameters:
    - embeddings: PyTorch tensor, the original embeddings.
    - epsilon: float, the parameter controlling the privacy guarantee of DP.
        Lower values provide stronger privacy but introduce more noise.
    - device: str, the device on which the computation will be performed (e.g., "cpu" or "cuda").

    Returns:
    - PyTorch tensor, the anonymized embeddings.

    Raises:
    - ValueError: If `epsilon` is negative.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    # Generate Gaussian noise and move it to the specified device
    noise = torch.randn_like(embeddings) * epsilon
    anonymized_embeddings = embeddings + noise.to(device)

    return anonymized_embeddings


def anonymize_embeddings_laplace(embeddings, epsilon, device="cpu"):
    """
    Anonymize embeddings using Differential Privacy (DP) with Laplace noise.

    This function adds Laplace noise to the embeddings to protect privacy while
    preserving utility. The amount of noise is controlled by the `epsilon` parameter,
    which determines the privacy-utility trade-off.

    Parameters:
    - embeddings: PyTorch tensor, the original embeddings.
    - epsilon: float, the parameter controlling the privacy guarantee of DP.
        Lower values provide stronger privacy but introduce more noise.
    - device: str, the device on which the computation will be performed (e.g., "cpu" or "cuda").

    Returns:
    - PyTorch tensor, the anonymized embeddings.

    Raises:
    - ValueError: If `epsilon` is negative.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    # Calculate Laplace noise scale and sample noise
    scale = epsilon / 2  # Relationship between epsilon and Laplace scale
    noise = torch.distributions.laplace.Laplace(loc=0, scale=scale).sample().to(device)

    anonymized_embeddings = embeddings + noise

    return anonymized_embeddings


def anonymize_embeddings_gaussian(embeddings, epsilon=0.1, device="cpu"):
    """
    Anonymize embeddings using Differential Privacy (DP) with Gaussian noise.

    This function adds Gaussian noise to the embeddings to protect privacy while
    preserving utility. The amount of noise is controlled by the `epsilon` parameter,
    which determines the privacy-utility trade-off.

    Parameters:
    - embeddings: PyTorch tensor, the original embeddings.
    - epsilon: float, the parameter controlling the privacy guarantee of DP.
        Lower values provide stronger privacy but introduce more noise.
    - device: str, the device on which the computation will be performed (e.g., "cpu" or "cuda").

    Returns:
    - PyTorch tensor, the anonymized embeddings.

    Raises:
    - ValueError: If `epsilon` is not non-negative.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be a positive value or 0.")

    anonymized_embeddings = (embeddings + torch.tensor(np.random.normal(scale=epsilon, size=embeddings.shape),
                                                       dtype=torch.float32)).to(device)
    return anonymized_embeddings



def anonymize_embeddings_permutation(embeddings):
    permutation = torch.randperm(embeddings.shape[1])
    return embeddings[:, permutation]


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


def anonymize_embeddings_cluster_creator(train_embeddings, eps, min_samples, train_labels):
    # Create a DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit the model to the original embeddings
    labels = dbscan.fit_predict(train_embeddings.numpy())  # Convert to NumPy array
    print("Train Cluster Labels:", labels)

    silhouette = silhouette_score(train_embeddings, labels)

    print(f"Epsilon: {eps}, Silhouette Score: {silhouette}")

    # Get the unique cluster labels
    unique_labels = np.unique(labels)
    print("Number of clusters:", len(unique_labels))

    # Initialize an array to store the cluster edges
    cluster_edges = []
    cluster_edges_labels = {}

    # Calculate the edges for each embedding
    for label in unique_labels:
        cluster_mask = torch.zeros(train_labels.shape[0], dtype=torch.bool)
        cluster_mask[train_labels == label] = True  # Use train_labels for creating cluster_mask

        cluster_embeddings = train_embeddings[cluster_mask]

        # Check if the cluster is not empty
        if cluster_embeddings.shape[0] > 0:
            min_values, max_values = get_cluster_edges(cluster_embeddings)

            # Convert min_values and max_values to tensors
            min_values_tensor = torch.tensor(min_values, dtype=torch.float32)
            max_values_tensor = torch.tensor(max_values, dtype=torch.float32)

            # Store the cluster edges
            cluster_edges.append((min_values_tensor, max_values_tensor))

            # Store the ground truth labels for this cluster edge
            cluster_edges_labels[label] = train_labels[cluster_mask].tolist()

    # Convert the dictionary values to a list of tensors
    cluster_edges_labels_list = [torch.tensor(v) for v in cluster_edges_labels.values()]

    # Concatenate the list of tensors along the appropriate dimension (assuming dimension 0)
    cluster_edges_labels_tensor = torch.cat(cluster_edges_labels_list, dim=0)

    # Convert the list to a tensor
    original_labels_tensor = torch.tensor(cluster_edges_labels_tensor, dtype=torch.long)

    return cluster_edges, original_labels_tensor


def anonymize_embeddings_cluster(cluster_edges, embeddings, noise_scale):
    anonymized_embeddings = []
    found_count = 0  # Counter for embeddings found in any cluster
    not_found_count = 0  # Counter for embeddings not found in any cluster

    for embedding in embeddings:
        condition_min_max = False

        for cluster_edge in cluster_edges:
            min_values, max_values = cluster_edge
            condition_min = torch.all(embedding >= min_values)
            condition_max = torch.all(embedding <= max_values)

            if condition_min and condition_max:
                # Test embedding is within the cluster, use cluster coordinates
                centroid = (min_values + max_values) / 2
                anonymized_embeddings.append(centroid)
                found_count += 1
                condition_min_max = True
                break

        if not condition_min_max:
            nearest_cluster_idx = find_nearest_cluster(embedding, cluster_edges)
            nearest_centroid = (cluster_edges[nearest_cluster_idx][0] + cluster_edges[nearest_cluster_idx][1]) / 2
            anonymized_embeddings.append(torch.tensor(nearest_centroid, dtype=torch.float32))
            not_found_count += 1

    print(f"Embeddings found in clusters: {found_count}")
    print(f"Embeddings not found in clusters: {not_found_count}")

    anonymized_embeddings = torch.stack(anonymized_embeddings)

    return anonymized_embeddings


#TODO: Add kNN based anonymization method
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
    print(f"Embeddings found in clusters: {found_count}")
    print(f"Embeddings not found in clusters: {not_found_count}")

    anonymized_embeddings = torch.stack(anonymized_embeddings)


    return anonymized_embeddings