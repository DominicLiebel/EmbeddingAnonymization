# visualization.py

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_vs_error(args, reconstruction_errors, accuracy_losses, method_to_test, all_epsilons, all_min_samples_values, all_noise_scale_values):
    """
    Plot accuracy loss vs. reconstruction error with annotations.

    Parameters:
    - reconstruction_errors (list): List of reconstruction errors.
    - accuracy_losses (list): List of accuracy losses.
    - method_to_test (str): Method being tested.
    - all_epsilons (list): List of all epsilon values.
    - all_min_samples_values (list): List of all min_samples values.
    - all_noise_scale_values (list): List of all noise_scale values.
    """
    plt.figure()
    plt.plot(reconstruction_errors, accuracy_losses, marker='o', linestyle='', label='All Epochs')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Accuracy Loss')
    plt.title('Accuracy Loss vs. Reconstruction Error')
    plt.suptitle(f'Parameters: {args.model}: {method_to_test}, {args.optimizer}, {args.loss_type}')

    plt.legend()
    plt.savefig('output_plot.png')
    plt.show()

def plot_accuracy_vs_error_every_epoch(args, reconstruction_errors, accuracy_losses, method_to_test, all_epsilons, all_min_samples_values, all_noise_scale_values):
    """
    Plot accuracy loss vs. reconstruction error with annotations.

    Parameters:
    - reconstruction_errors (list): List of reconstruction errors.
    - accuracy_losses (list): List of accuracy losses.
    - method_to_test (str): Method being tested.
    - all_epsilons (list): List of all epsilon values.
    - all_min_samples_values (list): List of all min_samples values.
    - all_noise_scale_values (list): List of all noise_scale values.
    """
    # Plotting for each combination
    plt.figure()
    plt.plot(reconstruction_errors, accuracy_losses, marker='o')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Accuracy Loss')
    plt.title('Accuracy Loss vs. Reconstruction Error')
    plt.suptitle(f'Parameters: {args.model}: {method_to_test}, {args.optimizer}, {args.loss_type}')

    # Add text annotations for each point with epsilon, min_samples, and noise_scale values
    for i, (error, loss, epsilon, min_samples, noise_scale) in enumerate(zip(reconstruction_errors, accuracy_losses, all_epsilons, all_min_samples_values, all_noise_scale_values)):
        plt.text(error, loss, f'({epsilon=:.2f}, {min_samples=}, {noise_scale=:.2f})', fontsize=6, ha='center', va='bottom')
    plt.savefig('output_plot.png')
    plt.show()

def visualize_clusters(embeddings, labels, method='t-SNE', n_components=2):
    if method == 't-SNE':
        tsne = TSNE(n_components=n_components, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)
    elif method == 'PCA':
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
    else:
        raise ValueError(f"Unsupported visualization method: {method}")

    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', s=20)
    plt.title(f'Clusters after Anonymization ({method})')
    plt.xlabel(f'{method} Component 1')
    plt.ylabel(f'{method} Component 2')
    plt.show()

def visualize_clusters_with_anonymized_2d(test_embeddings, anonymized_embeddings, cluster_edges, title='Embeddings Visualization with Clusters'):
    plt.figure(figsize=(10, 8))

    # Plot the original test embeddings
    plt.scatter(test_embeddings[:, 0], test_embeddings[:, 1], c='blue', label='Original Test Embeddings')

    # Plot the cluster edges
    for cluster_edge in cluster_edges:
        min_values, max_values = cluster_edge
        plt.plot([min_values[0], max_values[0], max_values[0], min_values[0], min_values[0]],
                 [min_values[1], min_values[1], max_values[1], max_values[1], min_values[1]],
                 color='red', linestyle='dashed', linewidth=2, alpha=0.7, label='Cluster Edges')

    # Convert anonymized_embeddings to a NumPy array
    anonymized_embeddings_np = np.array(anonymized_embeddings)

    # Plot the anonymized embeddings within the clusters
    plt.scatter(anonymized_embeddings_np[:, 0], anonymized_embeddings_np[:, 1], c='green', s=20, label='Anonymized Embeddings')

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def visualize_clusters_with_anonymized_3d(test_embeddings, anonymized_embeddings, cluster_edges, title='Embeddings Visualization with Clusters'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original test embeddings
    ax.scatter(test_embeddings[:, 0], test_embeddings[:, 1], test_embeddings[:, 2],s = 5, c='blue', label='Original Test Embeddings')

    # Plot the cluster edges
    for cluster_edge in cluster_edges:
        min_values, max_values = cluster_edge
        ax.plot([min_values[0], max_values[0], max_values[0], min_values[0], min_values[0]],
                [min_values[1], min_values[1], max_values[1], max_values[1], min_values[1]],
                [min_values[2], min_values[2], max_values[2], max_values[2], min_values[2]],
                color='red', linestyle='dashed', linewidth=2, alpha=0.7)

    # Convert anonymized_embeddings to a NumPy array
    anonymized_embeddings_np = np.array(anonymized_embeddings)

    # Plot the anonymized embeddings within the clusters
    ax.scatter(anonymized_embeddings_np[:, 0], anonymized_embeddings_np[:, 1], anonymized_embeddings_np[:, 2], c='green', s=20, label='Anonymized Embeddings')

    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.legend()
    plt.show()
