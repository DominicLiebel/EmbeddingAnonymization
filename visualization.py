# visualization.py

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

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
    # Plotting for each combination
    plt.figure()
    plt.plot(reconstruction_errors, accuracy_losses, marker='o')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Accuracy Loss')
    plt.title('Accuracy Loss vs. Reconstruction Error')
    plt.suptitle(f'Parameters: Method={method_to_test}, Model={args.model}')

    # Add text annotations for each point with epsilon, min_samples, and noise_scale values
    for i, (error, loss, epsilon, min_samples, noise_scale) in enumerate(zip(reconstruction_errors, accuracy_losses, all_epsilons, all_min_samples_values, all_noise_scale_values)):
        plt.text(error, loss, f'({epsilon=:.6f}, {min_samples=}, {noise_scale=:.4f})', fontsize=8, ha='middle', va='bottom')

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
