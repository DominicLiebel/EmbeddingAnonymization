# evaluation.py

import torch
from sklearn.metrics import accuracy_score


def calculate_relative_difference(original_embedding, anonymized_embedding):
    """
    Calculate the relative difference between original and anonymized embeddings.

    Parameters:
    - original_embedding: float, the original embedding value
    - anonymized_embedding: float, the anonymized embedding value

    Returns:
    - float: Relative difference as a percentage
    """
    if torch.any(original_embedding == 0):
        raise ValueError("Cannot calculate relative difference when the original embedding is 0.")

    difference = anonymized_embedding - original_embedding
    relative_difference = (difference / abs(original_embedding)) * 100.0

    return relative_difference


def calculate_mean_relative_difference(original_embeddings, anonymized_embeddings):
    """
    Calculate the mean relative difference for each image.

    Parameters:
    - original_embeddings: list of original embedding values
    - anonymized_embeddings: list of anonymized embedding values

    Returns:
    - list of floats: Mean relative difference for each image
    """
    mean_relative_differences = []
    for original, anonymized in zip(original_embeddings, anonymized_embeddings):
        relative_difference = calculate_relative_difference(original, anonymized)
        mean_relative_difference = torch.mean(relative_difference).item()
        mean_relative_differences.append(mean_relative_difference)

    return mean_relative_differences


def evaluate_model(model, test_embeddings, test_labels, device="cpu"):
    """
    Evaluate the model on the test set.

    Parameters:
    - model: PyTorch model
    - test_embeddings: PyTorch tensor or NumPy array, the test set of embeddings
    - test_labels: PyTorch tensor or NumPy array, the labels corresponding to the test embeddings

    Returns:
    - float: Accuracy of the model on the test set
    """
    with torch.no_grad():
        # Convert to PyTorch tensor if input is NumPy array
        if not isinstance(test_embeddings, torch.Tensor):
            test_embeddings = torch.as_tensor(test_embeddings, dtype=torch.float32).to(device)

        model.eval()
        test_outputs = model(test_embeddings).to(device)
        _, predicted_labels = torch.max(test_outputs, 1)

        # Convert to NumPy array if output is PyTorch tensor
        if isinstance(test_labels, torch.Tensor):
            test_labels = test_labels.cpu().numpy()

        accuracy = accuracy_score(test_labels, predicted_labels)

    return accuracy
