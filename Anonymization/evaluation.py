# evaluation.py

import torch
from torch.utils.data import TensorDataset, DataLoader

from anonymization import anonymize_embeddings
from train import train, validate
from train_util import adjust_learning_rate


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


def find_best_parameters(args, normalized_train_embeddings, normalized_test_embeddings, model,
                         optimizer, criterion, train_labels, test_labels):
    original_model_accuracy_cifar10 = 0.9893
    original_model_accuracy_cifar100 = 0.9120
    reconstruction_errors = []
    accuracy_losses = []

    all_epsilons = []
    all_min_samples_values = []
    all_noise_scale_values = []

    for eps in args.eps_tuning:
        for min_samples in args.min_samples_tuning:
            for noise_scale in args.noise_scale_tuning:
                # Anonymize embeddings using selected method
                test_anonymized_embeddings = (
                    anonymize_embeddings(normalized_test_embeddings, args.method, eps=eps, min_samples=min_samples, noise_scale=noise_scale))
                train_embeddings_anonymized = (
                    anonymize_embeddings(normalized_train_embeddings, args.method, eps=eps, min_samples=args.min_samples, noise_scale=args.noise_scale))

                train_dataset = TensorDataset(torch.from_numpy(train_embeddings_anonymized), torch.from_numpy(train_labels))
                test_dataset = TensorDataset(torch.from_numpy(test_anonymized_embeddings), torch.from_numpy(test_labels))

                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
                for epoch in range(args.epochs):
                    adjust_learning_rate(optimizer, epoch, args)

                    # train loop
                    train(epoch, train_loader, model, optimizer, criterion)

                    # validation loop
                    acc, cm = validate(epoch, test_loader, model, criterion, args.train_file_path)



                    # Calculate reconstruction error and accuracy loss
                    # Convert NumPy arrays to PyTorch tensors
                    normalized_test_embeddings_tensor = torch.from_numpy(normalized_test_embeddings)
                    test_anonymized_embeddings_tensor = torch.from_numpy(test_anonymized_embeddings)

                    reconstruction_error = torch.mean((normalized_test_embeddings_tensor - test_anonymized_embeddings_tensor)**2).item()
                    if "cifar100" in args.train_file_path:
                        accuracy_loss = original_model_accuracy_cifar100 - acc
                    elif "cifar10" in args.train_file_path:
                        accuracy_loss = original_model_accuracy_cifar10 - acc

                    # Append to lists
                    reconstruction_errors.append(reconstruction_error)
                    accuracy_losses.append(accuracy_loss)

                    # Update lists for all parameters
                    all_epsilons.append(eps)
                    all_min_samples_values.append(min_samples)
                    all_noise_scale_values.append(noise_scale)

                    # Print results for the current iteration
                    print(f"Iteration: Epsilon={eps}, Min Samples={min_samples}, Noise Scale={noise_scale}, "
                          f"Accuracy={acc * 100:.2f}%, "
                          f"Reconstruction Error={reconstruction_error:.4f} "
                          f"Accuracy Loss={accuracy_loss:.4f}")

    return (reconstruction_errors, accuracy_losses, all_epsilons, all_min_samples_values, all_noise_scale_values)
