# evaluation.py

import torch
from torch.utils.data import TensorDataset, DataLoader
from train import train, validate
from train_util import adjust_learning_rate
from visualization import visualize_clusters_with_anonymized_3d
from anonymization import anonymize_embeddings_uniform, anonymize_embeddings_gaussian, anonymize_embeddings_laplace, anonymize_embeddings_pca, anonymize_embeddings_cluster, anonymize_embeddings_cluster_creator

# Set seeds for reproducibility
torch.manual_seed(42)


def check_overlap(original_embeddings, anonymized_embeddings):
    """
    Check for overlap between original and anonymized embeddings.

    Parameters:
    - original_embeddings: NumPy array or PyTorch tensor, original embeddings
    - anonymized_embeddings: NumPy array or PyTorch tensor, anonymized embeddings

    Returns:
    - bool, True if there is overlap, False otherwise
    """
    # Convert PyTorch tensors to NumPy arrays if needed
    if isinstance(original_embeddings, torch.Tensor):
        original_embeddings = original_embeddings.numpy()
    if isinstance(anonymized_embeddings, torch.Tensor):
        anonymized_embeddings = anonymized_embeddings.numpy()

    # Convert to sets for efficient overlap check
    original_set = set(map(tuple, original_embeddings))
    anonymized_set = set(map(tuple, anonymized_embeddings))

    # Check for overlap
    overlap = bool(original_set.intersection(anonymized_set))

    return overlap


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


def find_best_parameters(args, train_embeddings, test_embeddings, model,
                         optimizer, criterion, train_labels, test_labels):
    original_model_accuracy_cifar10 = 0.9870
    original_model_accuracy_cifar100 = 0.8925
    reconstruction_errors = []
    accuracy_losses = []
    all_pca_dimension_values = []
    all_epsilons = []

    if args.method == "pca":
        for noise_scale in args.noise_scale_tuning:
            train_embeddings_anonymized = anonymize_embeddings_pca(train_embeddings, noise_scale)

            # Anonymize test embeddings using the same clusters
            test_embeddings_anonymized = anonymize_embeddings_pca(test_embeddings, noise_scale)

            train_dataset = TensorDataset(train_embeddings_anonymized, train_labels)
            test_dataset = TensorDataset(test_embeddings_anonymized, test_labels)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            for epoch in range(args.epochs):
                adjust_learning_rate(optimizer, epoch, args)

                # Train loop
                train(epoch, train_loader, model, optimizer, criterion)

                # Validation loop
                acc, cm = validate(epoch, test_loader, model, criterion, args.test_file_path)

                # Calculate reconstruction error and accuracy loss

                reconstruction_error = torch.mean((test_embeddings - test_embeddings_anonymized)**2).item()
                if "cifar100" in args.train_file_path:
                    accuracy_loss = original_model_accuracy_cifar100 - acc
                elif "cifar10" in args.train_file_path:
                    accuracy_loss = original_model_accuracy_cifar10 - acc

                # Append to lists
                reconstruction_errors.append(reconstruction_error)
                accuracy_losses.append(accuracy_loss)

                # Update lists for all parameters
                all_pca_dimension_values.append(noise_scale)

                has_overlap = check_overlap(test_embeddings, test_embeddings_anonymized)

                # Print results for the current iteration
                print(f"Iteration: Dimensions={noise_scale}, "
                      f"Accuracy={acc * 100:.2f}%, "
                      f"Reconstruction Error={reconstruction_error:.4f} "
                      f"Accuracy Loss={accuracy_loss:.4f} "
                      f"Overlap={has_overlap}")
        return (reconstruction_errors, accuracy_losses, all_pca_dimension_values)

    elif args.method == "uniform" or args.method == "gaussian" or args.method == "laplace":
        for eps in args.eps_tuning:
            if args.method == "uniform":
                train_embeddings_anonymized = anonymize_embeddings_uniform(train_embeddings, epsilon=eps)
                test_embeddings_anonymized = anonymize_embeddings_uniform(test_embeddings, epsilon=eps)
            elif args.method == "gaussian":
                train_embeddings_anonymized = anonymize_embeddings_gaussian(train_embeddings, epsilon=eps)
                test_embeddings_anonymized = anonymize_embeddings_gaussian(test_embeddings, epsilon=eps)
            elif args.method == "laplace":
                train_embeddings_anonymized = anonymize_embeddings_laplace(train_embeddings, epsilon=eps)
                test_embeddings_anonymized = anonymize_embeddings_laplace(test_embeddings, epsilon=eps)

            train_dataset = TensorDataset(train_embeddings_anonymized, train_labels)
            test_dataset = TensorDataset(test_embeddings_anonymized, test_labels)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            for epoch in range(args.epochs):
                adjust_learning_rate(optimizer, epoch, args)

                # Train loop
                train(epoch, train_loader, model, optimizer, criterion)

                # Validation loop
                acc, cm = validate(epoch, test_loader, model, criterion, args.test_file_path)

                # Calculate reconstruction error and accuracy loss
                reconstruction_error = torch.mean((test_embeddings - test_embeddings_anonymized) ** 2).item()
                if "cifar100" in args.train_file_path:
                    accuracy_loss = original_model_accuracy_cifar100 - acc
                elif "cifar10" in args.train_file_path:
                    accuracy_loss = original_model_accuracy_cifar10 - acc

                # Append to lists
                reconstruction_errors.append(reconstruction_error)
                accuracy_losses.append(accuracy_loss)

                # Update lists for all parameters
                all_epsilons.append(eps)

                has_overlap = check_overlap(test_embeddings, test_embeddings_anonymized)

                # Print results for the current iteration
                print(f"Iteration: Dimensions={eps}, "
                      f"Accuracy={acc * 100:.2f}%, "
                      f"Reconstruction Error={reconstruction_error:.4f} "
                      f"Accuracy Loss={accuracy_loss:.4f} "
                      f"Overlap={has_overlap}")

        return (reconstruction_errors, accuracy_losses, all_epsilons)

    elif args.method == "cluster":
        all_epsilons = []
        all_min_samples_values = []
        all_noise_scale_values = []

        for eps in args.eps_tuning:
            for min_samples in args.min_samples_tuning:
                for noise_scale in args.noise_scale_tuning:
                    # Anonymize embeddings using the cluster-based method
                    cluster_edges_train, cluster_edges_labels = anonymize_embeddings_cluster_creator(train_embeddings, eps, min_samples, train_labels)
                    train_embeddings_anonymized = anonymize_embeddings_cluster(cluster_edges_train, train_embeddings, noise_scale)
                    print(f"Number of clusters: {len(cluster_edges_train)}")

                    # Anonymize test embeddings using the same clusters
                    test_embeddings_anonymized = anonymize_embeddings_cluster(cluster_edges_train, test_embeddings, noise_scale)



                    visualize_clusters_with_anonymized_3d(test_embeddings, test_embeddings_anonymized, cluster_edges_train, title='Embeddings Visualization with Clusters before Training')

                    #TODO: Change visualization to show clusters in 3D and color anonymized dots in a cluster green and dots outside in red

                    train_dataset = TensorDataset(train_embeddings_anonymized, cluster_edges_labels)
                    test_dataset = TensorDataset(test_embeddings_anonymized, test_labels)

                    #TODO: Use anonymized embedding to train dataset?
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
                    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
                    for epoch in range(args.epochs):
                        adjust_learning_rate(optimizer, epoch, args)

                        # train loop
                        train(epoch, train_loader, model, optimizer, criterion)

                        # validation loop
                        acc, cm = validate(epoch, test_loader, model, criterion, args.test_file_path)

                        # Calculate reconstruction error and accuracy loss
                        # Convert NumPy arrays to PyTorch tensors

                        reconstruction_error = torch.mean((test_embeddings - test_embeddings_anonymized)**2).item()
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

                        has_overlap = check_overlap(test_embeddings, test_embeddings_anonymized)

                        # Print results for the current iteration
                        print(f"Iteration: Epsilon={eps}, Min Samples={min_samples}, Noise Scale={noise_scale}, "
                              f"Accuracy={acc * 100:.2f}%, "
                              f"Reconstruction Error={reconstruction_error:.4f} "
                              f"Accuracy Loss={accuracy_loss:.4f} "
                              f"Overlap={has_overlap}")
                    visualize_clusters_with_anonymized_3d(test_embeddings, test_embeddings_anonymized, cluster_edges_train, title='Embeddings Visualization with Clusters after Training')

            return (reconstruction_errors, accuracy_losses, all_epsilons, all_min_samples_values, all_noise_scale_values)
