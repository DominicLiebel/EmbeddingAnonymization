# train_util.py

import numpy as np
import torch
import yaml
from anonymization import anonymize_embeddings
from model import DropoutModel
from evaluation import evaluate_model

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_model(model, accuracy, device):
    """Saves the trained model to a specified filepath."""
    model_filepath = f"model_{accuracy:.3f}.pth"

    if device == "cuda":
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save(model_state_dict, model_filepath)
    print(f"Model saved successfully to: {model_filepath}")

def adjust_learning_rate(optimizer, epoch, args):
    epoch += 1
    if epoch <= args.warmup:
        lr = args.learning_rate * epoch / args.warmup
    elif epoch > args.steps[1]:
        lr = args.learning_rate * 0.01
    elif epoch > args.steps[0]:
        lr = args.learning_rate * 0.1
    else:
        lr = args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_and_evaluate(model, train_embeddings, train_labels, test_embeddings, test_labels, device="cpu"):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()


    for epoch in range(args.num_epochs):
        model.train()
        for i in range(0, len(train_embeddings), args.batch_size):
            inputs = torch.from_numpy((train_embeddings[i:i + args.batch_size].cpu().numpy()))
            targets = train_labels[i:i+args.batch_size]
            targets = torch.from_numpy(targets).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    accuracy = evaluate_model(model, test_embeddings, test_labels, device)
    return accuracy


def find_best_parameters(original_model_accuracy, normalized_train_embeddings, normalized_test_embeddings,
                         original_train_labels, original_test_labels, device, method,
                         epsilons, min_samples_values, noise_scale_values):
    best_epsilon = None
    best_min_samples = None
    best_noise_scale = None
    best_accuracy = 0.0

    best_reconstruction_error = float('inf')
    reconstruction_errors = []
    accuracy_losses = []

    all_epsilons = []
    all_min_samples_values = []
    all_noise_scale_values = []

    for eps in epsilons:
        for min_samples in min_samples_values:
            for noise_scale in noise_scale_values:
                # Anonymize embeddings using selected method
                test_anonymized_embeddings = (
                    anonymize_embeddings(normalized_test_embeddings, method,
                                         eps=eps, min_samples=min_samples, noise_scale=noise_scale, device=device))
                train_embeddings_anonymized = (
                    anonymize_embeddings(normalized_train_embeddings, method,
                                         eps=eps, min_samples=min_samples, noise_scale=noise_scale, device=device))

                test_anonymized_embeddings = test_anonymized_embeddings.to(device)
                train_embeddings_anonymized = train_embeddings_anonymized.to(device)
                normalized_test_embeddings = normalized_test_embeddings.to(device)

                # Train and evaluate the model on anonymized data
                anonymized_model = DropoutModel(input_size=test_anonymized_embeddings.shape[1],
                                                output_size=len(np.unique(original_test_labels))).to(device)
                anonymized_model_accuracy = train_and_evaluate(
                    anonymized_model,
                    train_embeddings_anonymized,
                    original_train_labels,
                    test_anonymized_embeddings,
                    original_test_labels
                )

                # Calculate reconstruction error and accuracy loss
                reconstruction_error = torch.mean((normalized_test_embeddings - test_anonymized_embeddings)**2).item()
                accuracy_loss = original_model_accuracy - anonymized_model_accuracy

                # Append to lists
                reconstruction_errors.append(reconstruction_error)
                accuracy_losses.append(accuracy_loss)

                # Update the best parameters based on accuracy or reconstruction error
                if anonymized_model_accuracy > best_accuracy:
                    best_accuracy = anonymized_model_accuracy
                    best_epsilon = eps
                    best_min_samples = min_samples
                    best_noise_scale = noise_scale
                    best_reconstruction_error = reconstruction_error

                # Update lists for all parameters
                all_epsilons.append(eps)
                all_min_samples_values.append(min_samples)
                all_noise_scale_values.append(noise_scale)

                # Print results for the current iteration
                print(f"Iteration: Epsilon={eps}, Min Samples={min_samples}, Noise Scale={noise_scale}, "
                      f"Accuracy={anonymized_model_accuracy * 100:.2f}%,"
                      f"Reconstruction Error={reconstruction_error:.4f}")

    return (best_epsilon, best_min_samples, best_noise_scale, best_accuracy, best_reconstruction_error,
            reconstruction_errors, accuracy_losses, all_epsilons, all_min_samples_values, all_noise_scale_values)
