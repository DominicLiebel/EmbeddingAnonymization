# main.py

import torch
from torch import nn
import argparse
import yaml
from model import DropoutModel, DropoutAndBatchnormModel, SimpleModel
from visualization import plot_accuracy_vs_error, plot_accuracy_vs_error_every_epoch
from data_loader import load_npz_files
from evaluation import find_best_parameters

# Set seeds for reproducibility
torch.manual_seed(42)


parser = argparse.ArgumentParser(description='xAI-Proj-M')
parser.add_argument('--config', default='./config.yaml')

def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.full_load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)



    # Load train and test datasets
    train_embeddings, train_labels = load_npz_files(args.train_file_path, args.normalize)
    test_embeddings, test_labels = load_npz_files(args.test_file_path, args.normalize)

    if args.normalize == True:
        print("Embeddings Loaded: Normalized")
    else:
        print("Embeddings loaded: Not Normalized")

    if "cifar10" in args.train_file_path:
        output_size = 10
    else:
        output_size = 100

    # Get the input size from the shape of test_embeddings
    input_size = test_embeddings.shape[1]

    if args.model == 'DropoutModel':
        model = DropoutModel(input_size,output_size)
    elif args.model == 'DropoutAndBatchnormModel':
        model = DropoutAndBatchnormModel(input_size, output_size)
    elif args.model == 'SimpleModel':
        model = SimpleModel(input_size, output_size)

    model = model.to(torch.float32)

    print(model)

    if torch.cuda.is_available():
        model = model.cuda()

    if args.loss_type == "CE":
        criterion = nn.CrossEntropyLoss()

    print(criterion)

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.reg)

        print("SGD")

    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        print("Adam")

    if args.tuning:
        reconstruction_errors, accuracy_losses, all_epsilons, all_min_samples_values, all_noise_scale_values = find_best_parameters(args, train_embeddings, test_embeddings, model,
                                                                                                                                    optimizer, criterion, train_labels, test_labels)
        plot_accuracy_vs_error_every_epoch(args, reconstruction_errors, accuracy_losses, args.method, all_epsilons, all_min_samples_values, all_noise_scale_values)



if __name__ == "__main__":
    main()