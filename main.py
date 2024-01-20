# main.py

import torch
from torch import nn
import argparse
import yaml
import torchvision.transforms as transforms
from evaluation import calculate_mean_relative_difference
from train_util import find_best_parameters
from model import DropoutModel, DropoutAndBatchnormModel, SimpleModel
from anonymization import anonymize_embeddings
import matplotlib.pyplot as plt
from data_loader import load_npz_files
from train_util import train_and_evaluate, adjust_learning_rate
from train import train, validate
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import copy

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
    train_embeddings, train_labels = load_npz_files(args.train_file_path)
    test_embeddings, test_labels = load_npz_files(args.test_file_path)

    # Normalize train and test embeddings
    scaler = StandardScaler()
    normalized_train_embeddings = scaler.fit_transform(train_embeddings)
    normalized_test_embeddings = scaler.transform(test_embeddings)
    print("Embeddings normalized")

    # Anonymize train and test embeddings
    train_embeddings_anonymized = anonymize_embeddings(normalized_train_embeddings, method=args.method,
                                                       eps=args.eps, min_samples=args.min_samples, noise_scale=args.noise_scale)
    test_embeddings_anonymized = anonymize_embeddings(normalized_test_embeddings, method=args.method,
                                                      eps=args.eps, min_samples=args.min_samples, noise_scale=args.noise_scale)
    print("Anonymized embeddings")

    # Create DataLoader instances
    train_dataset = TensorDataset(torch.from_numpy(normalized_train_embeddings), torch.from_numpy(train_labels))
    test_dataset = TensorDataset(torch.from_numpy(normalized_test_embeddings), torch.from_numpy(test_labels))

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("Datasets loaded")


    for batch in test_loader:
        test_embeddings, _ = batch
        break

    # Get the input size from the shape of test_embeddings
    input_size = test_embeddings.shape[1]

    if args.train_file_path == "CIFAR10":
        output_size = 10
    else:
        output_size = 100

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

    best = 0.0
    best_cm = None
    best_model = None
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train loop
        train(epoch, train_loader, model, optimizer, criterion)

        # validation loop
        acc, cm = validate(epoch, test_loader, model, criterion, args.train_file_path)

        if acc > best:
            best = acc
            best_cm = cm
            best_model = copy.deepcopy(model)

    print('Best Prec @1 Acccuracy: {:.4f}'.format(best))
    per_cls_acc = best_cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    if args.save_best:
        torch.save(best_model.state_dict(), './checkpoints/' + args.model.lower() + '.pth')


if __name__ == "__main__":
    main()
