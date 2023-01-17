import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
import math
from tqdm import tqdm

from models.resnet import get_pretrained_resnet50, get_pretrained_resnet50_approximate
from models.linear import ApproximateLinear
from utils import save_json
from datasets import ProcessedCIFAR100


# ==> Train Benchmark Model

def get_dataloader(train: bool):
    if args.freeze:
        return trainloader_processed if train else testloader_processed
    else:
        return trainloader if train else testloader

def train_step_benchmark(epoch):
    print(f'\nTraining epoch {epoch+1}..')
    net_benchmark.train()
    train_loss = 0
    correct = 0
    total = 0
    effective_time = 0
    start_time = time.time()

    pbar = tqdm(enumerate(get_dataloader(train=True)))
    for batch, (inputs, labels) in pbar:
        # load data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # forward and backward propagation
        temp_time = time.time()
        optimizer_benchmark.zero_grad()
        outputs = net_benchmark(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_benchmark.step()
        effective_time += time.time() - temp_time

        # stat updates
        train_loss += (loss.item() - train_loss)/(batch+1)
        total += labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()

        pbar.set_description(f'epoch {epoch+1} batch {batch+1}: train loss {train_loss}')

    # return stats
    total_time = time.time() - start_time
    train_acc = 100*correct/total
    return train_loss, train_acc, total_time, effective_time


def test_step(epoch, net):
    print(f'\nEvaluating epoch {epoch+1}..')
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(enumerate(get_dataloader(train=False)))
    with torch.no_grad():
        for batch, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += (loss.item() - test_loss)/(batch+1)
            total += labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            # test_acc = 100*correct/total
            # pbar.set_description(f'test loss: {test_loss}, test acc: {test_acc}')

    test_acc = 100*correct/total
    print(f'test loss: {test_loss}, test acc: {test_acc}')
    return test_loss, test_acc


def train_benchmark():
    stats = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'total_time': [],
        'effective_time': [],
        'epochs': [*range(epochs_benchmark)]
    }

    for epoch in range(epochs_benchmark):
        train_loss, train_acc, total_time, effective_time = train_step_benchmark(epoch)
        test_loss, test_acc = test_step(epoch, net_benchmark)
        scheduler_benchmark.step()

        stats['train_loss'].append(train_loss)
        stats['train_acc'].append(train_acc)
        stats['test_loss'].append(test_loss)
        stats['test_acc'].append(test_acc)
        stats['total_time'].append(total_time)
        stats['effective_time'].append(effective_time)

    for epoch in range(1, epochs_benchmark):
        stats['total_time'][epoch] += stats['total_time'][epoch-1]
        stats['effective_time'][epoch] += stats['effective_time'][epoch-1]

    return stats


# ==> Train Target Model

def train_step_target(epoch):
    print(f'\nTraining epoch {epoch+1}')
    net_target.train()
    train_loss = 0
    correct = 0
    total = 0
    effective_time = 0
    start_time = time.time()

    pbar = tqdm(enumerate(get_dataloader(train=True)))
    for batch, (inputs, labels) in pbar:
        # load data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # resample projection layer
        temp_time = time.time()
        if batch % resample_period == 0:
            if args.freeze:
                net_target.resample()
            else:
                net_target.fc.reset_projection(gaussian_mean, gaussian_std)

        # forward and backward propagation
        optimizer_target.zero_grad()
        outputs = net_target(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_target.step()
        effective_time += time.time() - temp_time

        # update stats
        train_loss += (loss.item() - train_loss) / (batch+1)
        total += labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_description(f'epoch {epoch+1} batch {batch+1}: train loss {train_loss}')

    # return stats
    total_time = time.time() - start_time
    train_acc = 100*correct/total
    return train_loss, train_acc, total_time, effective_time


def train_target():
    stats = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'total_time': [],
        'effective_time': [],
        'epochs': [*range(epochs_target)],
        'reduced_dim': reduced_dim,
        'resample_period': resample_period,
        'gaussian_params': (gaussian_mean, gaussian_std)
    }

    for epoch in range(epochs_target):
        train_loss, train_acc, total_time, effective_time = train_step_target(epoch)
        test_loss, test_acc = test_step(epoch, net_target)
        scheduler_target.step()

        stats['train_loss'].append(train_loss)
        stats['train_acc'].append(train_acc)
        stats['test_loss'].append(test_loss)
        stats['test_acc'].append(test_acc)
        stats['total_time'].append(total_time)
        stats['effective_time'].append(effective_time)

    for epoch in range(1, epochs_target):
        stats['total_time'][epoch] += stats['total_time'][epoch-1]
        stats['effective_time'][epoch] += stats['effective_time'][epoch-1]

    return stats

if __name__ == '__main__':

    # ==> Arg Parsing

    parser = argparse.ArgumentParser(description='Random Projection Training')
    parser.add_argument('--reduced_dim', '-d', default=100, type=int, help='reduced dimension')
    parser.add_argument('--resample_period', '-p', default=16, type=int, help='period of resampling random projection matrix')
    parser.add_argument('--std_multiplier', '-s', default=1, type=float, help='mean and std of gaussian layer')

    parser.add_argument('--train_benchmark', default=False, type=bool, help='whether to train benchmark model')
    parser.add_argument('--train_target', default=False, type=bool, help='whether to train target model')

    parser.add_argument('--dir_name', default='.', type=str, help='directory name for experiment results')
    parser.add_argument('--benchmark_name', default='benchmark', type=str, help='benchmark stats file name')
    parser.add_argument('--target_name', default='target', type=str, help='target stats file name')

    parser.add_argument('--epochs_benchmark', default=50, type=int)
    parser.add_argument('--epochs_target', default=50, type=int)

    parser.add_argument('--freeze', default=False, type=bool, help='whether to freeze all previous layers')
    args = parser.parse_args()


    # ==> Data Processing

    print('=== Preparing Data.. ===')

    # data transformation
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([                                                                   
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # dataset
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)


    # ==> Preprocess Data

    root_processed = './data/cifar-100-processed'
    resnet50_no_last = get_pretrained_resnet50(no_last=True)

    trainset_processed = ProcessedCIFAR100(
        root=root_processed, train=True, model=resnet50_no_last, dataloader=trainloader)
    trainloader_processed = torch.utils.data.DataLoader(
        trainset_processed, batch_size=128, shuffle=True, num_workers=2)

    testset_processed = ProcessedCIFAR100(
        root=root_processed, train=False, model=resnet50_no_last, dataloader=testloader)
    testloader_processed = torch.utils.data.DataLoader(
        testset_processed, batch_size=100, shuffle=False, num_workers=2)


    # ==> Build Model

    # parameters for training approximate model
    reduced_dim = args.reduced_dim
    resample_period = args.resample_period
    gaussian_mean = 0.
    gaussian_std = math.sqrt(reduced_dim/2048) * args.std_multiplier
    # compute std so that P^TP has expectation = I

    epochs_benchmark = args.epochs_benchmark
    epochs_target = args.epochs_target

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # linear model in 'freeze' model v.s. whole model otherwise
    if args.freeze:
        net_benchmark = nn.Sequential(nn.Linear(in_features=2048, out_features=100, bias=True)).to(device)
        net_target = ApproximateLinear(in_features=2048, out_features=100, reduced_dim=reduced_dim, bias=True).to(device)
    else:
        net_benchmark = get_pretrained_resnet50().to(device)
        net_target = get_pretrained_resnet50_approximate(reduced_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_benchmark = optim.SGD(net_benchmark.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler_benchmark = optim.lr_scheduler.CosineAnnealingLR(optimizer_benchmark, T_max=epochs_benchmark)

    optimizer_target = optim.SGD(net_target.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler_target = optim.lr_scheduler.CosineAnnealingLR(optimizer_target, T_max=epochs_target)


    # ==> Train Model

    print('=== Training models.. ===')
    if args.train_benchmark:
        print('\nTraining benchmark model..')
        stats_benchmark = train_benchmark()
        save_json(stats_benchmark, f'{args.dir_name}/{args.benchmark_name}.json')
    if args.train_target:
        print('\nTraining target model..')
        stats_target = train_target()
        save_json(stats_target, f'{args.dir_name}/{args.target_name}.json')