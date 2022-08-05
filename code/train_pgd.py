import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.optim import SGD, Optimizer
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import datasets
import models
from attack_methods import PGD_L2, DDN
import train_utils


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset_name', type=str, choices=datasets.DATASETS)
parser.add_argument('arch', type=str, choices=models.ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--num_epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from existing checkpoint')
parser.add_argument('--pretrained_model_path', type=str, default='',
                    help='Path to a pretrained model')
parser.add_argument('--use_unlabelled', action='store_true',
                    help='Use unlabelled data via self-training.')
parser.add_argument('--self_training_weight', type=float, default=1.0,
                    help='Weight of self-training.')

# Attack params
parser.add_argument('--adv-training', action='store_true')
parser.add_argument('--attack_method', default='DDN', type=str, choices=['DDN', 'PGD'])
parser.add_argument('--epsilon', default=64.0, type=float)
parser.add_argument('--num-steps', default=10, type=int)
parser.add_argument('--warmup', default=1, type=int, help="Number of epochs over which \
-                    the maximum allowed perturbation increases linearly from zero to args.epsilon.")
parser.add_argument('--num-noise-vec', default=1, type=int,
                    help="number of noise vectors to use for finding adversarial examples. `m_train` in the paper.")
parser.add_argument('--train-multi-noise', action='store_true', 
                    help="if included, the weights of the network are optimized using all the noise samples. \
-                       Otherwise, only one of the samples is used.")
parser.add_argument('--no-grad-attack', action='store_true',
                    help="Choice of whether to use gradients during attack or do the cheap trick")

# PGD-specific
parser.add_argument('--random-start', default=True, type=bool)

# DDN-specific
parser.add_argument('--init-norm-DDN', default=256.0, type=float)
parser.add_argument('--gamma-DDN', default=0.05, type=float)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if device.type == "cuda" else {}


def main():
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    # get dataset and iterate through each sample
    train_batch_size = 32
    test_batch_size = 1

    if args.dataset_name == "cifar10":
        data_dir = "data/CIFAR10/"
        trainloader, testloader, num_classes = datasets.load_cifar10_data(data_dir, train_batch_size, test_batch_size, None, kwargs)
    elif args.dataset_name == "imagenet":
        data_dir = "data/ImageNet/"
        trainloader, testloader, num_classes = datasets.load_imagenet_data(data_dir, train_batch_size, test_batch_size, None, kwargs)
    elif args.dataset_name == "mnist":
        data_dir = "data/MNIST/"
        trainloader, testloader, num_classes = datasets.load_mnist_data(data_dir, train_batch_size, test_batch_size, None, kwargs)
    else:
        raise Exception("Unrecognised dataset")

    if args.pretrained_model_path != '':
        assert args.arch == 'cifar_resnet110', 'Unsupported architecture for pretraining'
        model_checkpoint = torch.load(f=args.pretrained_model_path)
        model = models.get_architecture(model_checkpoint["arch"], args.dataset_name, device)
        model.load_state_dict(model_checkpoint['state_dict'])
        model[1].fc = nn.Linear(64, num_classes).to(device)
    else:
        model = models.get_architecture(args.arch, args.dataset_name, device)
    
    if args.attack_method == 'PGD':
        print('Attacker is PGD')
        attacker = PGD_L2(steps=args.num_steps, device=device, max_norm=args.epsilon)
    elif args.attack_method == 'DDN':
        print('Attacker is DDN')
        attacker = DDN(steps=args.num_steps, device=device, max_norm=args.epsilon, 
                    init_norm=args.init_norm_DDN, gamma=args.gamma_DDN)
    # else:
    #     raise Exception('Unknown attack')
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    for epoch in range(starting_epoch, args.num_epochs):
        scheduler.step(epoch)
        attacker.max_norm = np.min([args.epsilon, (epoch + 1) * args.epsilon/args.warmup])
        attacker.init_norm = np.min([args.epsilon, (epoch + 1) * args.epsilon/args.warmup])

        start_time = time.time()
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, args.noise_sd, attacker)
        test_loss, test_acc, test_acc_normal = test(testloader, model, criterion, args.noise_sd, attacker)
        time_elapsed = round(time.time() - start_time, 4)

        if args.adv_training:
            log(logfilename, "{}\t{:.2f}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, after - before,
                scheduler.get_lr()[0], train_loss, test_loss, train_acc, test_acc, test_acc_normal))
        else:
            log(logfilename, "{}\t{:.2f}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, after - before,
                scheduler.get_lr()[0], train_loss, test_loss, train_acc, test_acc))

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_path)

def train(dataloader: DataLoader, model: nn.Module, criterion, optimizer:optim.Optimizer, epoch_idx: int, noise_std: float, attacker):
    loss_lst = train_utils.AverageMeter()
    top1 = train_utils.AverageMeter()
    top5 = train_utils.AverageMeter()

    # switch to train mode
    model.train()
    train_utils.requires_grad_(model, True)

    for itr, (X_batch, Y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        if args.adv_training:
            model.eval()
            train_utils.requires_grad_(model, False)
            X_batch = attacker.attack(model, X_batch, Y_batch, 
                                    noise=noise, 
                                    num_noise_vectors=args.num_noise_vec, 
                                    no_grad=args.no_grad_attack
                                    )
            model.train()
            train_utils.requires_grad_(model, True)
        
        if args.train_multi_noise:
            X_batch_noise = X_batch + noise

            Y_pred = model(X_batch_noise)
            loss = criterion(Y_pred, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            inputs = inputs[::args.num_noise_vec] # subsample the samples
            noise = noise[::args.num_noise_vec]
            # noise = torch.randn_like(inputs, device='cuda') * noise_sd
            noisy_inputs_list.append(inputs + noise)



    
