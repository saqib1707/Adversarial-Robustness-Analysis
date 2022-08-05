"""This code loads the base classifier and runs PREDICT on many examples from a dataset
"""
import os
import time
import argparse
import numpy as np
import torch
from tqdm.notebook import tqdm

import datasets
import models
import train_utils
from core import SmoothClassifier

from attack_methods import PGD_L2, DDN


parser = argparse.ArgumentParser(description="predict on many examples")
parser.add_argument("dataset_name", choices=datasets.DATASETS_LIST, help="select datasets")
parser.add_argument("base_classifier_path", type=str, help="path to the saved classifier model")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("--N0", type=int, default=100, help="number of samples for selection")
parser.add_argument("--N", type=int, default=100000, help="number of samples for estimation")
parser.add_argument("--alpha", type=float, default=1e-3, help="failure probability")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--skip_examples", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max_examples", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--num_workers", default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument("outfile_path", type=str, help="path to the output file")

# attack options
parser.add_argument("--num_steps", default=100, type=int)
parser.add_argument("--epsilon", default=64.0, type=float)
parser.add_argument("--num_noise_vec", default=1, type=int,
                    help="number of noise vectors to use for finding adversarial examples")
parser.add_argument("--base_attack", action='store_true')
parser.add_argument("--attack_method", default=None, type=str, choices=['DDN', 'PGD'])
parser.add_argument("--no_grad_attack", action='store_true',
                    help="Choice of whether to use gradients during attack or do the cheap trick")

# PGD-specific
parser.add_argument("--random_start", default=True, type=bool)

# DDN-specific
parser.add_argument("--init_norm_DDN", default=256.0, type=float)
parser.add_argument("--gamma_DDN", default=0.05, type=float)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if device.type == "cuda" else {}

torch.manual_seed(0)


if __name__ == "__main__":
    args.epsilon /= 256.0
    args.init_norm_DDN /= 256.0

    if args.epsilon > 0:
        args.gamma_DDN = 1 - (3/510/args.epsilon)**(1/args.num_steps)

    # load the checkpoint from the base classifier
    model_checkpoint = torch.load(args.base_classifier_path)
    # print(model_checkpoint["arch"])
    base_classifier = models.get_architecture(model_checkpoint["arch"], args.dataset_name, device)
    base_classifier.load_state_dict(model_checkpoint['state_dict'])
    base_classifier.eval()

    train_utils.requires_grad_(model=base_classifier, requires_grad=False)

    # create the smoothed classifier g(x)
    smooth_classifier = SmoothClassifier(base_classifier, datasets.get_num_classes(args.dataset_name), args.sigma)

    # prepare output file for logging
    outfile = open(args.outfile_path, 'w')

    # define the attack method
    if args.attack_method == "PGD":
        print("Attack method - PGD")
        attacker = PGD_L2(steps=args.num_steps, device=device, max_norm=args.epsilon)
    elif args.attack_method == 'DDN':
        print('Attack method - DDN')
        attacker = DDN(steps=args.num_steps, device=device, max_norm=args.epsilon, 
                        init_norm=args.init_norm_DDN, gamma=args.gamma_DDN)
    # else:
    #     raise Exception("Unknown attack")

    # get dataset and iterate through each sample
    train_batch_size = 32
    test_batch_size = 1

    if args.dataset_name == "cifar10":
        data_dir = "data/CIFAR10/"
        _, testloader, num_classes = datasets.load_cifar10_data(data_dir, train_batch_size, test_batch_size, None, kwargs)
    elif args.dataset_name == "imagenet":
        data_dir = "data/ImageNet/"
        _, testloader, num_classes = datasets.load_imagenet_data(data_dir, train_batch_size, test_batch_size, None, kwargs)
    elif args.dataset_name == "mnist":
        data_dir = "data/MNIST/"
        _, testloader, num_classes = datasets.load_mnist_data(data_dir, train_batch_size, test_batch_size, None, kwargs)
    else:
        raise Exception("Unrecognised dataset")

    num_test_samples = len(testloader)
    num_corr_pred = 0
    num_samples_proc = 0
    base_smoothed_agree = 0

    for itr, (x, label) in enumerate(tqdm(testloader)):
        # only predict every args.skip_examples, and stop after args.max_examples
        if itr % args.skip_examples != 0:
            continue
        if itr // args.skip_examples == args.max_examples:
            break

        start_time = time.time()

        x = x.to(device)
        label = label.to(device)

        if args.attack_method in ['PGD', 'DDN']:
            print("Before repeat:", x.shape)
            x = x.repeat((args.num_noise_vec, 1, 1, 1))
            print("After repeat:", x.shape)
            noise = (1 - int(args.base_attack)) * torch.randn_like(x, device=device) * args.sigma
            x = attacker.attack(base_classifier, x, label, 
                                noise=noise, num_noise_vectors=args.num_noise_vec,
                                no_grad=args.no_grad_attack,
                                )
            
            print("After attack:", x.shape)
            x = x[:1]

        base_pred = torch.argmax(base_classifier(x), dim=1)   # f(x)

        x = torch.squeeze(x)
        smooth_pred = smooth_classifier.predict(x, args.N, args.alpha, args.batch_size)
        
        if base_pred == smooth_pred:    # checks whether f(x) == g(x)
            base_smoothed_agree += 1
        
        time_elapsed = round(time.time() - start_time, 4)
        num_corr_pred += int(smooth_pred == label)
        num_samples_proc += x.shape[0]
        running_test_acc = round(num_corr_pred / num_samples_proc, 4)

        print("Itr:{}/{} \t Label (GT/Pred): {}/{} \t Acc: {} \t Time: {} s".format(itr, num_test_samples, label.item(), smooth_pred, running_test_acc, time_elapsed), file=outfile, flush=True)
