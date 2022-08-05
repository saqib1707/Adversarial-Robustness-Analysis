"""This code loads a base classifier and runs CERTIFY (prediction + radius) on many examples from a dataset
"""
import os
import time
import argparse
import numpy as np
import torch
from tqdm.notebook import tqdm

import datasets
import models
from core import SmoothClassifier

parser = argparse.ArgumentParser(description="specify options for certification")
parser.add_argument("dataset_name", choices=datasets.DATASETS_LIST, help="select datasets")
parser.add_argument("base_classifier_path", type=str, help="path to the saved classifier model")
parser.add_argument("sigma", type=float, help="gaussian noise standard deviation hyperparameter")
parser.add_argument("--N0", type=int, default=100, help="number of samples for selection")
parser.add_argument("--N", type=int, default=100000, help="number of samples for estimation")
parser.add_argument("--alpha", type=float, default=1e-3, help="failure probability")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--skip_examples", type=int, default=1, help="number of samples to skip")
parser.add_argument("--max_examples", type=int, default=-1, help="stop after this many samples")
parser.add_argument("--num_workers", default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument("outfile_path", type=str, help="output file")

args = parser.parse_args()

# print(args.dataset_name, args.base_classifier_path, args.sigma, args.batch_size, args.N0, args.N, args.alpha)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if device.type == "cuda" else {}

if __name__ == "__main__":
    # load the checkpoint from the base classifier
    model_checkpoint = torch.load(args.base_classifier_path)
    # print(model_checkpoint["arch"])
    base_classifier = models.get_architecture(model_checkpoint["arch"], args.dataset_name, device)
    base_classifier.load_state_dict(model_checkpoint['state_dict'])
    base_classifier.eval()

    # create the smoothed classifier
    smooth_classifier = SmoothClassifier(base_classifier, datasets.get_num_classes(args.dataset_name), args.sigma)

    # prepare output file for logging
    outfile = open(args.outfile_path, 'w')
    # print("idx/ \t GT label \t Pred \t Radius \t Time taken", file=outfile, flush=True)

    # x = torch.randn((3, 32, 32), device=device)
    # prediction = smooth_classifier.predict(x, N=64, alpha=0.001, batch_size=1)
    # print(prediction)

    # get dataset and iterate through each sample
    print("Number of classes: ", datasets.get_num_classes(args.dataset_name))
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
    
    for itr, (x, label) in enumerate(tqdm(testloader)):
        if itr % args.skip_examples != 0:
            continue
        if itr // args.skip_examples == args.max_examples:
            break

        x = x.to(device)
        start_time = time.time()
        prediction, radius = smooth_classifier.certify(x, args.N0, args.N, args.alpha, args.batch_size)
        time_elapsed = round(time.time() - start_time, 4)
        num_samples_proc += x.shape[0]

        num_corr_pred += int(prediction == label)
        running_test_acc = round(num_corr_pred / num_samples_proc, 4)

        print("Itr:{}/{} \t Label (GT/Pred): {}/{} \t Acc: {} \t Time: {} s".format(itr, num_test_samples, label.item(), prediction, running_test_acc, time_elapsed), file=outfile, flush=True)
    
    print("Test accuracy:", round(num_corr_pred / num_samples_proc, 4) * 100, file=outfile, flush=True)

    outfile.close()