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
parser.add_argument("--skip_examples", type=int, default=1, help="samples to skip")
parser.add_argument("--max_examples", type=int, default=-1, help="maximum samples to certify")

args = parser.parse_args()

print(args.dataset_name, args.base_classifier_path, args.sigma, args.batch_size, args.N0, args.N, args.alpha)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == "cuda" else {}

if __name__ == "__main__":
    # load the checkpoint from the base classifier
    model_checkpoint = torch.load(args.base_classifier_path)
    # print(model_checkpoint["arch"])
    base_classifier = models.get_architecture(model_checkpoint["arch"], args.dataset_name, device)
    base_classifier.load_state_dict(model_checkpoint['state_dict'])

    # create the smoothed classifier
    print("Number of classes: ", datasets.get_num_classes(args.dataset_name))
    smooth_classifier = SmoothClassifier(base_classifier, datasets.get_num_classes(args.dataset_name), args.sigma)

    # x = torch.randn((3, 32, 32), device=device)
    # prediction = smooth_classifier.predict(x, N=64, alpha=0.001, batch_size=1)
    # print(prediction)

    # get dataset and iterate through each sample
    # dataset = datasets.get_dataset(args.dataset_name, args.data_split)
    data_dir = "data/CIFAR10/"
    train_batch_size = 32
    test_batch_size = 1
    _, testloader, num_classes = datasets.load_cifar10_data(data_dir, train_batch_size, test_batch_size, None, kwargs)

    num_test_samples = len(testloader)
    num_corr_pred = 0
    
    for itr, (x, label) in enumerate(tqdm(testloader)):
        if itr % args.skip_examples != 0:
            continue
        
        if itr // args.skip_examples == args.max_examples:
            break

        x = x.to(device)
        start_time = time.time()
        prediction, radius = smooth_classifier.certify(x, args.N0, args.N, args.alpha, args.batch_size)
        time_elapsed = round(time.time() - start_time, 4)

        num_corr_pred += int(prediction == label)
        running_test_acc = round(num_corr_pred / (itr+1), 4)

        print("Iteration:{}/{} \t Label (GT/Pred): {}/{} \t Acc: {} \t Time: {} s".format(itr, num_test_samples, label.item(), prediction, running_test_acc, time_elapsed))
    
    print("Test accuracy:", round(num_corr_pred / num_test_samples, 4) * 100)