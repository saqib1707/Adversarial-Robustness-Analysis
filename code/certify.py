import os
import time
import argparse
import numpy as np
import torch

import datasets
import models
import core

parser = argparse.ArgumentParser(description="specify options for certification")
parser.add_argument("dataset_name", choices=datasets.DATASETS_LIST, help="select datasets")
parser.add_argument("base_classifier", type=str, help="path to the saved classifier model")
parser.add_argument("sigma", type=float, help="gaussian noise standard deviation hyperparameter")
parser.add_argument("--batch", type=int, default=32, help="batch size")
parser.add_argument("--N0", type=int, default=100, help="number of samples for selection")
parser.add_argument("--N", type=int, default=100000, help="number of samples for estimation")
parser.add_argument("--alpha", type=float, default=1e-3, help="failure probability")
args = parser.parse_args()

# print(args.dataset_name, args.base_classifier, args.sigma, args.batch, args.N0, args.N, args.alpha)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == "cuda" else {}


if __name__ == "__main__":
    # load the checkpoint from the base classifier
    model_checkpoint = torch.load(args.base_classifier)
    base_classifier = models.get_architecture(model_checkpoint["arch"], args.dataset_name, device)
    base_classifier.load_state_dict(model_checkpoint['state_dict'])

    # create the smoothed classifier
    smooth_classifier = core.SmoothClassifier(base_classifier, datasets.get_num_classes(args.dataset_name), args.sigma)

