from typing import List
import numpy as np
import torch
import torchvision
import torchvision.transforms as TVtransforms
import torchvision.datasets as TVdatasets


DATASETS_LIST = ["mnist", "cifar10", "cifar100", "imagenet"]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def load_mnist_data(data_dir, train_batch_size, test_batch_size, preprocess, kwargs):
#     mnist_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
#                                           transforms.CenterCrop(224),
#                                           transforms.ToTensor()])

    if preprocess is None:
        preprocess = TVtransforms.Compose([TVtransforms.ToTensor()])

    train_data = TVdatasets.MNIST(root=data_dir, 
                                  train=True,
                                  transform=preprocess,
                                  download=True)

    test_data = TVdatasets.MNIST(root=data_dir,
                                 train=False,
                                 transform=preprocess,
                                 download=True)

    print("Number of train samples:", len(train_data))
    print("Number of test samples:", len(test_data))

    trainloader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=train_batch_size,
                                             shuffle=True,
                                             **kwargs)

    testloader = torch.utils.data.DataLoader(dataset=test_data,
                                            batch_size=test_batch_size,
                                            shuffle=True,
                                            **kwargs)
    
    return trainloader, testloader


def load_cifar10_data(data_dir, train_batch_size, test_batch_size, preprocess, kwargs):
    if preprocess is None:
        preprocess = TVtransforms.Compose([TVtransforms.ToTensor(), 
                                            TVtransforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    train_data = TVdatasets.CIFAR10(root=data_dir, 
                                    train=True, 
                                    download=True, 
                                    transform=preprocess)

    test_data = TVdatasets.CIFAR10(root=data_dir, 
                                   train=False, 
                                   download=True, 
                                   transform=preprocess)
    
    classes = train_data.classes

    print("Number of train samples:", len(train_data))
    print("Number of test samples:", len(test_data))

    trainloader = torch.utils.data.DataLoader(dataset=train_data,
                                             batch_size=train_batch_size,
                                             shuffle=True,
                                             **kwargs)

    testloader = torch.utils.data.DataLoader(dataset=test_data,
                                             batch_size=test_batch_size,
                                             shuffle=True,
                                            **kwargs)

    return trainloader, testloader, classes


def get_normalized_layer(dataset: str, device: torch.device) -> torch.nn.Module:
    # return the dataset's normalization layer
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV, device)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV, device)
    

class NormalizeLayer(torch.nn.Module):
    def __init__(self, channel_mean: List[float], channel_std: List[float], device: torch.device):
        super(NormalizeLayer, self).__init__()
        self.channel_mean = torch.tensor(channel_mean).to(device)
        self.channel_std = torch.tensor(channel_std).to(device)
    
    def forward(self, input: torch.tensor):
        batch_size, num_channels, height, width = input.shape
        channel_mean = self.channel_mean.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        channel_std = self.channel_std.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)

        return (input - channel_mean) / channel_std
    

def get_num_classes(dataset_name: str):
    if dataset_name == "imagenet":
        return 1000
    elif dataset_name == "cifar10" or dataset_name == "mnist":
        return 10
    elif dataset_name == "cifar100":
        return 100


