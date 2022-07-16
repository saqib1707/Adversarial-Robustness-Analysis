import numpy as np
import torch
import torchvision
import torchvision.transforms as TVtransforms
import torchvision.datasets as TVdatasets


def load_MNIST_data(data_dir, train_batch_size, test_batch_size, preprocess, kwargs):
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


def load_CIFAR10_data(data_dir, train_batch_size, test_batch_size, preprocess, kwargs):
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