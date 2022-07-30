import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

import torch
import torch.nn as nn
import torchvision.transforms as TVtransforms
import torchvision.datasets as TVdatasets
from tqdm.notebook import tqdm

import torchattacks
import datasets
import models


# define device type - cuda or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == "cuda" else {}


def adversarial_training(model, trainloader, attack, num_epochs, stepsize):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=stepsize)

    num_batches = len(trainloader)
    loss_lst = []

    for epoch in range(num_epochs):
        for itr, (batch_imgs, batch_labels) in enumerate(tqdm(trainloader)):
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            batch_adv_imgs = attack(batch_imgs, batch_labels)

            X_train = torch.cat((batch_adv_imgs, batch_imgs), 0)
            Y_train_gt = torch.cat((batch_labels, batch_labels), 0)
            Y_train_pred = model(X_train)

            loss_val = loss(Y_train_pred, Y_train_gt)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if itr % 100 == 0:
                print("Epoch [{}/{}], Iteration [{}/{}], Loss:{}".format(epoch+1, num_epochs, 
                                                                            itr+1, num_batches, 
                                                                            round(loss_val.item(),4)))

            loss_lst.append(loss_val.item())
    
    plt.figure(figsize=(10,7))
    plt.plot(loss_lst)
    plt.xlabel("iteration")
    plt.ylabel("objective function")
    plt.grid()
    plt.show()


def evaluate_accuracy(model, dataloader, attack_obj=None):
    num_samples = 0
    num_corr_pred = 0
    model.eval()

    for itr, (batch_imgs, batch_labels) in enumerate(tqdm(dataloader)):
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        if attack_obj is None:
            X_test = batch_imgs
        else:
            X_test = attack_obj(batch_imgs, batch_labels)
        
        Y_test_gt = batch_labels
        Y_test_pred = torch.argmax(model(X_test), dim=1)

        num_samples += batch_imgs.shape[0]
        num_corr_pred += float((Y_test_pred == Y_test_gt).sum())

    test_acc = round(100.0 * num_corr_pred / num_samples, 4)

    return test_acc

def test_attack(model, trainloader, testloader, fgsm_attack, pgd_attack):
    adversarial_training(model, trainloader, pgd_attack, num_epochs=1, stepsize=1e-3)

    test_acc = evaluate_accuracy(model, testloader, None)
    print("Standard accuracy: {} %".format(test_acc))

    test_acc = evaluate_accuracy(model, testloader, fgsm_attack)
    print("Robust accuracy (FGSM attack): {} %".format(test_acc))

    test_acc = evaluate_accuracy(model, testloader, pgd_attack)
    print("Robust accuracy (PGD attack): {} %".format(test_acc))




if __name__ == "__main__":
    test_model = models.CNN_net().to(device)

    train_batch_size = 32
    test_batch_size = 32

    # attack methods
    adv_eps = 10/255
    adv_alpha = 2/255
    adv_pgd_steps = 10

    fgsm_attack = torchattacks.FGSM(test_model, eps=adv_eps)
    pgd_attack = torchattacks.PGD(test_model, eps=adv_eps, alpha=adv_alpha, steps=adv_pgd_steps)

    # MNIST dataset
    data_dir = "../data/"
    trainloader, testloader = datasets.load_MNIST_data(data_dir, train_batch_size, test_batch_size, None, kwargs)
    print("Evaluation on MNIST dataset")
    test_attack(test_model, trainloader, testloader, fgsm_attack, pgd_attack)

    # CIFAR10 dataset
    data_dir = "../data/CIFAR10/"
    trainloader, testloader, _ = datasets.load_CIFAR10_data(data_dir, train_batch_size, test_batch_size, None, kwargs)
    print("Evaluation on CIFAR10 dataset")
    test_attack(test_model, trainloader, testloader, fgsm_attack, pgd_attack)
    