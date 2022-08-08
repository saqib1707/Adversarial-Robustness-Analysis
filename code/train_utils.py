import os
import torch
import torch.nn as nn


class AverageMeter(object):
    """computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n 
        self.count = self.count + n
        self.avg = self.sum / self.count


def compute_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the top-K predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]
        # print(maxk, batch_size, output.shape, target.shape)

        _, pred = torch.topk(output, k=maxk, dim=1, largest=True, sorted=True)
        # print(pred.shape)
        # pred = pred.t()
        # print(pred.shape)
        # print(target, target.shape)
        # print(target.view(-1, 1), target.view(-1, 1).shape)
        # print(target.view(-1, 1).expand_as(pred))
        correct = torch.eq(pred, target.view(-1, 1).expand_as(pred))
        # print("Correct shape:", correct.shape)

        # print(correct[:,:2].shape)
        # print(correct[:,:2].view(2, -1).shape)
        # print(correct[:,:2].float())
        # print(correct[:,:2].float().sum(dim=1, keepdim=True).shape)

        res = []
        for k in topk:
            correct_k = correct[:, :k].float().sum()
            # print(correct_k)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res


def requires_grad_(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)