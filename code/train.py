import os
import argparse
import time
import numpy as np
import torch

import train_utils
import datasets
import models


parser = argparse.ArgumentParser(description="Train the ImageNet classifier")
parser.add_argument("dataset_name", choices=datasets.DATASETS_LIST, help="select datasets")
parser.add_argument('arch', type=str, choices=models.ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--num_epochs', default=5, type=int, metavar='N',
                    help='total epochs to run')
parser.add_argument('--train_batch_size', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--test_batch_size', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_std', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print_freq', default=10, type=int, metavar='N',
                    help='print training frequency (default: 10)')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if device.type == "cuda" else {}


def train():
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # get dataset and iterate through each sample
    if dataset_name == "cifar10":
        data_dir = "data/CIFAR10/"
        trainloader, testloader, num_classes = datasets.load_cifar10_data(data_dir, args.train_batch_size, args.test_batch_size, None, kwargs)
    elif dataset_name == "imagenet":
        data_dir = "data/ImageNet/"
        trainloader, testloader, num_classes = datasets.load_imagenet_data(data_dir, args.train_batch_size, args.test_batch_size, None, kwargs)

    base_classifier = models.get_architecture(args.arch, args.dataset_name, device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,           weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    for epoch in range(args.num_epochs):
        scheduler.step(epoch)
        start_time = time.time()
        
        # train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, args.noise_sd)
        # batch_time = AverageMeter()
        # data_time = AverageMeter()
        # losses = AverageMeter()
        # top1 = AverageMeter()
        # top5 = AverageMeter()
        # end = time.time()

        # switch to train mode
        base_classifier.train()

        for itr, (x_batch, y_batch) in enumerate(trainloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # augment inputs with noise
            x_batch = x_batch + torch.randn_like(x_batch, device=device) * args.noise_std

            # compute output
            y_pred = base_classifier(x_batch)
            loss_batch = criterion(y_pred, y_batch)

            # measure accuracy and record loss
            acc1, acc5 = train_utils.accuracy(y_pred, y_batch, topk=(1, 5))
            # losses.update(loss.item(), x_train.size(0))
            # top1.update(acc1.item(), x_train.size(0))
            # top5.update(acc5.item(), x_train.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

            if itr % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
        
        # test_loss, test_acc = test(testloader, model, criterion, args.noise_std)

        time_elapsed = round(time.time() - start_time, 4)

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'checkpoint.pth.tar'))


if __name__ == "__main__":
    train()
