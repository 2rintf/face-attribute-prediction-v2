import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from celeba import CelebA
from models import MultiTaskNetwork


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='czd\'s PyTorch Training')
parser.add_argument('--data', metavar='DIR',default='/home/czd-2019/Projects/celebA_dataset'
                    help='path to dataset')
parser.add_argument('--data-file', metavar='N',default="",type=str,
                    help='path to dataset file(train_part.txt/val_par.txt/test_part.txt)')

# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                     choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='No', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.(Default gpu-used)')


best_acc1 = 0

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    showParam(args)
    # exit()

    main_worker(args)


def main_worker(args):
    
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    NUM_WORKERS = args.workers
    DIR = args.data
    DATA_FILE_DIR = args.data_file
    LR = args.lr

    model = MultiTaskNetwork()
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # print(model)

    exit()
    # Multi-label 选择使用BCE损失函数。注意pytorch的BCELoss要求:
    # ① label是FloatTensor
    # ② BCEWithLogitsLoss = Sigmoid + BCELoss
    criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)

    optimizer1 = optim.Adam(model.parameters(), 
                            lr=args.lr,
                            )
    optimizer2 = torch.optim.AdamW(model.parameters(),
                                    lr=args.lr,
                                    )
    optimizer = optimizer1

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    [30,60],
                                                    gamma=0.3)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data Loading
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    train_dataset = CelebA(
        DIR,
        'train_part.txt',
        transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_dataset = CelebA(
        DIR,
        'val_part.txt',
        transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    test_dataset = CelebA(
        DIR,
        'test_part.txt',
        transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)


    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader,model,criterion,optimizer,epoch,args)

        scheduler.step()
        print(scheduler.get_lr())





def train(train_loader,model,criterion,optimizer,epoch,args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    loss = []
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # attribute labels order by 8 subtasks.
        targets = get_each_attr_label(target)
        
        targets = [t.type(torch.FloatTensor) for t in targets]

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)
            targets = [t.cuda(args.gpu,non_blocking=True)]
            


        # compute output
        hair,eyes,nose,cheek,mouth,chin,neck,holistic = model(images)

        loss_1 = criterion(holistic,targets[0])
        loss_2 = criterion(hair,targets[1])
        loss_3 = criterion(eyes,targets[2])
        loss_4 = criterion(nose,targets[3])
        loss_5 = criterion(cheek,targets[4])
        loss_6 = criterion(mouth,targets[5])
        loss_7 = criterion(chin,targets[6])
        loss_8 = criterion(neck,targets[7])
        
        total_loss = (loss_1+loss_2+loss_3+loss_4+loss_5+loss_6+loss_7+loss_8)/8.
        
        # loss = criterion(output, target)

        torch.softmax()

        # measure accuracy and record loss
        # TODO: new approach of calculating the accuracy. [Weighted Accuracy]
        acc1, acc5 = accuracy(output, target, topk=(1, 5))



        losses.update(total_loss.item(), images.size(0))



        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)





def criterion(y_pred, y_true, log_vars):
    '''
    UNDONE
    '''
    pass
    # loss = 0
    # for i in range(len(y_pred)):
    # precision = torch.exp(-log_vars[i])
    # diff = (y_pred[i]-y_true[i])**2.
    # loss += torch.sum(precision * diff + log_vars[i], -1)
    # return torch.mean(loss)

def get_each_attr_label(target):
    holistic_target = target[:,0:9]
    hair_target = target[:,9:19]
    eyes_target = target[:,19:24]
    nose_target = target[:,24:26]
    cheek_target = target[:,26:30]
    mouth_target = target[:,30:35]
    chin_target = target[:,35:38]
    neck_target = target[:,38:40]
    return holistic_target,hair_target,eyes_target,nose_target,
            cheek_target,mouth_target,chin_target,neck_target


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def showParam(args):
    print("Input params: ")
    print("\tDataset dir : %s \n \
        Dataset file dir : %s \n \
        Num workers : %d \n \
        Epochs : %d \n \
        Batch size : %d \n \
        LR : %f \n \
        Momentum(SGD) : %f \n \
        Weight decay(AdamW): %f \n \
        GPU : %d \n \
        Evaluate Mode : %d \n \
        Resume : %s \n \
        Print freq : %d" % 
            (
                args.data,
                args.data_file if args.data_file != "" else "None",
                args.workers,
                args.epochs,
                args.batch_size,
                args.lr,
                args.momentum,
                args.weight_decay,
                args.gpu if args.gpu != None else -1,
                int(args.evaluate==True),
                args.resume,
                args.print_freq
            ))

if __name__ == '__main__':
    main()