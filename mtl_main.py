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

from torch.utils.tensorboard import SummaryWriter

from celeba import CelebA
from models import MultiTaskNetwork


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='czd\'s PyTorch Training')
parser.add_argument('--data', metavar='DIR',default='/home/czd-2019/Projects/celebA_dataset',
                    help='path to dataset')
parser.add_argument('--data-file', metavar='N',default="/home/czd-2019/Projects/face-attribute-prediction-v2/data_preprocess",
                    type=str,
                    help='path to dataset file(train_part.txt/val_par.txt/test_part.txt)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
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
    optimizer = optimizer2

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    [30],
                                                    gamma=0.3)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data Loading
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    train_dataset = CelebA(
        DIR,
        os.path.join(DATA_FILE_DIR,'train_part.txt'),
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
        os.path.join(DATA_FILE_DIR,'val_part.txt'),
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
        os.path.join(DATA_FILE_DIR,'test_part.txt'),
        transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return


    for epoch in range(args.start_epoch, args.epochs):
        total_loss = train(train_loader,model,criterion,optimizer,epoch,args)

        scheduler.step()
        print(scheduler.get_lr())

        # save checkpoint
        save_name = "checkpoint_epoch"+str(epoch+1)+".pth"
        print("checkpoint. Save name:"+save_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, "./checkpoint/"+save_name)

    

def train(train_loader,model,criterion,optimizer,epoch,args):
    batch_time = AverageMeter('Time:', ':6.3f')
    data_time = AverageMeter('Data:', ':6.3f')
    losses = AverageMeter('Total Loss:', ':.4e')
    losses_1 = AverageMeter('Loss1:', ':.4e')
    losses_2 = AverageMeter('Loss2:', ':.4e')
    losses_3= AverageMeter('Loss3:', ':.4e')
    losses_4 = AverageMeter('Loss4:', ':.4e')
    losses_5 = AverageMeter('Loss5:', ':.4e')
    losses_6 = AverageMeter('Loss6:', ':.4e')
    losses_7 = AverageMeter('Loss7:', ':.4e')
    losses_8 = AverageMeter('Loss8:', ':.4e')

    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, losses_1,losses_2,losses_3,losses_4,losses_5,losses_6,losses_7,losses_8],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    # loss = []
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
            targets = [t.cuda(args.gpu,non_blocking=True) for t in targets]

        # compute output
        hair,eyes,nose,cheek,mouth,chin,neck,holistic = model(images)

        print(targets[0].size())
        print(holistic.size())

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

        # accuracy
        # measure accuracy 
        # new approach of calculating the accuracy. [Weighted Accuracy]
        err_1 = 1 - sub_task_accuracy(torch.sigmoid(holistic),targets[0]) 
        err_2 = 1 - sub_task_accuracy(torch.sigmoid(hair),targets[1])
        err_3 = 1 - sub_task_accuracy(torch.sigmoid(eyes),targets[2])
        err_4 = 1 - sub_task_accuracy(torch.sigmoid(nose),targets[3])
        err_5 = 1 - sub_task_accuracy(torch.sigmoid(cheek),targets[4])
        err_6 = 1 - sub_task_accuracy(torch.sigmoid(mouth),targets[5])
        err_7 = 1 - sub_task_accuracy(torch.sigmoid(chin),targets[6])
        err_8 = 1 - sub_task_accuracy(torch.sigmoid(neck),targets[7])

        # TODO:record loss
        losses_1.update(loss_1.item(),images.size(0))
        losses_2.update(loss_2.item(),images.size(0))
        losses_3.update(loss_3.item(),images.size(0))
        losses_4.update(loss_4.item(),images.size(0))
        losses_5.update(loss_5.item(),images.size(0))
        losses_6.update(loss_6.item(),images.size(0))
        losses_7.update(loss_7.item(),images.size(0))
        losses_8.update(loss_8.item(),images.size(0))


        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(total_loss.item(), images.size(0))

        TB.add_scalar(
            "Training loss",
            total_loss.item(),
            epoch*len(train_loader)+i
        )
        TB.add_scalars(
            "Error Rate of Each Subtask",
            {
                'Holistic':err_1,
                'Hair':err_2,
                'Eyes':err_3,
                'Nose':err_4,
                'Cheek':err_5,
                'Mouth':err_6,
                'Chin':err_7,
                'Neck':err_8
            },
            epoch*len(train_loader)+i
        )

        # TODO:record error rate.

        # top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        # exit()
    return total_loss

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Total Loss:', ':.4e')
    # losses_1 = AverageMeter('Loss1:', ':.4e')
    # losses_2 = AverageMeter('Loss2:', ':.4e')
    # losses_3= AverageMeter('Loss3:', ':.4e')
    # losses_4 = AverageMeter('Loss4:', ':.4e')
    # losses_5 = AverageMeter('Loss5:', ':.4e')
    # losses_6 = AverageMeter('Loss6:', ':.4e')
    # losses_7 = AverageMeter('Loss7:', ':.4e')
    # losses_8 = AverageMeter('Loss8:', ':.4e')

    errs_1 = AverageMeter('Err1:',':4e')
    errs_2 = AverageMeter('Err2:',':4e')
    errs_3 = AverageMeter('Err3:',':4e')
    errs_4 = AverageMeter('Err4:',':4e')
    errs_5 = AverageMeter('Err5:',':4e')
    errs_6 = AverageMeter('Err6:',':4e')
    errs_7 = AverageMeter('Err7:',':4e')
    errs_8 = AverageMeter('Err8:',':4e')
    


    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time,losses, errs_1,errs_2,errs_3,errs_4,errs_5,errs_6,errs_7,errs_8],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            targets = get_each_attr_label(target)
        
            targets = [t.type(torch.FloatTensor) for t in targets]

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                # target = target.cuda(args.gpu, non_blocking=True)
                targets = [t.cuda(args.gpu,non_blocking=True) for t in targets]

            # compute output
            # output = model(images)
            # loss = criterion(output, target)

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

            err_1 = 1 - sub_task_accuracy(torch.sigmoid(holistic),targets[0])
            err_2 = 1 - sub_task_accuracy(torch.sigmoid(hair),targets[1])
            err_3 = 1 - sub_task_accuracy(torch.sigmoid(eyes),targets[2])
            err_4 = 1 - sub_task_accuracy(torch.sigmoid(nose),targets[3])
            err_5 = 1 - sub_task_accuracy(torch.sigmoid(cheek),targets[4])
            err_6 = 1 - sub_task_accuracy(torch.sigmoid(mouth),targets[5])
            err_7 = 1 - sub_task_accuracy(torch.sigmoid(chin),targets[6])
            err_8 = 1 - sub_task_accuracy(torch.sigmoid(neck),targets[7])


            
            # measure accuracy and record loss
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(total_loss.item(), images.size(0))

            errs_1.update(err_1)
            errs_2.update(err_2)
            errs_3.update(err_3)
            errs_4.update(err_4)
            errs_5.update(err_5)
            errs_6.update(err_6)
            errs_7.update(err_7)
            errs_8.update(err_8)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))

        print('[FINAL] Err1:{err1.avg:.3f}  Err2:{err2.avg:.3f}  \
Err3:{err3.avg:.3f}  Err4:{err4.avg:.3f}  Err5:{err5.avg:.3f}  Err6:{err6.avg:.3f}  Err7:{err7.avg:.3f}  Err8:{err8.avg:.3f}'
                .format(err1=errs_1,
                        err2=errs_2,
                        err3=errs_3,
                        err4=errs_4,
                        err5=errs_5,
                        err6=errs_6,
                        err7=errs_7,
                        err8=errs_8))

    return -1



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


def sub_task_accuracy(model_pred,labels,threshold=0.8):
    '''
        Return accuracy.
    '''
    print(model_pred[0:3,:])
    pred_result = model_pred > threshold
    label_temp = labels>0

    acc_mat = (pred_result.eq(label_temp)).data.cpu().sum(1).type(torch.FloatTensor)
    acc = torch.mean(acc_mat/labels.size(1))

    return  acc



def get_each_attr_label(target):
    holistic_target = target[:,0:9]
    hair_target = target[:,9:19]
    eyes_target = target[:,19:24]
    nose_target = target[:,24:26]
    cheek_target = target[:,26:30]
    mouth_target = target[:,30:35]
    chin_target = target[:,35:38]
    neck_target = target[:,38:40]
    return holistic_target,hair_target,eyes_target,nose_target,cheek_target,mouth_target,chin_target,neck_target


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
        fmtstr = '{name} {val' + self.fmt + '}({avg' + self.fmt + '})'
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
        Print freq : %d "% 
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
    TB = SummaryWriter()
    main()