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
parser.add_argument('-t', '--test', dest='test', action='store_true',
                    help='evaluate model on test set')

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.(Default gpu-used)')


best_acc1 = 0

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args = parser.parse_args()

    showParam(args)

    main_worker(args)


def main_worker(args):
    global best_acc1

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    NUM_WORKERS = args.workers
    DIR = args.data
    DATA_FILE_DIR = args.data_file
    LR = args.lr

    
    model = MultiTaskNetwork(isPretrained=args.pretrained)
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # log_vars = []
    log_vars = torch.zeros((8,),requires_grad=True,device="cuda")
    # log_vars.retain_grad = True
    # log_vars = log_vars.cuda()
    # log_vars.requires_grad = True
    # log_vars.append(log_vars)
    
    # print(len(log_vars))
    # print(model)
    # print(model.parameters)
    
    params =([p for p in model.parameters()]+[log_vars])
    # print(params)
    # print(model.parameters())
    


    # Multi-label 选择使用BCE损失函数。注意pytorch的BCELoss要求:
    # ① label是FloatTensor
    # ② BCEWithLogitsLoss = Sigmoid + BCELoss
    BCECriterion = nn.BCEWithLogitsLoss(reduction='none').cuda(args.gpu)

    optimizer1 = optim.Adam(params, 
                            lr=args.lr,
                            )
    optimizer2 = torch.optim.AdamW(params,
                                    lr=args.lr,
                                    )
    optimizer3 = torch.optim.SGD(params,lr=args.lr,momentum=0.9)
    optimizer = optimizer3

    # For Adam(AdamW)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    [30],
                                                    gamma=0.3)
    # For SGD
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    [20,40],
                                                    gamma=0.1)
    scheduler_temp = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    [4],
                                                    gamma=0.1)
    scheduler = scheduler2


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # val_acc = checkpoint['val_acc']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # todo: Save lr_scheduler.

            
            log_vars=checkpoint['log_var']

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            # print("=> val accuracy is {}"
            #         .format(val_acc))
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
        validate(val_loader, model, BCECriterion,log_vars, args)
        return
    
    if args.test:
        test(test_loader, model, BCECriterion,log_vars, args)
        return


    for epoch in range(args.start_epoch, args.epochs):
        total_loss = train(train_loader,model,BCECriterion,log_vars,optimizer,epoch,args)

        scheduler.step()
        print(scheduler.get_lr())

        val_acc = validate(val_loader,model,BCECriterion,log_vars,args)

        is_best = val_acc > best_acc1
        best_acc1 = max(val_acc, best_acc1)

        # save checkpoint
        save_name = "checkpoint_epoch"+str(epoch+1)+".pth"
        print("checkpoint. Save name:"+save_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            'val_acc':val_acc,
            'best_acc':best_acc1,
            'log_var':log_vars
        }, "./checkpoint/"+save_name)

    print("Start eval on test data.")
    test(test_loader, model, BCECriterion,log_vars, args)

    

def train(train_loader,model,BCECriterion,log_vars,optimizer,epoch,args):
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
        hair, eyes, nose, cheek, mouth, chin, neck, holistic = model(images)


        loss_1 = UWLossCriterion(holistic,targets[0],log_vars[0],BCECriterion)
        loss_2 = UWLossCriterion(hair,targets[1],log_vars[1],BCECriterion)
        loss_3 = UWLossCriterion(eyes,targets[2],log_vars[2],BCECriterion)
        loss_4 = UWLossCriterion(nose,targets[3],log_vars[3],BCECriterion)
        loss_5 = UWLossCriterion(cheek,targets[4],log_vars[4],BCECriterion)
        loss_6 = UWLossCriterion(mouth,targets[5],log_vars[5],BCECriterion)
        loss_7 = UWLossCriterion(chin,targets[6],log_vars[6],BCECriterion)
        loss_8 = UWLossCriterion(neck,targets[7],log_vars[7],BCECriterion)

        # loss_1 = criterion(holistic,targets[0])
        # loss_2 = criterion(hair,targets[1])
        # loss_3 = criterion(eyes,targets[2])
        # loss_4 = criterion(nose,targets[3])
        # loss_5 = criterion(cheek,targets[4])
        # loss_6 = criterion(mouth,targets[5])
        # loss_7 = criterion(chin,targets[6])
        # loss_8 = criterion(neck,targets[7])
        
        total_loss = loss_1+loss_2+loss_3+loss_4+loss_5+loss_6+loss_7+loss_8

        # accuracy
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

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            print(log_vars)

        # exit()
    return total_loss

def validate(val_loader, model,BCECriterion,log_vars,args):
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

            loss_1 = UWLossCriterion(holistic,targets[0],log_vars[0],BCECriterion)
            loss_2 = UWLossCriterion(hair,targets[1],log_vars[1],BCECriterion)
            loss_3 = UWLossCriterion(eyes,targets[2],log_vars[2],BCECriterion)
            loss_4 = UWLossCriterion(nose,targets[3],log_vars[3],BCECriterion)
            loss_5 = UWLossCriterion(cheek,targets[4],log_vars[4],BCECriterion)
            loss_6 = UWLossCriterion(mouth,targets[5],log_vars[5],BCECriterion)
            loss_7 = UWLossCriterion(chin,targets[6],log_vars[6],BCECriterion)
            loss_8 = UWLossCriterion(neck,targets[7],log_vars[7],BCECriterion)

            # loss_1 = criterion(holistic,targets[0])
            # loss_2 = criterion(hair,targets[1])
            # loss_3 = criterion(eyes,targets[2])
            # loss_4 = criterion(nose,targets[3])
            # loss_5 = criterion(cheek,targets[4])
            # loss_6 = criterion(mouth,targets[5])
            # loss_7 = criterion(chin,targets[6])
            # loss_8 = criterion(neck,targets[7])

            total_loss = loss_1+loss_2+loss_3+loss_4+loss_5+loss_6+loss_7+loss_8

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
        avg_err = (errs_1.avg+errs_2.avg+errs_3.avg
                    +errs_4.avg+errs_5.avg+errs_6.avg
                    +errs_7.avg+errs_8.avg)/8
        print('[VAL RESULT] Avg_Err:{avg_err:.3f}  Err1:{err1.avg:.3f}  Err2:{err2.avg:.3f}  Err3:{err3.avg:.3f}  Err4:{err4.avg:.3f}  Err5:{err5.avg:.3f}  Err6:{err6.avg:.3f}  Err7:{err7.avg:.3f}  Err8:{err8.avg:.3f}'
                .format(avg_err=avg_err,
                        err1=errs_1,
                        err2=errs_2,
                        err3=errs_3,
                        err4=errs_4,
                        err5=errs_5,
                        err6=errs_6,
                        err7=errs_7,
                        err8=errs_8))

    return avg_err


def test(test_loader,model,BCECriterion,log_vars,args):
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
        len(test_loader),
        [batch_time,losses, errs_1,errs_2,errs_3,errs_4,errs_5,errs_6,errs_7,errs_8],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):

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


            loss_1 = UWLossCriterion(holistic,targets[0],log_vars[0],BCECriterion)
            loss_2 = UWLossCriterion(hair,targets[1],log_vars[1],BCECriterion)
            loss_3 = UWLossCriterion(eyes,targets[2],log_vars[2],BCECriterion)
            loss_4 = UWLossCriterion(nose,targets[3],log_vars[3],BCECriterion)
            loss_5 = UWLossCriterion(cheek,targets[4],log_vars[4],BCECriterion)
            loss_6 = UWLossCriterion(mouth,targets[5],log_vars[5],BCECriterion)
            loss_7 = UWLossCriterion(chin,targets[6],log_vars[6],BCECriterion)
            loss_8 = UWLossCriterion(neck,targets[7],log_vars[7],BCECriterion)

            # loss_1 = criterion(holistic,targets[0])
            # loss_2 = criterion(hair,targets[1])
            # loss_3 = criterion(eyes,targets[2])
            # loss_4 = criterion(nose,targets[3])
            # loss_5 = criterion(cheek,targets[4])
            # loss_6 = criterion(mouth,targets[5])
            # loss_7 = criterion(chin,targets[6])
            # loss_8 = criterion(neck,targets[7])

            total_loss = loss_1+loss_2+loss_3+loss_4+loss_5+loss_6+loss_7+loss_8

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
        
        avg_err = (errs_1.avg+errs_2.avg+errs_3.avg
                    +errs_4.avg+errs_5.avg+errs_6.avg
                    +errs_7.avg+errs_8.avg)/8
        print('[TEST RESULT] Avg_Err:{avg_err:.3f}  Err1:{err1.avg:.3f}  Err2:{err2.avg:.3f}  Err3:{err3.avg:.3f}  Err4:{err4.avg:.3f}  Err5:{err5.avg:.3f}  Err6:{err6.avg:.3f}  Err7:{err7.avg:.3f}  Err8:{err8.avg:.3f}'
                .format(avg_err=avg_err,
                        err1=errs_1,
                        err2=errs_2,
                        err3=errs_3,
                        err4=errs_4,
                        err5=errs_5,
                        err6=errs_6,
                        err7=errs_7,
                        err8=errs_8))

    return avg_err

def UWLossCriterion(output, target, log_var, BCECriterion):
    '''
    UNTEST
    '''
    # print(log_var)
    diff = BCECriterion(output,target)
    # print(diff)
    loss = torch.exp(-log_var) * diff + log_var
    # print(loss)
    loss = torch.sum(loss,-1)
    m_loss = torch.mean(loss)

    # if m_loss.item() < 0:
    #     exit()

    return m_loss

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
        Pretrained weight: %d \n \
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
                int(args.pretrained==True),
                int(args.evaluate==True),
                args.resume,
                args.print_freq
            ))

if __name__ == '__main__':
    args = parser.parse_args()
    TB = None
    if not (args.evaluate and args.test):
        TB = SummaryWriter()
        print("Start tensorboard.")
    main(args)