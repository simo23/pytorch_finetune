import argparse
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torchvision.utils
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import models
import utils as utils
from utils import AverageMeter, accuracy
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-e','--exp_dir', type=str, dest='exp_dir', help='path to experiment folder')

best_prec1 = 0.0
args = None
conf = None
logger = None

def main():
    global args, best_prec1, logger, conf
    args = parser.parse_args()

    if not os.path.isdir(args.exp_dir):
        raise ValueError('E: insert a valid experiment folder, {} seems wrong'.format(args.exp_folder))

    gpus = [int(gpu) for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    
    # Load configuration
    conf = utils.load_config(os.path.join(args.exp_dir,"conf.json"))
    log_dir = args.exp_dir
    shutil.copyfile("./train.py",os.path.join(args.exp_dir,"backup_train.py"))
    shutil.copyfile("./utils.py",os.path.join(args.exp_dir,"backup_utils.py"))
    logger = SummaryWriter(log_dir)

    # Create model
    model_params = utils.get_model_params(conf["network"])
    model = models.__dict__[conf["network"]["arch"]](**model_params)

    # Set trainable scopes ( placed BEFORE DataParallel which modifies the name of the parameters with "module")
    set_trainable_params(model,conf["network"]["trainable_scopes"])
    trainable_parameters =  filter(lambda p: p.requires_grad, model.parameters())
    optimizer, scheduler = utils.create_optimizer(conf["optimizer"], trainable_parameters)

    n_gpus = len(gpus)
    if n_gpus == 1:
        model = model.cuda()
    else: 
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if conf["network"]["past_ckpt"]:
        if os.path.isfile(conf["network"]["past_ckpt"]):
            print("=> loading checkpoint '{}'".format(conf["network"]["past_ckpt"]))
            checkpoint = torch.load(conf["network"]["past_ckpt"])
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(conf["network"]["past_ckpt"], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(conf["network"]["past_ckpt"]))
    else:
        args.start_epoch = 0

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(conf["dataset"]["base_folder"], 'train')
    valdir = os.path.join(conf["dataset"]["base_folder"], 'val')

    train_transforms, val_transforms = utils.create_transforms(conf["input"])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose(train_transforms)), 
        batch_size=conf["optimizer"]["train_batch_size"] * n_gpus, 
        shuffle=True,
        num_workers=conf["general"]["num_workers"])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose(val_transforms)),
        batch_size=conf["optimizer"]["test_batch_size"] * n_gpus,
        shuffle=False,
        num_workers=conf["general"]["num_workers"])

    for epoch in range(args.start_epoch, conf["optimizer"]["schedule"]["epochs"]):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, it=epoch * len(train_loader))

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': conf["network"]["arch"],
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, log_dir)


def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    global logger, conf
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if conf["optimizer"]["schedule"]["mode"] == "epoch":
        scheduler.step(epoch)
    else:
        raise KeyError("unrecognized schedule mode {}".format(conf["optimizer"]["schedule"]["mode"]))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input = input.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % conf["general"]["log_freq"] == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}\t'
                  'Data {data_time.val:.3f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

        it = i + epoch * len(train_loader)
        logger.add_scalar("train/loss", losses.val, it)
        logger.add_scalar("train/lr", scheduler.get_lr()[0], it)
        logger.add_scalar("train/top1", top1.val, it)
        logger.add_scalar("train/top5", top5.val, it)


def validate(val_loader, model, criterion, it=None):
    global logger, best_prec1
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():

        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            target = target.cuda()
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % conf["general"]["log_freq"] == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'BestPrec@1 {best:.3f}'.format(
                    i, len(val_loader), batch_time=batch_time, 
                    loss=losses, top1=top1, top5=top5, best=best_prec1))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} BestPrec@1 {best}'.format(top1=top1, top5=top5, best=best_prec1))

        logger.add_scalar("val/loss", losses.avg, it)
        logger.add_scalar("val/top1", top1.avg, it)
        logger.add_scalar("val/top5", top5.avg, it)
        logger.add_scalar("val/best1", best_prec1, it)

    return top1.avg


def save_checkpoint(state, is_best, path):
    torch.save(state,os.path.join(path,'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(path,'checkpoint.pth.tar'),os.path.join(path,'model_best.pth.tar'))

def set_trainable_params(model, trainable_scopes):

    if trainable_scopes[0] == "all":
        print("-> Training ALL the parameters")
        for param in model.parameters():
            param.requires_grad = True
    else:
        # Disable grad on all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Enable grad on selected scopes
        for module in trainable_scopes:
            print("-> Training module {}".format(module))
            if hasattr(model, module):
                for param in getattr(model, module).parameters():
                    param.requires_grad = True
            else:
                print("W: Module {} not recognized".format(module))

if __name__ == '__main__':
    main()
