import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import sys
sys.path.append('..')
import models
import dataset as loader
import numpy as np
import torch.nn.functional as F
from loss import Myloss
from utils.seed import seed_torch
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from _datetime import datetime
from metrics import *
from models import Classifier

seed_torch()
parser = argparse.ArgumentParser(description='PyTorch CUB Training')





parser.add_argument('-lambda1', default=1e-5, type=float, metavar='N',
                    help='argument lambda1 (default: 0)')
parser.add_argument('-lambda2', default=0.001, type=float, metavar='N',
                    help='argument lambda2 (default: 0)')
parser.add_argument('-s1', default=10., type=float, metavar='N',
                    help='argument s1')
parser.add_argument('-m1', default=0, type=float, metavar='N',
                    help='argument m1')
parser.add_argument('-s2', default=10., type=float, metavar='N',
                    help='argument s2')
parser.add_argument('-m2', default=0.6, type=float, metavar='N',
                    help='argument m2')

parser.add_argument('--data', type=str, default='../CUB_200_2011',
                    help='dataset path')
parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                    help='batch size')
parser.add_argument('--pretrained', default=True, type=bool, help='pretrained')
parser.add_argument('--workers', default=4, type=int, help='workers')
parser.add_argument('--epochs', default=100, type=int, help='maxEpochs')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
parser.add_argument('--step_size', default=30, type=int, help='step_size')
parser.add_argument('--embedding-dim', default=256, type=int, help='embedding_dim')
parser.add_argument('--num-base-classes', default=100, type=int, help='num_base_classes')
parser.add_argument('--num-novel-classes', default=100, type=int, help='num_novel_classes')
parser.add_argument('--checkpoint', default='./imprint_ft', type=str, help='checkpoint dir')

parser.add_argument('--num-sample', default=1, type=int,
                    metavar='N', help='number of novel sample (default: 1)')
parser.add_argument('--random', action='store_true', help='whether use random novel weights')


best_novel_top1 = 0
best_novel_top5 = 0
best_all_top1 = 0
best_all_top5 = 0

args = parser.parse_args()

print(args)

def main():
    global best_novel_top1, best_novel_top5, best_all_top1, best_all_top5

    model = models.Net(embedding_dim=args.embedding_dim, num_classes=args.num_base_classes, pretrained=args.pretrained,
                       s=args.s2, m=args.m2).cuda()

    print('==> Loading from model checkpoint..')
    model_path = './pretrain/model_best.pth.tar'
    print('model_path: ', model_path)
    assert os.path.isfile(model_path), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model checkpoint '{}' (epoch {})"
          .format(model_path, checkpoint['epoch']))
    mkdir_p(args.checkpoint)
    # Data loading code
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    novel_dataset = loader.ImageLoader(
        args.data, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        train=True, num_classes=200,
        num_train_sample=args.num_sample,
        novel_only=True)

    novel_loader = torch.utils.data.DataLoader(
        novel_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    train_dataset = loader.ImageLoader(
        args.data, transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        train=True, num_classes=200,
        num_train_sample=args.num_sample)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_dataset.get_balanced_sampler(),
        num_workers=args.workers, pin_memory=True)

    # novel_only 测试集
    val_loader = torch.utils.data.DataLoader(
        loader.ImageLoader(args.data, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), num_classes=200, novel_only=True),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # all test
    all_val_loader = torch.utils.data.DataLoader(
        loader.ImageLoader(args.data, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), num_classes=200, novel_only=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    # imprint weights first
    imprint(novel_loader, model, args)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    criterion = Myloss(float(args.lambda2), num_classes=int(args.num_base_classes)+int(args.num_novel_classes)).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=float(args.lr), momentum=float(args.momentum), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.94)

    title = 'Impriningt + FT'

    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Test Novel Loss', 'Test All Loss', 'Train Acc.', 'Test Novel Top1', 'Test Novel Top5', 'Test All Top1', 'Test All Top5'])


    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        # train for one epoch
        train_loss, train_acc = train(novel_loader, model, criterion, optimizer, epoch)
        # train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
        # evaluate on validation set
        test_novel_loss, test_novel_acc, test_novel_acc5 = validate('novel', val_loader, model, criterion)
        test_all_loss, test_all_acc, test_all_acc5 = 0, 0, 0 #validate('all', all_val_loader, model, criterion)

        # append logger file
        logger.append([lr, train_loss, test_novel_loss, test_all_loss, train_acc, test_novel_acc, test_novel_acc5, test_all_acc, test_all_acc5])

        # remember best prec@1 and save checkpoint
        is_best = test_novel_acc > best_novel_top1
        best_novel_top1 = max(test_novel_acc, best_novel_top1)
        best_novel_top5 = max(test_novel_acc5, best_novel_top5)
        best_all_top1 = max(test_all_acc, best_all_top1)
        best_all_top5 = max(test_all_acc5, best_all_top5)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_novel_top1': best_novel_top1,
            'best_novel_top5': best_novel_top5,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))


def imprint(novel_loader, model, params):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    bar = Bar('Imprinting', max=len(novel_loader))
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(novel_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            input = input.cuda()

            # compute output
            output = model.extract(input)

            if batch_idx == 0:
                output_stack = output
                target_stack = target
            else:
                output_stack = torch.cat((output_stack, output), 0)
                target_stack = torch.cat((target_stack, target), 0)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                batch=batch_idx + 1,
                size=len(novel_loader),
                data=data_time.val,
                bt=batch_time.val,
                total=bar.elapsed_td,
                eta=bar.eta_td
            )
            bar.next()
        bar.finish()

    new_weight = torch.zeros(args.num_novel_classes, args.embedding_dim)
    for i in range(args.num_novel_classes):
        tmp = output_stack[target_stack == (i + args.num_base_classes)].mean(0) if not args.random else torch.randn(args.embedding_dim)
        new_weight[i] = tmp / tmp.norm(p=2)
    weight = torch.cat((model.classifier.fc.weight.data, new_weight.cuda()))
    model.classifier.fc = AddMarginProduct(args.num_base_classes + args.num_novel_classes, args.embedding_dim, s=args.s2, m=args.m2).cuda()
    model.classifier.fc.weight = nn.Parameter(weight)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    bar = Bar('Training  ', max=len(train_loader))
    for batch_idx, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # compute output
        feature_vectors, output = model(input, target)  # X is feature vector extracted from CNN

        loss = criterion(feature_vectors, output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # model.weight_norm()
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def validate(type, val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    bar = Bar(type+' Testing   ', max=len(val_loader))
    with torch.no_grad():
        end = time.time()
        for batch_idx, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda()
            target = target.cuda(non_blocking=True)

            # compute output
            # feature_vectors, output = model(input, target)
            feature_vectors, output = model.evaluate(input)
            loss = criterion(feature_vectors, output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(val_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg, top5.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()