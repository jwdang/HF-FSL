import argparse
import os
import shutil
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import models
import dataset as loader
from loss import Myloss
from utils.seed import seed_torch
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from metrics import *

seed_torch()
parser = argparse.ArgumentParser(description='PyTorch CUB Training')

parser.add_argument('-lambda1', default=1e-5, type=float, metavar='N',
                    help='argument lambda (default: 0)')
parser.add_argument('-m1', default=0., type=float, metavar='N',
                    help='argument m1')
parser.add_argument('-s1', default=10., type=float, metavar='N',
                    help='argument s1')

parser.add_argument('--data', type=str, default='../CUB_200_2011',
                    help='dataset path')
parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                    help='batch size')
parser.add_argument('--pretrained', default=True, type=bool, help='pretrained')
parser.add_argument('--workers', default=4, type=int, help='workers')
parser.add_argument('--epochs', default=100, type=int, help='maxEpochs')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
parser.add_argument('--step_size', default=30, type=int, help='step_size')
parser.add_argument('--embedding-dim', default=256, type=int, help='embedding_dim')
parser.add_argument('--num-classes', default=100, type=int, help='num_classes')
parser.add_argument('--checkpoint', default='./pretrain', type=str, help='checkpoint dir')



best_prec1 = 0
best_prec5 = 0

args = parser.parse_args()

print(args)


def main():
    global best_prec1, best_prec5
    title = 'CUB'

    mkdir_p(args.checkpoint)
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Valid Top5.'])

    model = models.Net(embedding_dim=args.embedding_dim, num_classes=args.num_classes, pretrained=True,
                       s=args.s1, m=args.m1).cuda()

    criterion = Myloss(float(args.lambda1), num_classes=int(args.num_classes)).cuda()

    extractor_params = list(map(id, model.extractor.parameters()))
    classifier_params = list(filter(lambda p: id(p) not in extractor_params, model.parameters()))

    optimizer = torch.optim.SGD([
                {'params': model.extractor.parameters()},
                {'params': classifier_params, 'lr': args.lr * 10},
    ], lr=args.lr, momentum=args.momentum, weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_dataset = loader.ImageLoader(
        args.data,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), train=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        loader.ImageLoader(args.data, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    for epoch in range(args.epochs):
        # print('s:', model.classifier.fc.s)
        lr = optimizer.param_groups[1]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
        test_loss, test_acc, test_prec5 = validate(val_loader, model, criterion)

        logger.append([lr, train_loss, test_loss, train_acc, test_acc, test_prec5])
        is_best = test_acc > best_prec1
        best_prec1 = max(test_acc, best_prec1)
        best_prec5 = max(test_prec5, best_prec5)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
    print('Best acc:')
    print(best_prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    bar = Bar('Training', max=len(train_loader))
    for batch_idx, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # compute output
        feature_vectors, output = model(input, target)
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

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
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


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    bar = Bar('Testing ', max=len(val_loader))
    with torch.no_grad():
        end = time.time()
        for batch_idx, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda()
            target = target.cuda(non_blocking=True)

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
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
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