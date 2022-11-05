import torch.nn as nn
import torch
import torch.nn.functional as F



class Myloss(nn.Module):
    def __init__(self, lambd, num_classes):
        super(Myloss, self).__init__()
        self.lambd = lambd
        self.num_classes = num_classes
        self.part2func = Part2.apply

    def regularization(self, X, target):
        # 上半部分
        Y = F.one_hot(target, num_classes=self.num_classes).float()
        Y = torch.mm(Y, Y.T)
        Y = torch.ones_like(Y) - Y
        part1 = (torch.sum(torch.mul(torch.mm(X, X.T), Y)) / X.shape[0]) ** 2
        part2 = self.part2func(X, target, self.num_classes)
        # return part2
        return part1 + part2

    def forward(self, X, output, target):
        '''
            X: feature_vectors extracted from CNN
            output:  predicted labels from the model
            target:  real labels
        '''
        reg = 0
        if self.lambd != 0:
            reg = self.regularization(X, target)
            # reg = self.regularization(X, target) / float(target.shape[0])
        return nn.CrossEntropyLoss()(output, target) + self.lambd * reg
        
class Part2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, target, num_classes):
        ctx.X = X
        ctx.target = target
        ctx.num_classes = num_classes
        Y = F.one_hot(target, num_classes=num_classes).float()
        Y = torch.mm(Y, Y.T)
        Y = torch.ones_like(Y) - Y
        t = (torch.sum(torch.mul(torch.mm(X, X.T) ** 2, Y)) / target.shape[0]) - 1 / X.shape[1]

        ctx.t = t
        part2 = t if t > 0 else torch.FloatTensor([0]).cuda()

        return part2

    @staticmethod
    def backward(ctx, grad_output):
        X = ctx.X
        target = ctx.target
        num_classes = ctx.num_classes
        t = ctx.t
        if t<0:
            return torch.zeros(X.shape[0], X.shape[1]).cuda(), None
        else:
            Y = F.one_hot(target, num_classes=num_classes).float()
            Y = torch.mm(Y, Y.T)
            Y = torch.ones_like(Y) - Y
            return grad_output * ( 2 * torch.sum(torch.mul(torch.mm(X, X.T), Y)) ) / X.shape[0] * torch.ones(X.shape[0], X.shape[1]).cuda(), None, None