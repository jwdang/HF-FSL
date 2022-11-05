import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import Parameter
from metrics import *


class Net(nn.Module):
    def __init__(self, embedding_dim, num_classes, pretrained, s, m, easy_margin=False):
        super(Net, self).__init__()
        self.extractor = Extractor(pretrained=pretrained)
        self.embedding = Embedding(embdim=embedding_dim)
        self.classifier = Classifier(num_classes, embdim=embedding_dim, s=s, m=m)

    def forward(self, x, target):
        x = self.extractor(x)
        x = self.embedding(x)
        feature_vectors = F.normalize(x)
        x = self.classifier(feature_vectors, target)
        return feature_vectors, x

    def evaluate(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        feature_vectors = F.normalize(x)
        return feature_vectors, F.linear(feature_vectors, F.normalize(self.classifier.fc.weight))

    def extract(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        x = self.l2_norm(x)
        return x

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def weight_norm(self):
        w = self.classifier.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier.fc.weight.data = w.div(norm.expand_as(w))

class Extractor(nn.Module):
    def __init__(self, pretrained):
        super(Extractor, self).__init__()
        basenet = models.resnet50(pretrained=pretrained)
        self.extractor = nn.Sequential(*list(basenet.children())[:-1])

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        return x

class Embedding(nn.Module):
    def __init__(self, embdim):
        super(Embedding, self).__init__()
        self.fc = nn.Linear(2048, embdim)
        
    def forward(self, x):
        x = self.fc(x)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes, embdim, s, m):
        super(Classifier, self).__init__()
        self.fc = AddMarginProduct(embdim, num_classes, s=s, m=m)

    def forward(self, feature_vectors, target):
        x = self.fc(feature_vectors, target)
        return x