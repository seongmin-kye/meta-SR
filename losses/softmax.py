import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()

        self.criterion = torch.nn.CrossEntropyLoss()
        print('Initialized Softmax Loss')

    def forward(self, inputs, label, model):

        inputs = inputs - torch.mean(inputs, dim=3, keepdim=True)
        inputs = inputs.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        inputs = model(inputs)
        logit = F.linear(inputs, F.normalize(model.weight))
        loss = self.criterion(logit, label)
        acc = self.accuracy(logit, label)

        return loss, acc

    def accuracy(self, logit, label):
        answer = (torch.max(logit, 1)[1].long().view(label.size()) == label).sum().item()
        n_total = logit.size(0)

        return answer / n_total