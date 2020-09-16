import torch
import torch.nn as nn
import torch.nn.functional as F

class Prototypical(nn.Module):
    def __init__(self):
        super(Prototypical, self).__init__()

        self.zero = torch.tensor(0).cuda()
        self.criterion = torch.nn.CrossEntropyLoss()
        print('Initialized Prototypical Loss')

    def forward(self, support, query, label_g, label_e, model, use_GC=True):

        support = support - torch.mean(support, dim=3, keepdim=True)
        query = query - torch.mean(query, dim=3, keepdim=True)
        support, query, label_g = support.cuda(), query.cuda(), label_g.cuda()

        support = model(support)  # out size:(batch size, #classes), for softmax
        query = model(query)

        logit_e = F.linear(query, F.normalize(support))
        loss_e = self.criterion(logit_e, label_e)
        acc_e = self.accuracy(logit_e, label_e)

        loss_g = self.zero
        acc_g = self.zero
        if use_GC:
            inputs = torch.cat((support, query), dim=0)
            logit_g = F.linear(inputs, F.normalize(model.weight))
            loss_g = self.criterion(logit_g, label_g)
            acc_g = self.accuracy(logit_g, label_g)

        loss = loss_e + loss_g

        return loss, loss_e, loss_g, acc_e, acc_g

    def accuracy(self, logit, label):
        answer = (torch.max(logit, 1)[1].long().view(label.size()) == label).sum().item()
        n_total = logit.size(0)

        return answer / n_total
