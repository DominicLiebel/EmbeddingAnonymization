#train.py

from train_util import AverageMeter
import time
import torch


def train(epoch, data_loader, model, optimizer, criterion):

    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    for idx, (data, target) in enumerate(data_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        data = data.to(torch.float32)

        out = model(data)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(out, target)

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
                  .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses, top1=acc))

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct.item() / batch_size

    return acc

def validate(epoch, val_loader, model, criterion, file_path):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    if "cifar100" in file_path:
        num_class = 100
    else:
        num_class = 10
    cm =torch.zeros(num_class, num_class)

    # evaluation loop
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        data = data.to(torch.float32)

        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target)


        batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\tTime {iter_time.val:.3f} ({iter_time.avg:.3f})\t').
                  format(epoch, idx, len(val_loader), iter_time=iter_time, loss=losses, top1=acc))

    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()

    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    print("* Prec @1: {top1.avg:.4f}".format(top1=acc))

    return acc.avg, cm