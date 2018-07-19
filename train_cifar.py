"""
    PyTorch training code for Wide Residual Networks:
    http://arxiv.org/abs/1605.07146
    The code reproduces *exactly* it's lua version:
    https://github.com/szagoruyko/wide-residual-networks
    2016 Sergey Zagoruyko
"""

import argparse
import os
import time
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models.wide_resnet_cifar import WideResNet
import numpy as np
from utils.monitors import meter
from utils.loggers.json_logger import JsonLogger

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--dataroot', default='.', type=str)
parser.add_argument('--nthread', default=4, type=int)

# Training options
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--schedule', nargs='*', default=[60,120,160], type=int,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--randomcrop_pad', default=4, type=float)

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')

def create_dataset(opt, train):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    if train:
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            transform
        ])

    ds = getattr(datasets, opt.dataset)(opt.dataroot, train=train, download=True, transform=transform)
    if train:
        ds.train_data = np.pad(ds.train_data, ((0,0), (4,4), (4,4), (0,0)), mode='reflect')
    return ds


def main():
    opt = parser.parse_args()
    state = opt.__dict__
    log = JsonLogger(opt.save, rand_folder=True)
    log.update(state)
    state['exp_dir'] = os.path.dirname(log.path)
    state['start_lr'] = state['lr']
    state["training_time"] = 0

    print('parsed options:', vars(opt))
    num_classes = 10 if opt.dataset == 'CIFAR10' else 100

    def create_iterator(train):
        return DataLoader(create_dataset(opt, train), batch_size=opt.batch_size, shuffle=train,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)

    model = WideResNet(opt.depth, opt.width, num_classes).cuda()
    if opt.ngpu > 1:
        model = torch.nn.DataParallel(model, list(range(opt.ngpu)))

    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        return SGD(model.parameters(), lr, 0.9, weight_decay=opt.weight_decay)

    optimizer = create_optimizer(opt, opt.lr)

    state["epoch"] = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        state["epoch"] = state_dict['epoch']
        state["training_time"] = state_dict["training_time"]
        optimizer.load_state_dict(state_dict['optimizer'])
        model.load_state_dict(state_dict["state_dict"])

    n_parameters = sum([np.prod(p.size()) for p in model.parameters()])
    print('\nTotal number of parameters:', n_parameters)

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    train_loss_meter = meter.Meter()
    val_loss_meter = meter.Meter()
    val_accuracy_meter = meter.Meter()

    def train():
        model.train()
        train_loss_meter.reset()
        for images, labels in train_loader:
            images, labels = Variable(images, requires_grad=False).cuda(), Variable(labels, requires_grad=False).cuda()
            optimizer.zero_grad()
            t = time.time()
            preds = model(images)
            loss = F.cross_entropy(preds, labels)
            loss.backward()
            optimizer.step()
            state["training_time"] = time.time() - t
            train_loss_meter.update(float(loss), labels.size(0))
        state["train_loss"] = train_loss_meter.mean()

    def eval():
        model.eval()
        val_loss_meter.reset()
        val_accuracy_meter.reset()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                val_loss_meter.update(float(loss), outputs.size(0))
                preds = outputs.max(1)[1]
                val_accuracy_meter.update(float((preds == labels).float().sum()), outputs.size(0))
        state['val_loss'] = val_loss_meter.mean()
        state['val_accuracy'] = val_accuracy_meter.mean()

    start_epoch = state["epoch"]
    for epoch in range(start_epoch, opt.epochs):
        state["epoch"] = epoch
        train()
        eval()
        torch.save(dict(state_dict=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        training_time=state["training_time"],
                        epoch=epoch + 1),
                   open(os.path.join(state["exp_dir"], 'model.pt7'), 'wb'))
        log.update(state)
        print(state)
        print("epoch: %d - Validation accuracy: %.03f" %(epoch, state["val_accuracy"]))



        if (epoch + 1) in opt.schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay_ratio
            state['lr'] *= 0.1


if __name__ == '__main__':
    main()
