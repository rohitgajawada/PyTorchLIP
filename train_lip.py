import torch
import torch.nn as nn
import numpy as np
import deeplab_resnet
import cv2
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys
import os
import random
import argparse

import opts
import setup
import data
import datasets

parser = opts.myargparser()

def loss_calc(out, label,gpu0):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label[:,:,0,:].transpose(2,0,1)
    label = torch.from_numpy(label).long()
    label = Variable(label).cuda(gpu0)
    m = nn.LogSoftmax()
    criterion = nn.NLLLoss2d()
    out = m(out)
    return criterion(out,label)

def lr_poly(base_lr, iter,max_iter,power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []
    b.append(model.Scale.conv1)
    b.append(model.Scale.bn1)
    b.append(model.Scale.layer1)
    b.append(model.Scale.layer2)
    b.append(model.Scale.layer3)
    b.append(model.Scale.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.Scale.layer5.parameters())
    for j in range(len(b)):
        for i in b[j]:
            yield i

def main():

    opt = parser.parse_args()
    print(opt)

    trainloader = datasets.init_data.load_data(opt)

    model = deeplab_resnet.Res_Deeplab(21)
    saved_state_dict = torch.load('/data/MS_DeepLab_resnet_pretrained_COCO_init.pth')
    model.load_state_dict(saved_state_dict)

    max_iter = opt.maxIter
    batch_size = 1
    weight_decay = opt.wtDecay
    base_lr = opt.lr

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': opt.base_lr }, {'params': get_10x_lr_params(model), 'lr': 10*base_lr} ], lr = base_lr, momentum = 0.9,weight_decay = weight_decay)

    for i, data in enumerate(trainloader, 0):
        images, labels = data
        images = Variable(images).cuda()
        out = model(images)

        loss = loss_calc(out[0], label[0], gpu0)
        iter_size = int(args['--iterSize'])
        for i in range(len(out)-1):
            loss = loss + loss_calc(out[i+1],label[i+1],gpu0)

        loss = loss/iter_size
        loss.backward()

        if iter %1 == 0:
            print 'iter = ',iter, 'of',max_iter,'completed, loss = ', iter_size*(loss.data.cpu().numpy())

        if iter % iter_size  == 0:
            optimizer.step()
            lr_ = lr_poly(base_lr,iter,max_iter,0.9)
            print '(poly lr policy) learning rate',lr_
            optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_ }, {'params': get_10x_lr_params(model), 'lr': 10*lr_} ], lr = lr_, momentum = 0.9,weight_decay = weight_decay)
            optimizer.zero_grad()

        if iter % 1000 == 0 and iter!=0:
            print 'taking snapshot ...'
            torch.save(model.state_dict(),'data/snapshots/VOC12_scenes_'+str(iter)+'.pth')


if __name__ == '__main__':
    main()
