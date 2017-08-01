#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import sys

sys.path.append("/mnt/git/Bilinear_CNN_dog_classifi/")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print sys.path
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from collections import OrderedDict
from resnet import resnet50, Bottleneck, resnet101
from inception import inception_v3
from dog_config import *
# from utils import data_loader
# from utils.data_loder import data_loader_
from utils.data_loader import data_loader_
# from data_augmentation import data_augmentation_img
from vggnet import vgg16
from load_image import load_data
import math


def train(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val.cuda(), requires_grad=False)
    y = Variable(y_val.cuda(), requires_grad=False)
    optimizer.zero_grad()
    fx = model.forward(x)
    # inecption 网络特有
    if type(fx) == tuple:
        t_y = fx[0].cpu().data.numpy().argmax(axis=1)
        acc = 1. * np.mean(t_y == y_val.numpy())
        output = loss.forward(fx[0], y)
    else:
        t_y = fx.cpu().data.numpy().argmax(axis=1)
        acc = 1. * np.mean(t_y == y_val.numpy())
        output = loss.forward(fx, y)
    output.backward()
    optimizer.step()
    return output.cuda().data[0], acc


def predict(model, x_val):
    x = Variable(x_val.cuda(), requires_grad=False)
    output = model.forward(x)
    if type(output) == tuple:
        return output[0].cpu().data.numpy().argmax(axis=1)
    return output.cpu().data.numpy().argmax(axis=1)


def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.lr / 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def preprocess_input(x):
    x /= 255.
    # x -= 0.5
    # x *= 2.
    return x

def main():
    torch.manual_seed(23)
    # Band_num = 2
    # Tag_id = 4
    data_l = data_loader_(batch_size=64,proportion=0.85, shuffle=True, data_add=2, onehot=False, data_size=224, nb_classes=100)
    print data_l.train_length
    print data_l.test_length
    # print 'loading....'
    # trX = np.load('bddog/trX.npy')
    # trY = np.load('bddog/trY.npy')
    # print 'load train data'
    # trX = torch.from_numpy(trX).float()
    # trY = torch.from_numpy(trY).long()
    # teX = np.load('bddog/teX.npy').astype(np.float)
    # teY = np.load('bddog/teY.npy')
    # print 'load test data'
    # teX[:, 0, ...] -= MEAN_VALUE[0]
    # teX[:, 1, ...] -= MEAN_VALUE[1]
    # teX[:, 2, ...] -= MEAN_VALUE[2]
    # teX = torch.from_numpy(teX).float()
    # teY = torch.from_numpy(teY).long()
    # print 'numpy data to tensor'
    # n_examples = len(trX)
    # n_classes = 100
    # model = torch.load('models/resnet_model_pretrained_adam_2_2_SGD_1.pkl')
    model = resnet50(pretrained=True, model_root=Model_Root)
    print '==============================='
    print model
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.classifier[-1] = nn.Linear(4096, 100)
    # n = model.classifier[-1].weight.size(1)
    # model.classifier[-1].weight.data.normal_(0, 0.01)
    # model.classifier[-1].bias.data.zero_()

    # VGG16 classifier层
    # model.classifier = nn.Sequential(
    #     nn.Linear(512 * 7 * 7, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(),
    #     nn.Linear(4096, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(),
    #     nn.Linear(4096, 100),
    # )
    # count = 0
    # print '==============================='
    # for module in model.modules():
    #     print '**** %d' % count
    #     print(module)
    #     count+=1
    # print '==============================='
    # count= 0
    # model.classifier[6] = nn.Linear(4096, 100)
    # for m in model.classifier:
    #     if count == 6:
    #         m = nn.Linear(4096, 100)
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             n = m.weight.size(1)
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()
    #     count+=1
    # try:
    #     print model.classifier[0]
    # except Exception as e:
    #     print e

    # print '==============================='
    # for module in model.modules()[-7:]:
    #     print '****'
    #     print(module)
    # resnet50 FC层
    model.group1 = nn.Sequential(
        OrderedDict([
            ('fc', nn.Linear(2048, 100))
        ])
    )
    # ignored_params = list(map(id, model.group2.parameters()))
    # base_params = filter(lambda p: id(p) not in ignored_params,
    #                      model.parameters())
    # print '==============================='
    # print model
    print '1'
    model = model.cuda()
    print '1'
    loss = torch.nn.CrossEntropyLoss(size_average=True)
    print '1'
    loss = loss.cuda()
    print '1'
    # 对局部优化
    # optimizer = optim.SGD(model.group2.parameters(), lr=(1e-03), momentum=0.9,weight_decay=0.001)
    # optimizer = optim.Adam([{'params':model.layer4[2].parameters()},
    #                         {'params':model.group2.parameters()}
    #                         ],lr=(1e-04),eps=1e-08, betas=(0.9, 0.999), weight_decay=0.0005)
    # optimizer_a = optim.Adam([{'params':model.group2.parameters()}
    #                         ],lr=(1e-04))

    # optimizer = optim.Adam(model.group1.parameters(),lr=(1e-04))

    # optimizer.lr = (1e-04)
    # print optimizer.lr
    # print optimizer.momentum
    # for param_group in optimizer.param_groups:
    #     print param_group['lr']
    # 全局优化
    optimizer = optim.SGD(model.parameters(), lr=(1e-03), momentum=0.9, weight_decay=0.0005)
    batch_size = data_l.batch_szie
    data_aug_num = data_l.data_add
    mini_batch_size = batch_size / data_aug_num
    epochs = 1000
    print '1'
    for e in range(epochs):
        cost = 0.0
        train_acc = 0.0
        if e == 12:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1


        num_batches_train = data_l.train_length / mini_batch_size
        print num_batches_train
        for k in range(num_batches_train+1):
            batch_train_data_X, batch_train_data_Y = data_l.get_train_data()
            batch_train_data_X = batch_train_data_X.transpose(0, 3, 1, 2)
            batch_train_data_X[:, 0, ...] -= MEAN_VALUE[0]
            batch_train_data_X[:, 1, ...] -= MEAN_VALUE[1]
            batch_train_data_X[:, 2, ...] -= MEAN_VALUE[2]
            # print batch_train_data_X.shape
            # print batch_train_data_Y.shape
            # batch_train_data_X = preprocess_input(batch_train_data_X)
            torch_batch_train_data_X = torch.from_numpy(batch_train_data_X).float()
            torch_batch_train_data_Y = torch.from_numpy(batch_train_data_Y).long()
            cost_temp, acc_temp = train(model, loss, optimizer, torch_batch_train_data_X, torch_batch_train_data_Y)
            train_acc += acc_temp
            cost += cost_temp
            if (k + 1) % 10 == 0:
                print 'now step train loss is : %f' % (cost_temp)
                print 'now step train acc is : %f' % (acc_temp)
            if (k + 1) % 20 == 0:
                print 'all average train loss is : %f' % (cost / (k + 1))
                print 'all average train acc is : %f' % (train_acc / (k + 1))
            if (k + 1) % 100 == 0:
                model.training = False
                acc = 0.0
                num_batches_test = data_l.test_length / batch_size
                for j in range(num_batches_test+1):
                    teX, teY = data_l.get_test_data()
                    teX = teX.transpose(0, 3, 1, 2)
                    # teX[:, 0, ...] -= MEAN_VALUE[0]
                    # teX[:, 1, ...] -= MEAN_VALUE[1]
                    # teX[:, 2, ...] -= MEAN_VALUE[2]
                    teX = preprocess_input(teX)
                    teX = torch.from_numpy(teX).float()
                    # teY = torch.from_numpy(teY).long()
                    predY = predict(model, teX)
                    # print predY.dtype
                    # print teY[start:end]
                    acc += 1. * np.mean(predY == teY)
                    # print ('Epoch %d ,Step %d, acc = %.2f%%'%(e,k,100.*np.mean(predY==teY[start:end])))
                model.training = True
                print 'Epoch %d ,Step %d, all test acc is : %f' % (e, k, acc / num_batches_test)
                torch.save(model, 'models/inception_model_pretrained_%s_%s_%s_1.pkl' % ('SGD', str(e), str(k)))
        model.training = False
        acc = 0.0
        num_batches_test = data_l.test_length / batch_size
        for j in range(num_batches_test):
            teX, teY = data_l.get_test_data()
            teX = teX.transpose(0, 3, 1, 2)
            teX[:, 0, ...] -= MEAN_VALUE[0]
            teX[:, 1, ...] -= MEAN_VALUE[1]
            teX[:, 2, ...] -= MEAN_VALUE[2]
            teX = torch.from_numpy(teX).float()
            # teY = torch.from_numpy(teY).long()
            predY = predict(model, teX)
            # print predY.dtype
            # print teY[start:end]
            acc += 1. * np.mean(predY == teY)
            # print ('Epoch %d ,Step %d, acc = %.2f%%'%(e,k,100.*np.mean(predY==teY[start:end])))
        model.training = True
        print 'Epoch %d ,Step %d, all test acc is : %f' % (e, k, acc / num_batches_test)
        torch.save(model, 'models/inception_model_pretrained_%s_%s_%s_1.pkl' % ('SGD', str(e), str(k)))
    print 'train over'


if __name__ == '__main__':
    main()
