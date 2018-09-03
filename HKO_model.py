#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-04-02
Modify Date: 2018-04-02
descirption: ""
'''
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import numpy as np
import sys
import cv2
import os
from forecaster import Forecaster
from encoder import Encoder
from BMSELoss import BMSELoss
# sys.path.append('/home/meteo/zihao.chen/model_service/utils')
# from data_transfrom import decode_radar_code, imgmap_tonumpy, encode_squs_code,imgmaps_tonumpy,imgmaps_tonumpy_crop

input_num_seqs = 1
output_num_seqs = 6
hidden_size = 3
input_channels_img = 3
output_channels_img = 3
size_image = 501
max_epoch = 4
cuda_flag = False
kernel_size = 3
batch_size = 1
lr= 0.0001
momentum = 0.5

import pickle

class HKOModel(nn.Module):
    def __init__(self, inplanes, input_num_seqs, output_num_seqs):
        super(HKOModel, self).__init__()
        self.input_num_seqs = input_num_seqs
        self.output_num_seqs = output_num_seqs
        self.encoder = Encoder(inplanes=inplanes, num_seqs=input_num_seqs)
        self.forecaster = Forecaster(num_seqs=output_num_seqs)
        if cuda_flag == True:
            self.encoder = self.encoder.cuda()
            self.forecaster = self.forecaster.cuda()

    def forward(self, data):
        self.encoder.init_h0()
        for time in range(self.input_num_seqs):
            self.encoder(data[time])
        all_pre_data = []
        self.forecaster.set_h0(self.encoder)
        for time in range(self.output_num_seqs):

            pre_data = self.forecaster(None)
            # print h_next.size()

            all_pre_data.append(pre_data)

        return all_pre_data


def train_by_stype(model, loss, optimizer, x_val, y_val):
    fx = model.forward(x_val)
    all_loss = 0

    for pre_id in range(len(fx)):
        output = loss.forward(fx[pre_id], y_val[pre_id])
        all_loss += output
        optimizer.zero_grad()
        output.backward(retain_graph=True)
        optimizer.step()
        # if pre_id == 1:
        #     print 'loss 1:',output
    if cuda_flag == True:
        return all_loss.cuda().data[0], fx
    else:
        return all_loss.data[0], fx


def train(model, loss, optimizer, x_val, y_val):

    optimizer.zero_grad()
    fx = model.forward(x_val)
    output = 0

    for pre_id in range(len(fx)):
        tmp = loss.forward(fx[pre_id], y_val[pre_id])
        output += tmp.data[0]

    output/=10.
    output.backward()
    optimizer.step()
    if cuda_flag == True:
        return output.cuda().data[0], fx
    else:
        return output.data[0], fx


def verify(model, loss, x_val, y_val):
    fx = model.forward(x_val)
    output = 0
    for pre_id in range(len(fx)):
        tmp= loss.forward(fx[pre_id], y_val[pre_id])
        output +=tmp.data[0]

    if cuda_flag == True:
        return output.cuda().data[0]
    else:
        return output.data[0]


def load_data(code_list):
    test_arr = None
    train_arr = None
    train_imgs_maps = {}
    test_imgs_maps = {}
    for code in code_list:
        file_f = open('%s.pkl'%code,'rb')
        map_l = pickle.load(file_f)
        file_f.close()
        if test_arr is None:
            # test_arr = map_l['test_arr']
            # train_arr = map_l['train_arr']
            test_arr = map_l[:3]
            train_arr = map_l[3:6]
        else:
            test_arr = np.concatenate((test_arr,map_l['test_arr']),axis=0)
            train_arr = np.concatenate((train_arr, map_l['train_arr']), axis=0)
        # train_imgs_maps[code] = map_l['train_imgs_map']
        # test_imgs_maps[code] = map_l['test_imgs_map']
        train_imgs_maps[code] = map_l[6:9]
        test_imgs_maps[code] = map_l[9:12]

    return train_arr,test_arr,train_imgs_maps,test_imgs_maps

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_t = lr
    lr_t = lr_t * (0.3 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_t


def touch_dir(path):
    result = False
    try:
        path = path.strip().rstrip("\\")
        if not os.path.exists(path):
            os.makedirs(path)
            result = True
        else:
            result = True
    except:
        result = False
    return result


def mtest(input_channels_img, output_channels_img, size_image, max_epoch, model, cuda_test):

    criterion = nn.MSELoss()
    if cuda_test == True:
        criterion = criterion.cuda()
    # optimizer = optim.SGD(model.parameters(), lr=(lr), momentum=0.9, weight_decay=0.005)
    optimizer = optim.Adam(model.parameters(), lr=(lr), weight_decay=0.005)
    # print model
    # print optimizer
    # print criterion
    for i in range(max_epoch):
        # adjust_learning_rate(optimizer, i)
        # print 'epoch :', i
        # print train_arr.shape
        nnn = range(train_arr.shape[0])
        np.random.shuffle(nnn)
        train_arr_b = train_arr[nnn]
        batch_num = train_arr_b.shape[0] // batch_size
        # print batch_num
        model.train()
        all_error = 0.
        for j in range(batch_num):
            # batch_img = imgmaps_tonumpy_crop(train_arr_b[j * batch_size:(j + 1) * batch_size, ...], train_imgs_maps)
            batch_img = train_arr
            input_image = batch_img[:batch_size] / 255.
            target_image = batch_img[:batch_size] / 255.
            # input_image = torch.from_numpy(input_image).float()
            # target_image = torch.from_numpy(target_image).float()

            if cuda_test == True:
                input_gru = Variable(input_image.cuda())
                target_gru = Variable(target_image.cuda())
            else:
                input_gru = Variable(input_image)
                target_gru = Variable(target_image)

            print "!!!!!!!!!!!!",input_gru.shape, target_gru.shape
            error, pre_list = train(model, criterion, optimizer, input_gru, target_gru)
            # all_error += error
            print(j, ' : ', error)
        # print 'epoch train %d %f' % (i, all_error / batch_num)
        batch_num = test_arr.shape[0] // batch_size
        model.eval()
        all_error = 0.
        for j in range(batch_num):
            # batch_img = imgmaps_tonumpy(test_arr[j * batch_size:(j + 1) * batch_size, ...], test_imgs_maps)
            batch_img = test_arr
            input_image = batch_img[:batch_size] / 255.
            target_image = batch_img[:batch_size] / 255.
            # input_image = torch.from_numpy(input_image).float()
            # target_image = torch.from_numpy(target_image).float()

            if cuda_test == True:
                input_gru = Variable(input_image.cuda())
                target_gru = Variable(target_image.cuda())
            else:
                input_gru = Variable(input_image)
                target_gru = Variable(target_image)


            error = verify(model, criterion, input_gru, target_gru)
            # all_error += error
            # print j, ' : ', error
        # print 'epoch test %d %f' % (i, all_error / batch_num)
    model.eval()
    for i in range(test_arr.shape[0]):
        # temp_path = test_arr[i, 0, 0]
        # start_i = temp_path.find('201')
        # time_str = temp_path[start_i:start_i + 12]
        # print time_str
        # start_i = temp_path.find('AZ')
        # radar_code = temp_path[start_i:start_i + 6]
        save_path = 'imgs'
        # save_path = '/imgs/%s/%s/' % (radar_code,time_str)
        touch_dir(save_path)
        temp_arr = test_arr[i]
        temp_arr = temp_arr[np.newaxis, ...]
        # batch_img = imgmaps_tonumpy(temp_arr, test_imgs_maps)
        # batch_img = test_arr
        # input_image = batch_img[0, ...]
        # target_image = batch_img[0,:,0, ...]
        # input_image_t = torch.from_numpy(input_image / 255.).float()

        batch_img = train_arr
        input_image = batch_img[:batch_size] / 255.

        input_image_t = input_image / 255.

        if cuda_test == True:
            input_gru = Variable(input_image_t.cuda())
        else:
            input_gru = Variable(input_image_t)

        fx = model.forward(input_gru)
        for pre_id in range(len(fx)):
            temp_xx = fx[pre_id].cpu().data.numpy()
            tmp_img = temp_xx[0, 0, ...]
            tmp_img = tmp_img * 255.
            true_img = target_image[pre_id, 0,  ...]
            encode_img = input_image[pre_id, 0, 0, ...]
            # cv2.imwrite(os.path.join(save_path, 'a_%s.png' % pre_id), encode_img)
            cv2.imwrite(os.path.join(save_path, 'c_%s.png' % pre_id), tmp_img)
            # cv2.imwrite(os.path.join(save_path, 'b_%s.png' % pre_id), true_img)

    # for pre_data in pre_list:
    #     temp = pre_data.cpu().data.numpy()
    #     print temp.mean()
def tmp_data():
    train_arr = torch.rand(input_num_seqs, batch_size, input_channels_img, size_image, size_image)
    test_arr = torch.rand(input_num_seqs, batch_size, input_channels_img, size_image, size_image)
    # train_imgs_maps = torch.rand(input_num_seqs, batch_size, output_channels_img, size_image, size_image)
    # test_imgs_maps = torch.rand(input_num_seqs, batch_size, output_channels_img, size_image, size_image)
    return train_arr, test_arr

# train_arr, test_arr, train_imgs_maps, test_imgs_maps = load_data(['AZ9010','AZ9200'])
train_arr, test_arr = tmp_data()

if __name__ == '__main__':
    m = HKOModel(inplanes=input_channels_img, input_num_seqs=input_num_seqs, output_num_seqs=output_num_seqs)
    if cuda_flag == True:
        m = m.cuda()

    mtest(input_channels_img, output_channels_img, size_image, max_epoch, model=m, cuda_test=cuda_flag)
