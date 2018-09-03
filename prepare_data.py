#-*- coding:utf-8 -*-
import numpy as np
import os
import cv2
import numpy as np
import pickle
import sys


# def load_dataset(filename):
#     z = np.load(filename+'.npy')
#     timg,tgt = z['timg'],z['tgt']
#     print(timg.shape,tgt.shape)
#     return timg,tgt
#
# # ret = []
# def read_img(path):
#     files = os.listdir(path)
#     s = []
#     cnt = 0
#     # print("!!!"+path)
#     for file in files:
#         if not os.path.isdir(path + '/' + file):
#             if file == '.DS_Store': continue
#             img = cv2.imread(path + '/' + file)
#
#             s.append(img)
#         else:
#             s.e
#             s.append(read_img(path + '/' + file))
#             cnt += 1
#             if cnt > 10:
#                 break
#
#     return s
# #
# # path = '/Users/summer/Downloads/SRAD2018_TRAIN_001'
# path = '../input/SRAD2018_TRAIN_010'
# train = read_img(path)
# pickle.dump(train, open('../input/train_data.pkl','wb'))
#
# path = '../input/SRAD2018_Test_1'
# test = read_img(path)
# pickle.dump(test, open('../input/test_data.pkl','wb'))

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import numpy as np
import sys
import cv2
import os

# from forecaster import Forecaster
# from encoder import Encoder
# from BMSELoss import BMSELoss
# sys.path.append('/home/meteo/zihao.chen/model_service/utils')
# from data_transfrom import decode_radar_code, imgmap_tonumpy, encode_squs_code,imgmaps_tonumpy,imgmaps_tonumpy_crop

input_num_seqs = 1
output_num_seqs = 6
hidden_size = 3
input_channels_img = 3
output_channels_img = 3
size_image = 501
max_epoch = 100
cuda_flag = True
kernel_size = 3
batch_size = 60
lr = 0.0001
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
        for time in xrange(self.input_num_seqs):
            self.encoder(data[0])
        all_pre_data = []
        self.forecaster.set_h0(self.encoder)
        for time in xrange(self.output_num_seqs):
            pre_data = self.forecaster(None)
            # print h_next.size()

            all_pre_data.append(pre_data.data[0])

        return all_pre_data


def train_by_stype(model, loss, optimizer, x_val, y_val):
    fx = model.forward(x_val)
    all_loss = 0

    for pre_id in range(len(fx)):
        output = loss.forward(fx[pre_id], y_val[pre_id])
        all_loss += output.data[0]
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

    output /= 10.
    output.backward()
    optimizer.step()
    if cuda_flag == True:
        return output.cuda(), fx
    else:
        return output, fx


def verify(model, loss, x_val, y_val):
    fx = model.forward(x_val)
    output = 0
    for pre_id in range(len(fx)):
        output = loss.forward(fx[pre_id], y_val[pre_id])

    if cuda_flag == True:
        return output.cuda().data[0]
    else:
        return output.data[0]


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_t = lr
    lr_t = lr_t * (0.3 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_t



def mtest(input_channels_img, output_channels_img, size_image, max_epoch, model, cuda_test):
    criterion = nn.MSELoss()
    if cuda_test == True:
        criterion = criterion.cuda()
    # optimizer = optim.SGD(model.parameters(), lr=(lr), momentum=0.9, weight_decay=0.005)
    optimizer = optim.Adam(model.parameters(), lr=(lr), weight_decay=0.005)
    print model
    print optimizer
    print criterion

    i=0
    def read_img(path):
        files = os.listdir(path)
        s = []
        cnt = 0
        # print("!!!"+path)
        for file in files:
            if not os.path.isdir(path + '/' + file):
                if file == '.DS_Store': continue
                img = cv2.imread(path + '/' + file)

                s.append(img)
                print 'epoch :', i
                print train_arr.shape
                # nnn = range(train_arr.shape[0])
                # np.random.shuffle(nnn)
                # train_arr_b = train_arr[nnn]
                # batch_num = train_arr_b.shape[0] // batch_size
                # print batch_num
                model.train()
                all_error = 0.
                # for j in range(batch_num):
                    # batch_img = imgmaps_tonumpy_crop(train_arr_b[j * batch_size:(j + 1) * batch_size, ...], train_imgs_maps)
                batch_img = img
                input_image = batch_img[:31] / 255.
                target_image = batch_img[31:61:5] / 255.

                if cuda_test == True:
                    input_gru = Variable(input_image.cuda())
                    target_gru = Variable(target_image.cuda())
                else:
                    input_gru = Variable(input_image)
                    target_gru = Variable(target_image)

                error, pre_list = train(model, criterion, optimizer, input_gru, target_gru)
                # all_error += error
                print j, ' : ', error
                # print 'epoch train %d %f' % (i, all_error / batch_num)
                # batch_num = test_arr.shape[0] // batch_size

            else:
                s.e
                s.append(read_img(path + '/' + file))
                cnt += 1
                if cnt > 10:
                    break

        return s

    for i in range(max_epoch):
        # adjust_learning_rate(optimizer, i)
        print 'epoch :', i
        print train_arr.shape
        nnn = range(train_arr.shape[0])
        np.random.shuffle(nnn)
        train_arr_b = train_arr[nnn]
        batch_num = train_arr_b.shape[0] // batch_size
        print batch_num
        model.train()
        all_error = 0.
        for j in range(batch_num):
            # batch_img = imgmaps_tonumpy_crop(train_arr_b[j * batch_size:(j + 1) * batch_size, ...], train_imgs_maps)
            batch_img = train_arr[j*batch_size:(j+1)*batch_size]
            input_image = batch_img[:31] / 255.
            target_image = batch_img[31:61:5] / 255.

            if cuda_test == True:
                input_gru = Variable(input_image.cuda())
                target_gru = Variable(target_image.cuda())
            else:
                input_gru = Variable(input_image)
                target_gru = Variable(target_image)

            error, pre_list = train(model, criterion, optimizer, input_gru, target_gru)
            # all_error += error
            print j, ' : ', error
        # print 'epoch train %d %f' % (i, all_error / batch_num)
        batch_num = test_arr.shape[0] // batch_size
        with torch.no_grad():
            # model.eval()
            all_error = 0.
            for j in range(batch_num):
                # batch_img = imgmaps_tonumpy(test_arr[j * batch_size:(j + 1) * batch_size, ...], test_imgs_maps)
                batch_img = test_arr[j*batch_size:(j+1)*batch_size]
                input_image = batch_img[:31] / 255.
                target_image = test_arr[31:61:5] / 255.
                # input_image = torch.from_numpy(input_image).float()
                # target_image = torch.from_numpy(target_image).float()

                if cuda_test == True:
                    input_gru = Variable(input_image.cuda())
                    target_gru = Variable(target_image.cuda())
                else:
                    input_gru = Variable(input_image)
                    target_gru = Variable(target_image)
                #             input_gru = input_image.cuda()
                #             target_gru = target_image.cuda()

                error = verify(model, criterion, input_gru, target_gru)
                # all_error += error
                print j, ' : ', error
            # print 'epoch test %d %f' % (i, all_error / batch_num)


def tmp_data():
    train_arr = torch.rand(input_num_seqs, batch_size, input_channels_img, size_image, size_image)
    test_arr = torch.rand(input_num_seqs, batch_size, input_channels_img, size_image, size_image)
    # train_imgs_maps = torch.rand(input_num_seqs, batch_size, output_channels_img, size_image, size_image)
    # test_imgs_maps = torch.rand(input_num_seqs, batch_size, output_channels_img, size_image, size_image)

    return train_arr, test_arr


def load_train_data():
    path = '../input/train_data.pkl' # to be confirmed
    train_arr = pickle.load(open(path,'rb'))
    return train_arr

total_data = load_train_data()
train_size = int(len(total_data) * 0.7)
train_arr, test_arr = total_data[:train_size], total_data[train_size:]


if __name__ == '__main__':
    m = HKOModel(inplanes=input_channels_img, input_num_seqs=input_num_seqs, output_num_seqs=output_num_seqs)
    if cuda_flag == True:
        m = m.cuda()

    mtest(input_channels_img, output_channels_img, size_image, max_epoch, model=m, cuda_test=cuda_flag)
