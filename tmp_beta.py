#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-03-29
Modify Date: 2018-03-29
descirption: ""
'''
import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


# cuda_flag = True


class ConvGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(p=0.5)
        self.ConvGates = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, self.kernel_size,
                                   padding=self.kernel_size // 2)
        self.Conv_ct = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, self.kernel_size,
                                 padding=self.kernel_size // 2)
        dtype = torch.FloatTensor

    def forward(self, input, hidden):
        if hidden is None:
            # print (input.data.size()[0])
            # print (self.hidden_size)
            # print (list(input.data.size()[2:]))
            size_h = [input.data.size()[0], self.hidden_size] + list(input.data.size()[2:])
            # print size_h

            hidden = Variable(torch.zeros(size_h).cuda())
        if input is None:
            # print (input.data.size()[0])
            # print (self.hidden_size)
            # print (list(input.data.size()[2:]))
            size_h = [hidden.data.size()[0], self.input_size] + list(hidden.data.size()[2:])
            # print size_h
            #             if cuda_flag == True:
            input = Variable(torch.zeros(size_h).cuda())
        #             else:
        #                 input = Variable(torch.zeros(size_h))
        # print input.size()
        # print hidden.size()
        #         hidden = hidden.cuda()
        c1 = self.ConvGates(torch.cat((input, hidden), 1))
        (rt, ut) = c1.chunk(2, 1)
        reset_gate = self.dropout(f.sigmoid(rt))
        update_gate = self.dropout(f.sigmoid(ut))
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        ct = f.tanh(p1)
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return next_h


# !/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-03-29
Modify Date: 2018-03-29
descirption: "conv GRU encoder stack"
'''

import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


# from ConvGRUCell import ConvGRUCell


def conv2_act(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    layers = []
    layers += [nn.Conv2d(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias)]
    # layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.LeakyReLU(negative_slope=0.2)]
    return nn.Sequential(*layers)


def downsmaple(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    # torch.cat((x, x, x), dim = 0)
    # the downsample layer input is last rnn output,like:output[-1]
    ret = conv2_act(inplanes, out_channels, kernel_size, stride, padding, bias)
    return ret


class Encoder(nn.Module):
    def __init__(self, inplanes, num_seqs):
        super(Encoder, self).__init__()
        self.num_seqs = num_seqs
        out_channels = 16
        self.conv1_act = conv2_act(inplanes, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv2_act = conv2_act(8, out_channels=16, kernel_size=4, stride=2, padding=1, bias=True)
        # num_filter = [64, 192, 192]
        # kernel_size_l = [7,7,5]
        num_filter = [8, 16, 16]
        kernel_size_l = [7, 7, 5]
        rnn_block_num = len(num_filter)
        stack_num = [2, 3, 3]
        encoder_rnn_block_states = []
        self.rnn1_1 = ConvGRUCell(input_size=out_channels, hidden_size=num_filter[0],
                                  kernel_size=kernel_size_l[0]).cuda()
        self.rnn1_1_h = None
        self.rnn1_2 = ConvGRUCell(input_size=num_filter[0], hidden_size=num_filter[0],
                                  kernel_size=kernel_size_l[0]).cuda()
        self.rnn1_2_h = None
        self.downsample1 = downsmaple(inplanes=num_filter[0], out_channels=num_filter[1], kernel_size=4, stride=2,
                                      padding=0)

        self.rnn2_1 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1],
                                  kernel_size=kernel_size_l[1]).cuda()
        self.rnn2_1_h = None
        self.rnn2_2 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1],
                                  kernel_size=kernel_size_l[1]).cuda()
        self.rnn2_2_h = None
        self.rnn2_3 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_3_h = None

        self.downsample2 = downsmaple(inplanes=num_filter[1], out_channels=num_filter[2], kernel_size=5, stride=3,
                                      padding=1)

        self.rnn3_1 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn3_1_h = None
        self.rnn3_2 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn3_2_h = None
        self.rnn3_3 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn3_3_h = None

    def init_h0(self):
        self.rnn1_1_h = None
        self.rnn1_2_h = None
        self.rnn2_1_h = None
        self.rnn2_2_h = None
        self.rnn2_3_h = None
        self.rnn3_1_h = None
        self.rnn3_2_h = None
        self.rnn3_3_h = None

    def forward(self, data):
        # print data.size()
        data = self.conv1_act(data)
        #         print data.size()

        # data = self.conv2_act(data)
        # print data.size()
        self.rnn1_1_h = self.rnn1_1(data, self.rnn1_1_h)

        #         print self.rnn1_1_h.size()
        self.rnn1_2_h = self.rnn1_2(self.rnn1_1_h, self.rnn1_2_h)
        #         print self.rnn1_2_h.size()
        # data = torch.cat(self.rnn1_2_h, dim=0)
        # print data.size()
        data = self.downsample1(self.rnn1_2_h)
        #         print data.size()
        self.rnn2_1_h = self.rnn2_1(data, self.rnn2_1_h)
        #         print self.rnn2_1_h.size()
        self.rnn2_2_h = self.rnn2_2(self.rnn2_1_h, self.rnn2_2_h)
        #         print self.rnn2_2_h.size()
        self.rnn2_3_h = self.rnn2_3(self.rnn2_2_h, self.rnn2_3_h)
        #         print self.rnn2_3_h.size()
        # data = torch.cat(*self.rnn2_3_h, dim=0)
        data = self.downsample2(self.rnn2_3_h)
        #         print data.size()
        self.rnn3_1_h = self.rnn3_1(data, self.rnn3_1_h)
        #         print self.rnn3_1_h.size()
        self.rnn3_2_h = self.rnn3_2(self.rnn3_1_h, self.rnn3_2_h)
        #         print self.rnn3_2_h.size()
        self.rnn3_3_h = self.rnn3_3(self.rnn3_2_h, self.rnn3_3_h)
        #         print self.rnn3_3_h.size()
        return self.rnn2_3_h


import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


# from ConvGRUCell import ConvGRUCell

def deconv2_act(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    layers = []
    layers += [
        nn.ConvTranspose2d(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=bias)]
    layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.LeakyReLU(negative_slope=0.2)]
    return nn.Sequential(*layers)


def conv2_act(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    layers = []
    layers += [nn.Conv2d(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias)]
    layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.LeakyReLU(negative_slope=0.2)]
    return nn.Sequential(*layers)


def upsmaple(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    # torch.cat((x, x, x), dim = 0)
    # the downsample layer input is last rnn output,like:output[-1]
    ret = deconv2_act(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias)
    return ret


class Forecaster(nn.Module):
    def __init__(self, num_seqs):
        super(Forecaster, self).__init__()

        # num_filter = [64, 192, 192]
        num_filter = [8, 16, 16]
        kernel_size_l = [7, 7, 5]
        self.rnn1_1 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn1_1_h = None
        self.rnn1_2 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn1_2_h = None
        self.rnn1_3 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn1_3_h = None
        #
        self.upsample1 = upsmaple(inplanes=num_filter[2], out_channels=num_filter[2], kernel_size=5, stride=3,
                                  padding=1)

        self.rnn2_1 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_1_h = None
        self.rnn2_2 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_2_h = None
        self.rnn2_3 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_3_h = None

        self.upsample2 = upsmaple(inplanes=num_filter[1], out_channels=num_filter[1], kernel_size=7, stride=2,
                                  padding=1)

        self.rnn3_1 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[0], kernel_size=kernel_size_l[0])
        self.rnn3_1_h = None
        self.rnn3_2 = ConvGRUCell(input_size=num_filter[0], hidden_size=num_filter[0], kernel_size=kernel_size_l[0])
        self.rnn3_2_h = None

        self.deconv1 = deconv2_act(inplanes=num_filter[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.deconv2 = deconv2_act(inplanes=8, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.conv_final = conv2_act(inplanes=16, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.conv_pre = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1)

    def set_h0(self, encoder):
        self.rnn1_1_h = encoder.rnn3_3_h
        self.rnn1_2_h = encoder.rnn3_2_h
        self.rnn1_3_h = encoder.rnn3_1_h
        self.rnn2_1_h = encoder.rnn2_3_h
        self.rnn2_2_h = encoder.rnn2_2_h
        self.rnn2_3_h = encoder.rnn2_1_h
        self.rnn3_1_h = encoder.rnn1_2_h
        self.rnn3_2_h = encoder.rnn1_1_h

    def forward(self, data):
        # print data.size()
        self.rnn1_1_h = self.rnn1_1(data, self.rnn1_1_h)
        self.rnn1_2_h = self.rnn1_1(self.rnn1_1_h, self.rnn1_2_h)
        self.rnn1_3_h = self.rnn1_1(self.rnn1_2_h, self.rnn1_3_h)
        # # print self.rnn1_3_h.size()
        #         print self.rnn1_3_h.size()
        data = self.upsample1(self.rnn1_3_h)  # dimension is 6
        #         print data.size()

        self.rnn2_1_h = self.rnn2_1(data, self.rnn2_1_h)  # here is a bug, 6 and 8 are not the same dimension

        self.rnn2_2_h = self.rnn2_2(self.rnn2_1_h, self.rnn2_2_h)

        self.rnn2_3_h = self.rnn2_3(self.rnn2_2_h, self.rnn2_3_h)
        #         print self.rnn2_3_h.size()
        data = self.upsample2(self.rnn2_3_h)
        #         print data.size()
        self.rnn3_1_h = self.rnn3_1(data, self.rnn3_1_h)

        self.rnn3_2_h = self.rnn3_2(self.rnn3_1_h, self.rnn3_2_h)
        #         print self.rnn3_2_h.size()
        data = self.deconv1(self.rnn3_2_h)
        #         print data.size()
        # data = self.deconv2(data)
        # print data.size()
        data = self.conv_final(data)
        #         print data.size()
        pre_data = self.conv_pre(data)
        #         print pre_data.size()
        return pre_data


import torch


class BMSELoss(torch.nn.Module):

    def __init__(self):
        super(BMSELoss, self).__init__()
        self.w_l = [1, 2, 5, 10, 30]
        self.y_l = [0.283, 0.353, 0.424, 0.565, 1]

    def forward(self, x, y):
        w = y.clone()
        for i in range(len(self.w_l)):
            w[w < self.y_l[i]] = self.w_l[i]
        return torch.mean(w * ((y - x) ** 2))


# !/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-03-29
Modify Date: 2018-03-29
descirption: ""
'''
import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


# cuda_flag = True


class ConvGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(p=0.5)
        self.ConvGates = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, self.kernel_size,
                                   padding=self.kernel_size // 2)
        self.Conv_ct = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, self.kernel_size,
                                 padding=self.kernel_size // 2)
        dtype = torch.FloatTensor

    def forward(self, input, hidden):
        if hidden is None:
            # print (input.data.size()[0])
            # print (self.hidden_size)
            # print (list(input.data.size()[2:]))
            size_h = [input.data.size()[0], self.hidden_size] + list(input.data.size()[2:])
            # print size_h

            hidden = Variable(torch.zeros(size_h))
        if input is None:
            # print (input.data.size()[0])
            # print (self.hidden_size)
            # print (list(input.data.size()[2:]))
            size_h = [hidden.data.size()[0], self.input_size] + list(hidden.data.size()[2:])
            # print size_h
            if cuda_flag == True:
                input = Variable(torch.zeros(size_h).cuda())
            else:
                input = Variable(torch.zeros(size_h))
        # print input.size()
        # print hidden.size()
        hidden = hidden.cuda()
        c1 = self.ConvGates(torch.cat((input, hidden), 1))
        (rt, ut) = c1.chunk(2, 1)
        reset_gate = self.dropout(f.sigmoid(rt))
        update_gate = self.dropout(f.sigmoid(ut))
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        ct = f.tanh(p1)
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return next_h


# !/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-03-29
Modify Date: 2018-03-29
descirption: "conv GRU encoder stack"
'''

import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


# from ConvGRUCell import ConvGRUCell


def conv2_act(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    layers = []
    layers += [nn.Conv2d(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias)]
    # layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.LeakyReLU(negative_slope=0.2)]
    return nn.Sequential(*layers)


def downsmaple(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    # torch.cat((x, x, x), dim = 0)
    # the downsample layer input is last rnn output,like:output[-1]
    ret = conv2_act(inplanes, out_channels, kernel_size, stride, padding, bias)
    return ret


class Encoder(nn.Module):
    def __init__(self, inplanes, num_seqs):
        super(Encoder, self).__init__()
        self.num_seqs = num_seqs
        out_channels = 16
        self.conv1_act = conv2_act(inplanes, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv2_act = conv2_act(8, out_channels=16, kernel_size=4, stride=2, padding=1, bias=True)
        # num_filter = [64, 192, 192]
        # kernel_size_l = [7,7,5]
        num_filter = [8, 16, 16]
        kernel_size_l = [7, 7, 5]
        rnn_block_num = len(num_filter)
        stack_num = [2, 3, 3]
        encoder_rnn_block_states = []
        self.rnn1_1 = ConvGRUCell(input_size=out_channels, hidden_size=num_filter[0],
                                  kernel_size=kernel_size_l[0])
        self.rnn1_1_h = None
        self.rnn1_2 = ConvGRUCell(input_size=num_filter[0], hidden_size=num_filter[0],
                                  kernel_size=kernel_size_l[0])
        self.rnn1_2_h = None
        self.downsample1 = downsmaple(inplanes=num_filter[0], out_channels=num_filter[1], kernel_size=4, stride=2,
                                      padding=0)

        self.rnn2_1 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1],
                                  kernel_size=kernel_size_l[1])
        self.rnn2_1_h = None
        self.rnn2_2 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1],
                                  kernel_size=kernel_size_l[1])
        self.rnn2_2_h = None
        self.rnn2_3 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_3_h = None

        self.downsample2 = downsmaple(inplanes=num_filter[1], out_channels=num_filter[2], kernel_size=5, stride=3,
                                      padding=1)

        self.rnn3_1 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn3_1_h = None
        self.rnn3_2 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn3_2_h = None
        self.rnn3_3 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn3_3_h = None

    def init_h0(self):
        self.rnn1_1_h = None
        self.rnn1_2_h = None
        self.rnn2_1_h = None
        self.rnn2_2_h = None
        self.rnn2_3_h = None
        self.rnn3_1_h = None
        self.rnn3_2_h = None
        self.rnn3_3_h = None

    def forward(self, data):
        # print data.size()
        data = self.conv1_act(data)
        #         print data.size()

        # data = self.conv2_act(data)
        # print data.size()
        self.rnn1_1_h = self.rnn1_1(data, self.rnn1_1_h)

        #         print self.rnn1_1_h.size()
        self.rnn1_2_h = self.rnn1_2(self.rnn1_1_h, self.rnn1_2_h)
        #         print self.rnn1_2_h.size()
        # data = torch.cat(self.rnn1_2_h, dim=0)
        # print data.size()
        data = self.downsample1(self.rnn1_2_h)
        #         print data.size()
        self.rnn2_1_h = self.rnn2_1(data, self.rnn2_1_h)
        #         print self.rnn2_1_h.size()
        self.rnn2_2_h = self.rnn2_2(self.rnn2_1_h, self.rnn2_2_h)
        #         print self.rnn2_2_h.size()
        self.rnn2_3_h = self.rnn2_3(self.rnn2_2_h, self.rnn2_3_h)
        #         print self.rnn2_3_h.size()
        # data = torch.cat(*self.rnn2_3_h, dim=0)
        data = self.downsample2(self.rnn2_3_h)
        #         print data.size()
        self.rnn3_1_h = self.rnn3_1(data, self.rnn3_1_h)
        #         print self.rnn3_1_h.size()
        self.rnn3_2_h = self.rnn3_2(self.rnn3_1_h, self.rnn3_2_h)
        #         print self.rnn3_2_h.size()
        self.rnn3_3_h = self.rnn3_3(self.rnn3_2_h, self.rnn3_3_h)
        #         print self.rnn3_3_h.size()
        return self.rnn2_3_h


import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


# from ConvGRUCell import ConvGRUCell

def deconv2_act(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    layers = []
    layers += [
        nn.ConvTranspose2d(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=bias)]
    layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.LeakyReLU(negative_slope=0.2)]
    return nn.Sequential(*layers)


def conv2_act(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    layers = []
    layers += [nn.Conv2d(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias)]
    layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.LeakyReLU(negative_slope=0.2)]
    return nn.Sequential(*layers)


def upsmaple(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    # torch.cat((x, x, x), dim = 0)
    # the downsample layer input is last rnn output,like:output[-1]
    ret = deconv2_act(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias)
    return ret


class Forecaster(nn.Module):
    def __init__(self, num_seqs):
        super(Forecaster, self).__init__()

        # num_filter = [64, 192, 192]
        num_filter = [8, 16, 16]
        kernel_size_l = [7, 7, 5]
        self.rnn1_1 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn1_1_h = None
        self.rnn1_2 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn1_2_h = None
        self.rnn1_3 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn1_3_h = None
        #
        self.upsample1 = upsmaple(inplanes=num_filter[2], out_channels=num_filter[2], kernel_size=5, stride=3,
                                  padding=1)

        self.rnn2_1 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_1_h = None
        self.rnn2_2 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_2_h = None
        self.rnn2_3 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_3_h = None

        self.upsample2 = upsmaple(inplanes=num_filter[1], out_channels=num_filter[1], kernel_size=7, stride=2,
                                  padding=1)

        self.rnn3_1 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[0], kernel_size=kernel_size_l[0])
        self.rnn3_1_h = None
        self.rnn3_2 = ConvGRUCell(input_size=num_filter[0], hidden_size=num_filter[0], kernel_size=kernel_size_l[0])
        self.rnn3_2_h = None

        self.deconv1 = deconv2_act(inplanes=num_filter[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.deconv2 = deconv2_act(inplanes=8, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.conv_final = conv2_act(inplanes=16, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.conv_pre = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1)

    def set_h0(self, encoder):
        self.rnn1_1_h = encoder.rnn3_3_h
        self.rnn1_2_h = encoder.rnn3_2_h
        self.rnn1_3_h = encoder.rnn3_1_h
        self.rnn2_1_h = encoder.rnn2_3_h
        self.rnn2_2_h = encoder.rnn2_2_h
        self.rnn2_3_h = encoder.rnn2_1_h
        self.rnn3_1_h = encoder.rnn1_2_h
        self.rnn3_2_h = encoder.rnn1_1_h

    def forward(self, data):
        # print data.size()
        self.rnn1_1_h = self.rnn1_1(data, self.rnn1_1_h)
        self.rnn1_2_h = self.rnn1_1(self.rnn1_1_h, self.rnn1_2_h)
        self.rnn1_3_h = self.rnn1_1(self.rnn1_2_h, self.rnn1_3_h)
        # # print self.rnn1_3_h.size()
        #         print self.rnn1_3_h.size()
        data = self.upsample1(self.rnn1_3_h)  # dimension is 6
        #         print data.size()

        self.rnn2_1_h = self.rnn2_1(data, self.rnn2_1_h)  # here is a bug, 6 and 8 are not the same dimension

        self.rnn2_2_h = self.rnn2_2(self.rnn2_1_h, self.rnn2_2_h)

        self.rnn2_3_h = self.rnn2_3(self.rnn2_2_h, self.rnn2_3_h)
        #         print self.rnn2_3_h.size()
        data = self.upsample2(self.rnn2_3_h)
        #         print data.size()
        self.rnn3_1_h = self.rnn3_1(data, self.rnn3_1_h)

        self.rnn3_2_h = self.rnn3_2(self.rnn3_1_h, self.rnn3_2_h)
        #         print self.rnn3_2_h.size()
        data = self.deconv1(self.rnn3_2_h)
        #         print data.size()
        # data = self.deconv2(data)
        # print data.size()
        data = self.conv_final(data)
        #         print data.size()
        pre_data = self.conv_pre(data)
        #         print pre_data.size()
        return pre_data


import torch


class BMSELoss(torch.nn.Module):

    def __init__(self):
        super(BMSELoss, self).__init__()
        self.w_l = [1, 2, 5, 10, 30]
        self.y_l = [0.283, 0.353, 0.424, 0.565, 1]

    def forward(self, x, y):
        w = y.clone()
        for i in range(len(self.w_l)):
            w[w < self.y_l[i]] = self.w_l[i]
        return torch.mean(w * ((y - x) ** 2))


# -*- coding:utf-8 -*-
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
            self.encoder(data[time])
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
    print len(fx)
    for pre_id in range(len(fx)):
        tmp = loss.forward(fx[pre_id], y_val[pre_id])
        output += tmp

    output /= 10.
    print output
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
        output += loss.forward(fx[pre_id], y_val[pre_id]).data[0]

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
    #     print model
    #     print optimizer
    #     print criterion

    i = 0
    #     def read_img(path):
    path = '../input/SRAD2018_TRAIN_010'
    files = os.listdir(path)
    s = []
    cnt = 0
    tot_size = 10
    train_size = 5
    for file in files:
        cnt += 1
        img = []
        t_path = path + '/' + file
        t_files = os.listdir(t_path)
        for f in t_files:
            if not os.path.isdir(t_path + '/' + f):
                if f == '.DS_Store': continue
                tmp = cv2.imread(t_path + '/' + f)

                img.append(tmp)

        #             print img
        img = torch.FloatTensor(img)
        print 'epoch :', cnt
        print img.shape

        if cnt < train_size:
            model.train()
            all_error = 0.

            batch_img = img
            input_image = batch_img[:31] / 255.
            target_image = batch_img[31::5] / 255.

            input_gru = Variable(input_image.cuda(), requires_grad=True)
            target_gru = Variable(target_image.cuda(), requires_grad=True)
            print target_gru.shape
            input_gru = input_gru.reshape(31, input_num_seqs, input_channels_img, size_image, size_image)
            target_gru = target_gru.reshape(6, input_channels_img, size_image, size_image)

            error, pre_list = train(model, criterion, optimizer, input_gru, target_gru)
            # all_error += error
            print  'error : ', error
            # print 'epoch train %d %f' % (i, all_error / batch_num)
            # batch_num = test_arr.shape[0] // batch_size
        elif cnt < tot_size:
            with torch.no_grad():
                # model.eval()
                all_error = 0.

                # batch_img = imgmaps_tonumpy(test_arr[j * batch_size:(j + 1) * batch_size, ...], test_imgs_maps)
                batch_img = img
                input_image = batch_img[:31] / 255.
                target_image = batch_img[31::5] / 255.
                # input_image = torch.from_numpy(input_image).float()
                # target_image = torch.from_numpy(target_image).float()

                if cuda_test == True:
                    input_gru = Variable(input_image.cuda(), requires_grad=True)
                    target_gru = Variable(target_image.cuda(), requires_grad=True)
                else:
                    input_gru = Variable(input_image)
                    target_gru = Variable(target_image)
                input_gru = input_gru.reshape(31, input_num_seqs, input_channels_img, size_image, size_image)
                target_gru = target_gru.reshape(6, input_channels_img, size_image, size_image)
                error = verify(model, criterion, input_gru, target_gru)
                # all_error += error
                print  'test_error : ', error
                # print 'epoch test %d %f' % (i, all_error / batch_num)
        if cnt > tot_size:
            torch.save(m, '../output/train_model.pkl')
            print "save successfully"
            break


def tmp_data():
    train_arr = torch.rand(input_num_seqs, batch_size, input_channels_img, size_image, size_image)
    test_arr = torch.rand(input_num_seqs, batch_size, input_channels_img, size_image, size_image)
    # train_imgs_maps = torch.rand(input_num_seqs, batch_size, output_channels_img, size_image, size_image)
    # test_imgs_maps = torch.rand(input_num_seqs, batch_size, output_channels_img, size_image, size_image)

    return train_arr, test_arr


def load_train_data():
    path = '../input/train_data.pkl'  # to be confirmed
    train_arr = pickle.load(open(path, 'rb'))
    return train_arr


# total_data = load_train_data()
# train_size = int(len(total_data) * 0.7)
# train_arr, test_arr = total_data[:train_size], total_data[train_size:]


if __name__ == '__main__':
    m = HKOModel(inplanes=input_channels_img, input_num_seqs=input_num_seqs, output_num_seqs=output_num_seqs)
    if cuda_flag == True:
        m = m.cuda()

    mtest(input_channels_img, output_channels_img, size_image, max_epoch, model=m, cuda_test=cuda_flag)



