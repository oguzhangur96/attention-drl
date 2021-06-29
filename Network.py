import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear 
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable

import numpy as np

class QNet_LSTM(nn.Module):
    def __init__(self):
        super(QNet_LSTM, self).__init__()

        # layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.ConvOutSize = self.get_conv_out_size()

        self.lstm = nn.LSTMCell(self.ConvOutSize * self.ConvOutSize * 64, 512)

        self.Q = nn.Sequential(nn.Linear(512, 64),
                                nn.ReLU(64),
                                nn.Linear(64,4)
                                )

        self.initialize_weights()
    
    def forward(self, x, hidden):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.ConvOutSize * self.ConvOutSize * 64)

        h, c = self.lstm(x, hidden)

        q = self.Q(h)

        return q, (h, c)
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def get_conv_out_size(self):
        test_tensor = torch.FloatTensor(1, 1, 84, 84)
        out_tensor = self.conv3(self.conv2(self.conv1(test_tensor)))
        conv_out_size = out_tensor.size()[-1]
        return conv_out_size

class QNet_DARQN(nn.Module):
    def __init__(self):
        super(QNet_DARQN, self).__init__()

        # layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.ConvOutSize = self.get_conv_out_size()
        self.attention_h = nn.Sequential(nn.Linear(512,128,bias=False),
                                        nn.ReLU(128),
                                        nn.Linear(128,64, bias=False)
                                        )

        self.attention_linear_x = nn.Sequential(nn.Linear(64,64),
                                                nn.ReLU(64),
                                                nn.Linear(64,64)
                                                )

        self.attention_linear_z = nn.Linear(64,64)


        self.lstm = nn.LSTMCell(64, 512)
        self.Q = nn.Sequential(nn.Linear(512, 64),
                                nn.ReLU(64),
                                nn.Linear(64,4)
                                )

        self.initialize_weights()

    def forward(self, x, hidden):
        # CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 64, self.ConvOutSize * self.ConvOutSize) # 1, 64, 49,
        h_att = self.attention_h(hidden[0])
        x = x.reshape((64,49)) # 64 ,49
        x = x.T # 49 ,64
        x_att = self.attention_linear_x(x)
        z = torch.tanh(x_att+h_att) # (49,64) + (49,64) = (49,64)
        z = self.attention_linear_z(z) # (49,64)
        z = torch.softmax(z, dim=1) # (49,64)
        out = z*x # (49,64)
        lstm_input = torch.sum(out,0) # 64
        lstm_input = lstm_input.reshape((1,64)) # 1 ,64
        #LSTM
        h, c = self.lstm(lstm_input, hidden)

        q = self.Q(h)
        return q, (h, c)

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if not module.bias == None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def get_conv_out_size(self):
        test_tensor = torch.FloatTensor(1, 1, 84, 84)
        out_tensor = self.conv3(self.conv2(self.conv1(test_tensor)))
        conv_out_size = out_tensor.size()[-1]
        return conv_out_size


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
        # self.initialize_weights()
        # self.conv1 = nn.Conv2d(2, 32, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.ConvOutSize = self.get_conv_out_size()

    def forward(self, x):
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
        """
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def get_conv_out_size(self):
        test_tensor = torch.FloatTensor(1, 1, 84, 84)
        out_tensor = self.conv3(self.conv2(self.conv1(test_tensor)))
        conv_out_size = out_tensor.size()[-1]
        return conv_out_size


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        self.lstm = nn.LSTMCell(729, 512)
        self.Q = nn.Linear(512, 12)

    def forward(self, x, hidden):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting

        out = x_out * scale
        lstm_input = torch.sum(out, 0)  # 64
        lstm_input = lstm_input.reshape((1, 729))  # 1 ,64
        # LSTM
        h, c = self.lstm(lstm_input, hidden)
        q = self.Q(h)
        return q, (h, c)
        # return x * scale
