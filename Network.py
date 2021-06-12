import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable

import numpy as np

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()

        # layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.ConvOutSize = self.get_conv_out_size()

        self.fc = nn.Linear(self.ConvOutSize * self.ConvOutSize * 64, 512)

        self.Q = nn.Linear(512, 5)

        self.initialize_weights()
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.ConvOutSize * self.ConvOutSize * 64)

        x = F.relu(self.fc(x))
        q = self.Q(x)
        return q
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def get_conv_out_size(self):
        test_tensor = torch.FloatTensor(1, 4, 84, 84)
        out_tensor = self.conv3(self.conv2(self.conv1(test_tensor)))
        conv_out_size = out_tensor.size()[-1]
        return conv_out_size

class QNet_LSTM(nn.Module):
    def __init__(self):
        super(QNet_LSTM, self).__init__()

        # layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.ConvOutSize = self.get_conv_out_size()

        self.lstm = nn.LSTMCell(self.ConvOutSize * self.ConvOutSize * 64, 512)

        self.Q = nn.Linear(512, 5)

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
        self.attention_h = nn.Linear(64,64)
        # self.attention_xW = Variable(torch.randn((64),requires_grad=True)).to("cuda")
        # self.attention_xb = Variable(torch.randn((64), requires_grad=True)).to("cuda")
        self.attention_linear_x = nn.Linear(64,64)
        self.attention_linear_z = nn.Linear(64,64)


        self.lstm = nn.LSTMCell(64, 64)
        self.Q = nn.Linear(64, 5)

        self.initialize_weights()

    def forward(self, x, hidden):

        # CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # print(x.size())
        # x = x.view(-1, self.ConvOutSize * self.ConvOutSize * 64)


        # attention
        # x = self.attention_linear_one(x)
        # temp = self.attention_W(hidden[0])
        # z = x.add(temp)
        # z = self.attention_linear_two(z)
        # z = F.softmax(z)
        # z = x*z
        x = x.view(-1, 64, self.ConvOutSize * self.ConvOutSize) # 1, 64, 49,
        # k = torch.chunk(x,49,2) # (1,64) * 49
        # for i,chunk in enumerate(k):
        #     chunk = chunk.reshape((1,64))
        #     # h_att = torch.matmul(hidden[0],self.attention_hW) + self.attention_hb
        #     h_att = self.attention_h(hidden[0]) #64
        #     x_att = self.attention_linear_x(chunk)
        #     z = F.tanh(h_att + x_att)
        #     z = self.attention_linear_z(z)
        #     z = F.softmax(z)
        #     if i == 0:
        #         out = z*chunk
        #     elif i > 0:
        #         out = torch.cat((out,z*chunk))
        h_att = self.attention_h(hidden[0]).reshape(64) #64
        # print(f"h_att size = {h_att.size()}")
        x = x.reshape((64,49)) # 64 ,49
        x = x.T # 49 ,64
        # print(f"x_1 size = {x.size()}") # 49, 64
        x = self.attention_linear_x(x) #(49,64) * (64) + (64) = (49,64)
        # print(f"x_2 size = {x.size()}")
        z = F.tanh(x+h_att) # (49,64) + (49,64) = (49,64)
        z = self.attention_linear_z(z) # (49,64)
        z = F.softmax(z) # (49,64)
        out = z*x # (49,64)
        # print(f"out size = {out.size()}")
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
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def get_conv_out_size(self):
        test_tensor = torch.FloatTensor(1, 1, 84, 84)
        out_tensor = self.conv3(self.conv2(self.conv1(test_tensor)))
        conv_out_size = out_tensor.size()[-1]
        return conv_out_size