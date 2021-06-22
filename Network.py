import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical

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

        self.Q = nn.Linear(512, 4)

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

        self.Q = nn.Linear(512, 4)

        self.initialize_weights()
    
    def forward(self, x, hidden):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.ConvOutSize * self.ConvOutSize * 64)

        # Attention
        h_att = self.attention_h(hidden[0]).reshape(64)
        x = x.reshape((64,49))
        x = x.T
        z = F.tanh(self.attention_linear_x(x)+h_att)
        z = self.attention_linear_z(z)
        z = F.softmax(z)
        out = z*x
        lstm_input = torch.sum(out,0)
        lstm_input = lstm_input.reshape((1,64))

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

class ActorCriticNet(nn.Module):
    def __init__(self):
        super(ActorCriticNet, self).__init__()

        # layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.ConvOutSize = self.get_conv_out_size()

        self.fc = nn.Linear(self.ConvOutSize * self.ConvOutSize * 64, 512)

        self.Pi = nn.Linear(512, 4)
        self.V = nn.Linear(512, 1)

        self.initialize_weights()
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.ConvOutSize * self.ConvOutSize * 64)

        x = F.relu(self.fc(x))

        prob = self.Pi(x)
        prob = F.softmax(prob, dim=-1)

        value = self.V(x)

        return prob, value
    
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

class ActorCriticNet_LSTM(nn.Module):
    def __init__(self):
        super(ActorCriticNet_LSTM, self).__init__()

        # layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.ConvOutSize = self.get_conv_out_size()

        self.lstm = nn.LSTMCell(self.ConvOutSize * self.ConvOutSize * 64, 512)

        self.Pi = nn.Linear(512, 4)
        self.V = nn.Linear(512, 1)

        self.initialize_weights()
    
    def forward(self, x, hidden):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.ConvOutSize * self.ConvOutSize * 64)

        h, c = self.lstm(x, hidden)

        prob = self.Pi(h)
        prob = F.softmax(prob, dim=-1)

        value = self.V(h)

        return prob, value, (h, c)
    
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
