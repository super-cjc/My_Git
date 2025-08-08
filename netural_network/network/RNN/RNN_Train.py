import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from RNN_NetWork import RNNNet
import Config


# 定义模型
model = RNNNet(Config.RNNConfig['input_size'], Config.RNNConfig['hidden_size'],
               Config.RNNConfig['output_size'])

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()    # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=Config.RNNConfig['learning_rate'])

losses = []
hidden = None

def RNNTrain():
    global hidden

