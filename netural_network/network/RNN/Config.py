"""
参数定义
"""
import numpy as np
import torch

# 数据集准备
# char_to_index = {'h':0,'e':1,'l':3,'o':4}
char_str = list('hello')
char_to_index = {c:i for i,c in enumerate(char_str)}
index_to_char = {i:c for i,c in enumerate(char_str)}


# 数据准备(将输入的字母数据转化为数字)
input_str = "hello"
target_str = "eollh"
input_data = [char_to_index[c] for c in input_str]      # input_data = [0,1,3,3,4]
target_data = [char_to_index[c] for c in target_str]

# 转化为独热码
input_one_hot = np.eye(len(char_str))[input_data]

# 转化为tensor
# 张量表
# tensor([[1., 0., 0., 0., 0.],
#         [0., 1., 0., 0., 0.],
#         [0., 0., 0., 1., 0.],
#         [0., 0., 0., 1., 0.],
#         [0., 0., 0., 0., 1.]])
inputs  = torch.tensor(input_one_hot, dtype=torch.float32)
targets = torch.tensor(target_data, dtype=torch.float32)    # tensor([1., 4., 3., 3., 0.])


# RNN参数
RNNConfig = {
    # 模型超参数
    'input_size': len(char_str),
    'hidden_size': 8,
    'output_size': len(char_str),
    'num_epochs': 100,
    'learning_rate': 0.1,
}
