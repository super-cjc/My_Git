import numpy as np
import torch
char_list = list('hello')

char_index = {c:i for i,c in enumerate(char_list)}
print(char_index)
input_str = "hello"
input_data = [char_index[c] for c in input_str]
print(input_data)
input_one_hot = np.eye(len(char_list))[input_data]
input_torch = torch.tensor(input_one_hot, dtype=torch.float32)
print(input_torch)
target_str = "eollh"
target_input = [char_index[c] for c in target_str]
print(target_input)
targets = torch.tensor(target_input, dtype=torch.float32)
print(targets)
