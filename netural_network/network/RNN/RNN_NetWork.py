import torch
import torch.nn as nn



class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNNet, self).__init__()
        # batch_first = True
        # out: (batch_size, seq_len, hidden_size)
        self.rnn = nn.RNN(input_size,hidden_size,output_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)

        # 前向传输
        def forward(self,x,hidden):
            # out: (batch_size, seq_len, hidden_size)
            out,hidden = self.rnn(x,hidden)
            out = self.fc(out)
            return out,hidden



