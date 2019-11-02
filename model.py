import torch.nn as nn
from torch.nn import functional as F


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size=14,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # return F.log_softmax(out)
        return out


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 5),
            nn.BatchNorm1d(128),
        )
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(14, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out