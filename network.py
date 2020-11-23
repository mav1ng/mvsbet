import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.dropout_rate = 0.2

        self.conv1 = nn.Conv1d(10, 20, 3, padding=1)
        self.dropout_1 = nn.Dropout(p=self.dropout_rate)

        self.conv2 = nn.Conv1d(20, 8, 3, padding=1)
        self.dropout_2 = nn.Dropout(p=self.dropout_rate)

        self.conv3 = nn.Conv1d(8, 1, 3, padding=1)
        self.dropout_3 = nn.Dropout(p=self.dropout_rate)

        self.fc1 = nn.Linear(69, 32)
        self.dropout_4 = nn.Dropout(p=self.dropout_rate)

        self.fc2 = nn.Linear(32, 16)
        self.dropout_5 = nn.Dropout(p=self.dropout_rate)

        self.fc3 = nn.Linear(16, 3)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):

        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.dropout_1(x)

        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.dropout_2(x)

        x = self.conv3(x)
        x = torch.tanh(x)
        x = self.dropout_3(x)

        x = torch.cat([x[:, 0, :].clone(), y], dim=1)
        x = torch.tanh(self.fc1(x))
        x = self.dropout_4(x)
        x = torch.tanh(self.fc2(x))
        x = self.dropout_5(x)
        x = self.fc3(x)
        x = torch.tanh(x)

        x = self.softmax(x)

        return x


class LinNet(nn.Module):

    def __init__(self):
        super(LinNet, self).__init__()

        self.dropout_rate = 0.2

        self.fc1 = nn.Linear(7, 32)
        self.dropout_4 = nn.Dropout(p=self.dropout_rate)

        self.fc2 = nn.Linear(32, 16)
        self.dropout_5 = nn.Dropout(p=self.dropout_rate)

        self.fc3 = nn.Linear(16, 3)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = torch.tanh(self.fc1(x))
        x = self.dropout_4(x)
        x = torch.tanh(self.fc2(x))
        x = self.dropout_5(x)
        x = self.fc3(x)
        x = torch.tanh(x)

        x = self.softmax(x)

        return x


class MSE_Odds(nn.Module):
    def __init__(self, c):
        super(MSE_Odds, self).__init__()

        self.c = c

    def forward(self, pred, lab, odds):
        (bs, n) = pred.size()
        loss = 1/bs * torch.sum(torch.pow(pred - lab, 2) - self.c * torch.pow(pred - 1 / odds, 2))
        return loss