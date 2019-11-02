import numpy as np
import pandas as pd
import torch
import model
from torch.utils import data
from torch import optim, nn
from torch.nn import functional as F


def load_data(train_data, train_label, test_data, test_label):
    train_data = train_data[:, np.newaxis]
    train_label = np.squeeze(train_label, axis=1)
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_label = torch.tensor(train_label)
    dataset = data.TensorDataset(train_data, train_label)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)

    test_data = test_data[:, np.newaxis]
    test_label = np.squeeze(test_label, axis=1)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_label = torch.tensor(test_label)
    dataset_test = data.TensorDataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batchsize, shuffle=False)
    return data_loader, test_loader


def train(epoch, train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = net(data)
        loss = criterion(output, target)
        if batch_idx % 20 == 0:
            print('Epoch:[{}/{}] [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, epochs, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(test_loader):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = net(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def run():
    train_loader, test_loader = load_data(train_data, train_label, test_data, test_label)
    for epoch in range(1, epochs):
        train(epoch, train_loader)
        evaluate(test_loader)


if __name__ == '__main__':

    batchsize = 128
    epochs = 10000

    train_data = np.array(pd.read_csv('data/train.csv', header=None))
    train_label = np.array(pd.read_csv('data/trainlabel.csv', header=None))
    test_data = np.array(pd.read_csv('data/test.csv', header=None))
    test_label = np.array(pd.read_csv('data/testlabel.csv', header=None))

    device = torch.device("cuda")
    net = model.LSTM().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    run()