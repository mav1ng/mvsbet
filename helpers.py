import data as d
import numpy as np
import network as n
import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
import torch.optim as optim


def train(nb_epochs, bs, c, lr):

    loss_list = []
    accuracy_list = []

    device = torch.device('cuda:0')

    dataset = d.FootballData('data/data.hkl')
    dataloader = DataLoader(dataset, batch_size=bs,
                            shuffle=True, num_workers=0)
    model = n.Net()
    model.to(torch.double)
    model.to(device)
    criterion = n.MSE_Odds(c=c).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.001)

    for epoch in range(nb_epochs):

        running_loss = 0.0
        running_corr = 0.
        running = 0.
        for i, batch in enumerate(dataloader):
            input1 = torch.tensor(batch[0], device=device).permute(0, 2, 1)
            input2 = torch.tensor(batch[1], device=device)
            odds = torch.tensor(batch[1][:, -3:], device=device)
            results = torch.tensor(batch[2], device=device)

            print(input1.size(), input2.size(), odds.size(), results.size())

            input1.requires_grad = True
            input2.requires_grad = True
            odds.requires_grad = True
            results.requires_grad = True

            optimizer.zero_grad()
            outputs = model(input1, input2)
            loss = criterion(outputs, results, odds)
            if loss.item() != loss.item():
                print('nan')
                break
            loss.backward()

            # for p in model.parameters():
            #     print(p.grad)

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            for b in range(outputs.size(0)):
                if torch.argmax(outputs[b]) == torch.argmax(results[b]):
                    running_corr += 1.
                    running += 1.
                else:
                    running += 1.

            if i % 50 == 49:  # print every 2000 mini-batches
                loss_list.append(running_corr / 50)
                accuracy_list.append(running_corr / running)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                print('[%d, %5d] Accuracy: %.3f' %
                      (epoch + 1, i + 1, running_corr/running))
                running_corr = 0.
                running = 0.
                running_loss = 0.0
        print('Finished Training')
    return loss_list, accuracy_list


def lin_train(nb_epochs, bs, c, lr):

    loss_list = []
    accuracy_list = []

    device = torch.device('cuda:0')

    dataset = d.FootballData('data/data.hkl')
    dataloader = DataLoader(dataset, batch_size=bs,
                            shuffle=True, num_workers=0)
    model = n.LinNet()
    model.to(torch.double)
    model.to(device)
    criterion = n.MSE_Odds(c=c).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.001)

    for epoch in range(nb_epochs):

        running_loss = 0.0
        running_corr = 0.
        running = 0.
        for i, batch in enumerate(dataloader):
            input1 = torch.tensor(batch[0], device=device).permute(0, 2, 1)
            input2 = torch.tensor(batch[1], device=device)
            odds = torch.tensor(batch[1][:, -3:], device=device)
            results = torch.tensor(batch[2], device=device)

            input1.requires_grad = True
            input2.requires_grad = True
            odds.requires_grad = True
            results.requires_grad = True

            optimizer.zero_grad()
            outputs = model(input2)
            loss = criterion(outputs, results, odds)
            if loss.item() != loss.item():
                print('nan')
                break
            loss.backward()

            # for p in model.parameters():
            #     print(p.grad)

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            for b in range(outputs.size(0)):
                if torch.argmax(outputs[b]) == torch.argmax(results[b]):
                    running_corr += 1.
                    running += 1.
                else:
                    running += 1.

            if i % 50 == 49:  # print every 2000 mini-batches
                loss_list.append(running_corr / 50)
                accuracy_list.append(running_corr / running)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                print('[%d, %5d] Accuracy: %.3f' %
                      (epoch + 1, i + 1, running_corr/running))
                running_corr = 0.
                running = 0.
                running_loss = 0.0
        print('Finished Training')
    return loss_list, accuracy_list