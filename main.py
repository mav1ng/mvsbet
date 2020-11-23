import data as d
import numpy as np
import network as n
import helpers as h
import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
import matplotlib.pyplot as plt

# us, ws, fd = d.process_raw_data()
# data = d.get_base_data(us, ws, fd)
# tdata = d.prepare_data(data)
# tdata = d.prepare_short_data(tdata)
# d.save_np(tdata, 'data/data.hkl')
# tdata = d.load_np('data/data.hkl')
# a, a_rat, b, b_rat = d.get_train_data(tdata, date=tdata[0, 120], home=tdata[1, 120], away=tdata[2, 120])
"""

team_dict = d.prepare_team_dict()
print(team_dict['Schalke 04'])
print(team_dict['Augsburg'])
print('180505')

loss_list, accuracy_list = h.lin_train(10, 1, 0.0, 0.002)

x = np.arange(0, len(loss_list))
plt.plot(x, np.array(loss_list))
plt.show()

x = np.arange(0, len(accuracy_list))
plt.plot(x, np.array(accuracy_list))
plt.show()
"""

dataset = d.FootballData('data/data.hkl')
dataloader = DataLoader(dataset, batch_size=dataset.__len__(),
                        shuffle=True, num_workers=0)

beta = np.zeros(28)


def linreg(x, y):
    x = np.array(x).T
    y = np.array(y)
    print(x.shape)
    print(y.shape)
    return np.matmul(np.linalg.inv(np.matmul(x.T, x)), np.matmul(x.T, y))


for i, batch in enumerate(dataloader):
    beta1 = linreg(batch[1].T, batch[2][:, 0].T)
    beta2 = linreg(batch[1].T, batch[2][:, 1].T)
    beta3 = linreg(batch[1].T, batch[2][:, 2].T)

    print(beta1)
    print(beta2)
    print(beta3)

    print(beta3.shape)
    print(np.array(batch[1]).T.shape)

    pred1 = np.dot(np.array(batch[1]), beta1)
    pred2 = np.dot(np.array(batch[1]), beta2)
    pred3 = np.dot(np.array(batch[1]), beta3)
    pred = np.concatenate([np.expand_dims(pred1, axis=1), np.expand_dims(pred2, axis=1),
                                                       np.expand_dims(pred3, axis=1)], axis=1)
    print(pred1.shape)

    reg = np.array(batch[2]) - pred
    print(pred.shape)
    print(np.array(batch[2]).shape)

    acc = np.sum(np.argmax(pred, axis=1) == np.argmax(np.array(batch[2]), axis=1)) / pred.shape[0]
    print(acc)


for i, batch in enumerate(dataloader):
    beta1 = linreg(batch[0].T, batch[2][:, 0].T)
    beta2 = linreg(batch[0].T, batch[2][:, 1].T)
    beta3 = linreg(batch[0].T, batch[2][:, 2].T)

    print(beta1)
    print(beta2)
    print(beta3)

    print(beta3.shape)
    print(np.array(batch[0]).T.shape)

    pred1 = np.dot(np.array(batch[0]), beta1)
    pred2 = np.dot(np.array(batch[0]), beta2)
    pred3 = np.dot(np.array(batch[0]), beta3)
    pred = np.concatenate([np.expand_dims(pred1, axis=1), np.expand_dims(pred2, axis=1),
                                                       np.expand_dims(pred3, axis=1)], axis=1)
    print(pred1.shape)

    reg = np.array(batch[2]) - pred
    print(pred.shape)
    print(np.array(batch[2]).shape)

    acc = np.sum(np.argmax(pred, axis=1) == np.argmax(np.array(batch[2]), axis=1)) / pred.shape[0]
    print(acc)
