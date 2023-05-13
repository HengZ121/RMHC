import torch
import random
import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
import torch.nn.functional as F

from fmri import *
from tqdm import tqdm
from statistics import mean
from sklearn import metrics
from ae1 import AutoEncoder as AE1
from ae2 import AutoEncoder as AE2
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold


# Parameters:
epoch = 5
learning_rate = 0.00003

### Get Data
#Task Name
task = "DMS"
#Area of Brain
area = "DLPFC_left"

dataset = FMRIData(task, area)
dataset_cp = list(dataset)
shape = dataset.getImgShape()


AE1 = AE1()

# AE2 = AE2()

optimizer1 = torch.optim.Adam(AE1.parameters(),lr=learning_rate)
# optimizer2 = torch.optim.Adam(AE2.parameters(),lr=learning_rate)
loss_func = nn.MSELoss()

for e in range(epoch):
    print("Epoch ", e, ":")
    random.shuffle(dataset_cp)
    for step,(x,y) in enumerate(dataset_cp):
        x = torch.tensor(np.array(x).reshape(shape[0],shape[1])).float()
        encoded1,decoded1 = AE1(x)
        loss = loss_func(decoded1, x)
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        print('Epoch :', e,'|','AEH train_loss:%.4f'%loss.data)
    random.shuffle(dataset_cp)
    for step,(x,y) in enumerate(dataset_cp):
        x = torch.tensor(np.array(x).reshape(shape[0],shape[1])).float()
        encoded1,decoded1 = AE1(x)
        loss = loss_func(decoded1, x)
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        print('Epoch :', e,'|','AEH train_loss:%.4f'%loss.data)


    # for step,(x,y) in enumerate(dataset_cp):
    #     x = torch.tensor(np.array(x).reshape(shape[0],shape[1])).float()
    #     encoded1,decoded1 = AE1(x)
    #     en_x = encoded1.ravel()
    #     encoded2, decoded2 = AE2(en_x)
    #     loss = loss_func(decoded2, en_x)
    #     optimizer2.zero_grad()
    #     loss.backward()
    #     optimizer2.step()
    #     print('Epoch :', e,'|','AEL train_loss:%.4f'%loss.data)


print('________________________________________')
print('finish training')


# outputs = torch.empty((0, 3), dtype=torch.float32)
# for data, _ in dataset:
#     x = torch.tensor(np.array(data).reshape(shape[0],shape[1])).float()
#     encoded_x, _ = AE1(x)
#     encoded_x = encoded_x.ravel()
#     encoded_x, _ = AE2(encoded_x)
#     print(encoded_x)
#     outputs = torch.cat((outputs, encoded_x.view(1,3)),0)
# fig = plt.figure(2)
# ax = Axes3D(fig)

# X = outputs.data[:, 0].numpy()
# Y = outputs.data[:, 1].numpy()
# Z = outputs.data[:, 2].numpy()

# values = np.array(dataset.images_descriptions)
# for x, y, z, s in zip(X, Y, Z, values):
#     ax.text(x, y, z, s)

# ax.set_xlim(X.min(), X.max())
# ax.set_ylim(Y.min(), Y.max())
# ax.set_zlim(Z.min(), Z.max())
# plt.show()


plt.ion()
plt.show()


for data, description in dataset:
    x = torch.tensor(np.array(data).reshape(shape[0],shape[1])).float()

    encoded_x, decoded_x = AE1(x)
    # encoded_x = encoded_x.ravel()
    # encoded_x, decoded_x = AE2(encoded_x)
    # decoded_x = torch.reshape(decoded_x, [2902,3])

    im_result = decoded_x
    # print(im_result.size())
    plt.figure(1, figsize=(10, 3))
    plt.subplot(121)
    plt.title(description)
    plt.imshow(x.numpy(),cmap='Greys', vmin = 0, vmax= 8000, aspect='auto')

    plt.figure(1, figsize=(10, 3))
    plt.subplot(122)
    plt.title('Auto Encoder Edge')
    plt.imshow(im_result.detach().numpy(), vmin = 0, vmax= 8000, cmap='Greys', aspect='auto')
    plt.show()
    plt.pause(1)

plt.ioff()

