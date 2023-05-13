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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold


# Parameters:
epoch = 5
learning_rate = 0.01

### Define CNN
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size = (4, 4))
        # self.pool1 = nn.MaxPool2d((3, 1))

        # self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size = (4, 4))
        # self.pool2 = nn.MaxPool2d((3, 1))
        self.encoder1  =  nn.Sequential(
            nn.Linear(358, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 16),
            nn.Tanh(),
            nn.Linear(16, 3),
        )
        self.encoder2  =  nn.Sequential(
            nn.Linear(2902*3, 500),
            nn.Tanh(),
            nn.Linear(500, 100),
            nn.Tanh(),
            nn.Linear(100, 20),
            nn.Tanh(),
            nn.Linear(20, 3)
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(3,20),
            nn.Tanh(),
            nn.Linear(20, 100),
            nn.Tanh(),
            nn.Linear(100, 500),
            nn.Tanh(),
            nn.Linear(500, 2902*3),
        )
        self.decoder2  =  nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 50),
            nn.Tanh(),
            nn.Linear(50, 100),
            nn.Tanh(),
            nn.Linear(100, 358)
        )

    def forward(self, x):
        # x = self.pool1(F.relu(self.conv1(x)))
        # x = self.pool2(F.relu(self.conv2(x)))
        x = self.encoder1(x)
        x = x.ravel()
        encoded = self.encoder2(x)

        # print("******************************************")
        # print(encoded)
        # print("******************************************")

        decoded = self.decoder1(encoded)
        # print("******************************************")
        # print(decoded)
        # print("******************************************")
        decoded = torch.reshape(decoded, [2902,3])
        decoded = self.decoder2(decoded)
        return encoded,decoded

### Get Data

#Task Name
task = "DMS"
#Area of Brain
area = "DLPFC_left"

dataset = FMRIData(task, area)
dataset_cp = list(dataset)
shape = dataset.getImgShape()


AE = AutoEncoder()

optimizer = torch.optim.Adam(AE.parameters(),lr=learning_rate)
loss_func = nn.MSELoss()

for e in range(epoch):
    print("Epoch ", e, ":")
    random.shuffle(dataset_cp)
    for step,(x,y) in enumerate(dataset_cp):
        b_x = torch.tensor(np.array(x).reshape(shape[0],shape[1])).float()
        
        encoded,decoded = AE(b_x)
        loss = loss_func(decoded,b_x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch :', e,'|','train_loss:%.4f'%loss.data)
    random.shuffle(dataset_cp)
    for step,(x,y) in enumerate(dataset_cp):
        b_x = torch.tensor(np.array(x).reshape(shape[0],shape[1])).float()
        
        encoded,decoded = AE(b_x)
        loss = loss_func(decoded,b_x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch :', e,'|','train_loss:%.4f'%loss.data)


print('________________________________________')
print('finish training')

outputs = torch.empty((0, 3), dtype=torch.float32)
for data, _ in dataset:
    x = torch.tensor(np.array(data).reshape(shape[0],shape[1])).float()
    encoded_data, _ = AE(x)
    print(encoded_data)
    outputs = torch.cat((outputs, encoded_data.view(1,3)),0)

# print(encoded_data)

fig = plt.figure(2)
ax = Axes3D(fig)


X = outputs.data[:, 0].numpy()
Y = outputs.data[:, 1].numpy()
Z = outputs.data[:, 2].numpy()



values = np.array(dataset.images_descriptions)
for x, y, z, s in zip(X, Y, Z, values):
    ax.text(x, y, z, s)

ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())

plt.show()

plt.ion()
plt.show()


for data, description in dataset:
    x = torch.tensor(np.array(data).reshape(shape[0],shape[1])).float()
    _,result = AE(x)

    im_result = result
    # print(im_result.size())
    plt.figure(1, figsize=(10, 3))
    plt.subplot(121)
    plt.title(description)
    plt.imshow(x.numpy(),cmap='Greys', aspect='auto')

    plt.figure(1, figsize=(10, 3))
    plt.subplot(122)
    plt.title('Auto Encoder Edge')
    plt.imshow(im_result.detach().numpy(), cmap='Greys', aspect='auto')
    plt.show()
    plt.pause(1)

plt.ioff()

