import torch
import random
import statistics
import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
import torch.nn.functional as F

from data import *
from tqdm import tqdm
from statistics import mean
from sklearn import metrics
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# DLPFC_left = Dataset('DMS', 'DLPFC_left')
# DLPFC_right = Dataset('DMS', 'DLPFC_right')
MOTOR1_left = Dataset('DMS', 'MOTOR1_left')
# VISUAL1_left = Dataset('DMS', 'VISUAL1_left')
# VISUAL1_right = Dataset('DMS', 'VISUAL1_right')

# Parameters:
epoch = 20
learning_rate1 = 0.01
batch_size = 1
# brain_areas = [DLPFC_left, DLPFC_right, MOTOR1_left, VISUAL1_left, VISUAL1_right]
brain_areas = [MOTOR1_left]

### Define CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 5), padding= 0, stride = 1)
        self.fc1 = nn.Linear(6, 1)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x

kf = KFold(n_splits=5)
for area in brain_areas:
    fold = 1
    print('***************************************************************')
    print('* ', area.area)
    print('***************************************************************')

    dataset_cp = list(area)
    mse = []
    r2 = []

    for train_index, test_index in kf.split(area):
        print("Fold ", fold)
        
        cnn = Net()
        train_subset = torch.utils.data.dataset.Subset(dataset_cp, train_index)
        test_subset = torch.utils.data.dataset.Subset(dataset_cp, test_index)
        traindataloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size , shuffle=True)
        testdataloader = torch.utils.data.DataLoader(test_subset, batch_size=1 , shuffle=True)
        optimizer = torch.optim.Adam(cnn.parameters(),lr=learning_rate1)
        loss_func = nn.MSELoss()

        print('------------------------Training--------------------------------')
        for e in range(epoch):
            cnn.train()
            print("Epoch ", e, ":")
            for x,y in traindataloader:
                x = x.unsqueeze(1)
                pred_y = cnn(x)
                loss = loss_func(pred_y, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Epoch :', e,'|','AEH train_loss:%.4f'%loss.data)

        
        fold +=1
        print('------------------------Evaluation--------------------------------')
        cnn.eval()
        test_y = []
        pred_y = []
        for x, y in testdataloader:
            x = x.unsqueeze(1)
            test_y.append(y)
            pred_y.append(cnn(x).item())

        
        mse.append(mean_squared_error(test_y, pred_y))
        r2.append(r2_score(test_y, pred_y))
        

        plt.figure(1, figsize=(10, 3))
        plt.subplot(121)
        plt.title("Actual Label " + area.area + str(fold))
        ax = plt.gca()
        ax.set_ylim([-1.1, 1.1])
        plt.scatter([x for x in range(len(test_y))], test_y, color="blue")

        plt.figure(1, figsize=(10, 3))
        plt.subplot(122)
        plt.title("Predicted Label")
        ax = plt.gca()
        ax.set_ylim([-1.1, 1.1])
        plt.scatter([x for x in range(len(pred_y))], pred_y, color="red")
        plt.show()
        fold+=1

    print("*", " Average MSE Score is: ", statistics.mean(mse))
    print("*", " Average R2 Score is:  ", statistics.mean(r2))
    print('***************************************************************')
    print()