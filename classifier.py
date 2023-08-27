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
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

DLPFC_left = Dataset('DLPFC_left')
DLPFC_right = Dataset('DLPFC_right')
MOTOR1_left = Dataset('MOTOR1_left')
MOTOR1_right = Dataset('MOTOR1_right')
Hippo_left = Dataset('Hippo_left')
VISUAL1_left = Dataset('VISUAL1_left')
VISUAL1_right = Dataset('VISUAL1_right')

# Parameters:
batch_size = 10
brain_areas = [DLPFC_left, DLPFC_right, MOTOR1_left, MOTOR1_right, Hippo_left, VISUAL1_left, VISUAL1_right]
height = 0
width = 0

### Define CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (10, 10), padding= 0, stride = 1)
        self.pool1 = nn.MaxPool2d((3, 3))
        self.fc1 = nn.Linear(int((height - 9)/3) * int((width - 9)/3), 100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

kf = KFold(n_splits=5, shuffle=True)

for area in brain_areas:
    print('***************************************************************')
    print('* ', area.area)
    print('***************************************************************')
    fold = 1
    dataset_cp = list(area)
    mse = []
    r2 = []
    height = area.height
    width = area.width
    p = []
    r = []
    f1 = []

    best_a = 0
    best_p = 0
    best_r = 0

    for train_index, test_index in kf.split(area):
        print("Fold ", fold)
        l = []
        
        cnn = Net()
        train_subset = torch.utils.data.dataset.Subset(dataset_cp, train_index)
        test_subset = torch.utils.data.dataset.Subset(dataset_cp, test_index)
        traindataloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size , shuffle=True)
        testdataloader = torch.utils.data.DataLoader(test_subset, batch_size=1 , shuffle=True)
        optimizer = torch.optim.Adam(cnn.parameters(),lr=area.learning_rate)

        print('------------------------Training--------------------------------')
        for e in tqdm(range(area.epoch)):
            cnn.train()
            for x,y in traindataloader:
                x = x.unsqueeze(1)
                pred_y = cnn(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            l.append(loss.data)
            # print('Epoch :', e,'|','MSE train_loss:%.4f'%loss.data)

        print('------------------------Evaluation--------------------------------')
        cnn.eval()
        test_y = []
        pred_y = []
        for x, y in testdataloader:
            x = x.unsqueeze(1)
            test_y.append(y.tolist()[0])
            pred_y.append(cnn(x).tolist()[0])

        y_labels = []
        for y in test_y:
            y_labels.append(y.index(1))
        y_output_labels = []
        for y in pred_y:
            y_output_labels.append(y.index(max(y)))
        print(y_labels)
        print(y_output_labels)
        p.append(p_in_fold:=precision_score(y_labels, y_output_labels, zero_division=1, average= 'macro'))
        r.append(r_in_fold := recall_score(y_labels, y_output_labels, zero_division=1, average= 'macro'))
        f1.append(f1_in_fold := f1_score(y_labels, y_output_labels, zero_division=1, average= 'macro'))
        fold +=1

        # mse.append(mean_squared_error(test_y, pred_y))
        # r2.append(r2_score(test_y, pred_y))
        

        plt.figure(1, figsize=(10, 3))
        plt.subplot(121)
        plt.title("Actual (Blue) versus Prediceted (Red)")
        plt.scatter([x for x in range(len(y_labels))], [y for y in y_labels], color="blue")
        plt.scatter([x for x in range(len(y_output_labels))], [y for y in y_output_labels], color="red")

        plt.figure(1, figsize=(10, 3))
        plt.subplot(122)
        plt.title("Loss")
        plt.scatter([x for x in range(len(l))], [data for data in l], color="blue")
        plt.show()
        print("*", " Precision Score is: ", p_in_fold)
        print("*", " Recall Score is: ", r_in_fold)
        print("*", " F1 Score is: ", f1_in_fold)
        
        # if (fold_p > best_p and fold_r > best_r and fold_a > best_a):
        #     best_a, best_p, best_r = fold_a, fold_p, fold_r
        #     torch.save(cnn.state_dict(),''.join([area.area, '_', str(area.learning_rate), '_', str(area.epoch), '.pt']))
        #     print("Model cached!")
        
        print('***************************************************************')
        print()
    print("*", " Precision Score is: ", fold_p := statistics.mean(p))
    print("*", " Recall Score is: ", fold_r := statistics.mean(r))
    print("*", " F1 Score is: ", fold_a := statistics.mean(f1))