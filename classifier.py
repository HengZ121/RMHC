import torch
import statistics
import torch.nn as nn
import torch.nn.functional as F

from data import *
from tqdm import tqdm
from statistics import mean
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

DLPFC_left = Dataset('DLPFC_left')
# DLPFC_right = Dataset('DLPFC_right')
# MOTOR1_left = Dataset('MOTOR1_left')
# MOTOR1_right = Dataset('MOTOR1_right')
# Hippo_left = Dataset('Hippo_left')
# VISUAL1_left = Dataset('VISUAL1_left')
# VISUAL1_right = Dataset('VISUAL1_right')

# Parameters:
batch_size = 10
# brain_areas = [DLPFC_left, DLPFC_right, MOTOR1_left, VISUAL1_left, VISUAL1_right]
brain_areas = [ DLPFC_left]
height = 0
width = 0

### Define CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (10, 10), padding= 0, stride = 1)
        self.pool1 = nn.MaxPool2d((3, 3))
        self.fc1 = nn.Linear(int((height - 9)/3) * int((width - 9)/3), 100)
        self.fc2 = nn.Linear(100, 1)

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
                loss = torch.nn.functional.mse_loss(pred_y, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print('Epoch :', e,'|','MSE train_loss:%.4f'%loss.data)
            l.append(loss.data)

        print('------------------------Evaluation--------------------------------')
        mse = []
        r2 = []
        cnn.eval()
        test_y = []
        pred_y = []
        for x, y in testdataloader:
            x = x.unsqueeze(1)
            test_y.append(y.tolist()[0])
            pred_y.append(cnn(x).tolist()[0])

        print(test_y)
        print(pred_y)
        fold +=1

        mse.append(mean_squared_error(test_y, pred_y))
        r2.append(r2_score(test_y, pred_y))
        

        plt.figure(1, figsize=(10, 3))
        plt.subplot(121)
        plt.title("Actual (Blue) versus Prediceted (Red)")
        plt.scatter([x for x in range(len(test_y))], [y for y in test_y], color="blue")
        plt.scatter([x for x in range(len(pred_y))], [y for y in pred_y], color="red")

        plt.figure(1, figsize=(10, 3))
        plt.subplot(122)
        plt.title("Loss")
        plt.scatter([x for x in range(len(l))], [data for data in l], color="blue")
        plt.show()

        
        print("*", " MSE is: ", fold_p := statistics.mean(mse))
        print("*", " R2 Score is: ", fold_r := statistics.mean(r2))
        # if (fold_p > best_p and fold_r > best_r and fold_a > best_a):
        #     best_a, best_p, best_r = fold_a, fold_p, fold_r
        #     torch.save(cnn.state_dict(),''.join([area.area, '_', str(area.learning_rate), '_', str(area.epoch), '.pt']))
        #     print("Model cached!")
        
        print('***************************************************************')
        print()