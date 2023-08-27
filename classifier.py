import torch
import statistics
import torch.nn as nn
import torch.nn.functional as F

from data import *
from tqdm import tqdm
from statistics import mean
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

ds = Dataset()

# Parameters:
batch_size = 8
epoch = 80
lr = 0.0002
height = 0
width = 0

### Define CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (5, 5), padding= 0, stride = 1)
        self.pool1 = nn.MaxPool2d((3, 3))
        self.fc1 = nn.Linear(int((height - 4)/3) * int((width - 4)/3), 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

kf = KFold(n_splits=8, shuffle=True)


print('***************************************************************')
fold = 1
dataset_cp = list(ds)
mse = []
r2 = []
height = ds.height
width = ds.width

best_a = 0
best_p = 0
best_r = 0

for train_index, test_index in kf.split(ds):
    print("Fold ", fold)
    
    l = []

    cnn = Net()
    train_subset = torch.utils.data.dataset.Subset(dataset_cp, train_index)
    test_subset = torch.utils.data.dataset.Subset(dataset_cp, test_index)
    traindataloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size , shuffle=True)
    testdataloader = torch.utils.data.DataLoader(test_subset, batch_size=1 , shuffle=True)
    optimizer = torch.optim.Adam(cnn.parameters(),lr=lr)

    print('------------------------Training--------------------------------')
    for e in tqdm(range(epoch)):
        cnn.train()
        for x,y in traindataloader:
            x = x.unsqueeze(1)
            pred_y = cnn(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print('Epoch :', e,'|','MSE train_loss:%.4f'%loss.data)
        l.append(loss.data)

    print('------------------------Evaluation--------------------------------')
    p = []
    r = []
    a = []
    
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

    p.append(precision_score(y_labels, y_output_labels, zero_division=1, average= 'macro'))
    r.append(recall_score(y_labels, y_output_labels, zero_division=1, average= 'macro'))
    a.append(accuracy_score(y_labels, y_output_labels))
    fold +=1
    

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

    
    print("*", " Precision Score is: ", fold_p := statistics.mean(p))
    print("*", " Recall Score is: ", fold_r := statistics.mean(r))
    print("*", " Accuracy Score is: ", fold_a := statistics.mean(a))


    if (fold_p > 0.9 and fold_r > 0.9 and fold_a > 0.9):
        torch.save(cnn.state_dict(),'0'.join(['.pt']))
        print("Model cached!")
    
    print('***************************************************************')
    print()