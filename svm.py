from data import *
from sklearn import svm

from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

dataset = Dataset('DMS', 'DLPFC_left')
kf = KFold(n_splits=5)
fold = 1
for train_index, test_index in kf.split(dataset):
    print("Fold ", fold)

    x_train = [dataset.features[x] for x in train_index]
    y_train = [dataset.labels[x] for x in train_index]
    x_test = [dataset.features[x] for x in test_index]
    y_test = [dataset.labels[x] for x in test_index]

    clf = svm.SVC()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print("#", fold, " FOLD: Precision Score is: ", p := precision_score(y_test, y_pred))
    print("#", fold, " FOLD: Recall Score is: ", r := recall_score(y_test, y_pred))
    print("#", fold, " FOLD: Accuracy Score is: ", a := accuracy_score(y_test, y_pred))
    fold+=1