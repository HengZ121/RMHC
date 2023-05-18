import statistics

from data import *
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

DLPFC_left = Dataset('DMS', 'DLPFC_left')
DLPFC_right = Dataset('DMS', 'DLPFC_right')
Hippo_left = Dataset('DMS', 'Hippo_left')
MOTOR1_left = Dataset('DMS', 'MOTOR1_left')
VISUAL1_left = Dataset('DMS', 'VISUAL1_left')
VISUAL1_right = Dataset('DMS', 'VISUAL1_right')

brain_areas = [DLPFC_left, DLPFC_right, Hippo_left, MOTOR1_left, VISUAL1_left, VISUAL1_right]

kf = KFold(n_splits=5)
for area in brain_areas:
    fold = 1
    print('***************************************************************')
    print('* ', area.area)
    print('***************************************************************')
    p = []
    r = []
    a = []

    for train_index, test_index in kf.split(area):
        # print("Fold ", fold)

        x_train = [area.features[x] for x in train_index]
        y_train = [area.labels[x] for x in train_index]
        x_test = [area.features[x] for x in test_index]
        y_test = [area.labels[x] for x in test_index]

        # clf = svm.SVC()
        clf = LogisticRegression(random_state=1)
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        p.append(precision_score(y_test, y_pred, zero_division=1))
        r.append(recall_score(y_test, y_pred, zero_division=1))
        a.append(accuracy_score(y_test, y_pred))
        fold+=1

        p.sort()
        r.sort()
        a.sort()

    print("*", " Average Precision Score is: ", statistics.mean(p[1:5]))
    print("*", " Average Recall Score is: ", statistics.mean(r[1:5]))
    print("*", " Average Accuracy Score is: ", statistics.mean(a[1:5]))
    print('***************************************************************')
    print()