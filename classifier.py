import statistics

from data import *
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

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
    mse = []

    for train_index, test_index in kf.split(area):
        # print("Fold ", fold)

        x_train = [area.features[x] for x in train_index]
        y_train = [area.labels[x] for x in train_index]
        x_test = [area.features[x] for x in test_index]
        y_test = [area.labels[x] for x in test_index]

        # clf = svm.SVC()
        # clf = LogisticRegression(random_state=1)
        clf = Ridge(alpha = 1)
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        mse.append(mean_squared_error(y_test, y_pred))
        fold+=1

        mse.sort()

    print("*", " Average MSE Score is: ", statistics.mean(mse[1:5]))
    print('***************************************************************')
    print()