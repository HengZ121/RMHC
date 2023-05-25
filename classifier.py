import statistics

from data import *
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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
    r2 = []

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
        r2.append(r2_score(y_test, y_pred))
        fold+=1

        plt.figure(1, figsize=(10, 3))
        plt.subplot(121)
        plt.title("Actual Label " + area.area + str(fold))
        plt.scatter([x for x in range(len(y_test))], y_test, color="blue")

        plt.figure(1, figsize=(10, 3))
        plt.subplot(122)
        plt.title("Predicted Label")
        plt.scatter([x for x in range(len(y_test))], y_pred, color="red")
        plt.show()

    print("*", " Average MSE Score is: ", statistics.mean(mse))
    print("*", " Average R2 Score is:  ", statistics.mean(r2))
    print('***************************************************************')
    print()