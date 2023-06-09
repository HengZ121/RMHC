In this branch,

1. calculated the auto-correlation window function for voxels
2. designed a convolutional neural network with 1 convolution layer with kernal size (10,10), 1 pooling layer with filter size (3,3), and 2 linear layer to draw 3 labels' probabilities from params
3. 3 labels are the tasks fMRI data corresponding to, DMS, NBACK, VSTM
4. minor class are duplicated to cope with class imbalance: NBACK x 2 and VSTM x 10

replace the paths in code to fMRI data folder and raw psychology data folder, and run <python3 classifier.py> to review


MOTOR1_left accuracy score is 81-91% when learning rate is 0.0004




