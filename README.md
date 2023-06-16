In this branch,

1. calculated the auto-correlation window function for voxels
2. designed a convolutional neural network with 1 convolution layer with kernal size (10,10), 1 pooling layer with filter size (3,3), and 2 linear layer to draw 3 labels' probabilities from params
3. 3 labels are the tasks fMRI data corresponding to, DMS, NBACK, VSTM
4. minor class are duplicated to cope with class imbalance: NBACK x 2 and VSTM x 10

### How to run the code:

replace the paths in code to fMRI data folder and raw psychology data folder, and run <python3 classifier.py>

### Validation

The idea performance of models should be dollowing:

DLPFC_left accuracy score is varying from about 83 - 100 % 
DLPFC_right accuracy score is varying from about 73 - 100 % 
MOTOR1_left accuracy score is varying from 81-100% when learning rate is 0.0004
MOTOR1_right accuracy score is varying from 80-100%
VISUAL_left accuracy score is varying from about 92- 100 %  (2 out of 5 folds are 100%)
VISUAL_right accuracy score is varying from about 85- 100 %
Hippo_left accuracy score is varying from about 80 - 100 %

If the performance is worse than above, try to re-run the code for times

### Saved Models

This repo also contain the saved models, which achieved 100% accuracy in fold. (These .pt files)

### Note

It is possible to fine-tune the convolutional neural network for each dataset if interested





