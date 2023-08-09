In this branch,

1. calculated the auto-correlation window function for voxels
2. designed a convolutional neural network with 1 convolution layer with kernal size (10,10), 1 pooling layer with filter size (3,3), and 2 linear layer to draw 3 labels' probabilities from params
3. 3 labels are the tasks fMRI data corresponding to, DMS, NBACK, VSTM
4. minor class are duplicated to cope with class imbalance: NBACK x 2 and VSTM x 10

### How to run the code:

replace the paths in code to fMRI data folder and raw psychology data folder, and run <python3 classifier.py>

### Saved Model (0.pt):

Fold  4
------------------------Training--------------------------------
████████████████████████████████████████████████████████████████| 80/80 [00:18<00:00,  4.22it/s] 
------------------------Evaluation--------------------------------
[1, 3, 2, 3, 1, 1, 1, 2, 0, 1, 2, 0, 0, 0, 3, 0, 3, 2, 3, 3, 0, 2, 3, 0, 0, 1, 2, 1, 2, 3, 0, 2] *actual labels
[1, 3, 2, 3, 1, 1, 0, 2, 0, 1, 2, 0, 0, 0, 3, 0, 3, 1, 3, 3, 0, 2, 3, 0, 0, 1, 2, 1, 2, 3, 0, 2] *output labels
*  Precision Score is:  0.9392857142857143
*  Recall Score is:  0.9330357142857143
*  Accuracy Score is:  0.9375
Model cached!
***************************************************************





