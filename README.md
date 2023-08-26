In this branch,

1. calculated the auto-correlation window function for voxels
2. designed a convolutional neural network with 1 convolution layer with kernal size (5,5), 1 pooling layer with filter size (3,3), and 2 linear layer to draw 4 labels' probabilities from params
3. labels are the cortexes' names

### How to run the code:

replace the paths in code to fMRI data folder and raw psychology data folder, and run <python3 classifier.py>

### Saved Model (0.pt):

Fold  8
------------------------Training--------------------------------
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:14<00:00,  5.34it/s] 
------------------------Evaluation--------------------------------
[3, 0, 1, 3, 1, 3, 3, 2, 0, 2, 1, 1, 2, 0, 2, 2, 2, 0, 0, 2, 1, 0, 0, 2, 3, 0, 2, 1, 1, 2, 3]
[3, 0, 1, 3, 1, 3, 3, 3, 0, 1, 1, 1, 2, 0, 2, 2, 2, 0, 0, 2, 1, 0, 0, 2, 3, 0, 2, 1, 1, 2, 3]
*  Precision Score is:  0.9330357142857143
*  Recall Score is:  0.95
*  Accuracy Score is:  0.9354838709677419
Model cached!
***************************************************************




