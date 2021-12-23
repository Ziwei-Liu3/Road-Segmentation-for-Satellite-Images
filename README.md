# Project Road Segmentation

This project focuses on an image binary semantic segmentation problem, which extract roads from a set of satellite images. We implemented a model based on Dinknet152, and also introduced data augmentation schemes and mechanism to fine-tune hyperparameters. We finally reached an accuracy of 0.957 and a F1 score of 0.918, ranking 7th on AIcrowd leaderboard. 

## Structure of the project
```
├── README.md
├── dataset
│   ├── test_set_images
│   └── train
│       ├── groundtruth
│       └── images
│
├── networks
│   ├── __init__.py
│   └── dinknet.py
│
├── submits
│   └── DinkNet152_0.957 on AICrowd
│
├── weights
├── DinkNet152_0.957 on AICrowd.csv
├── data_preprocessing.py
├── framework.py
├── loss.py
├── mask_to_submission.py
├── run.py
├── test.py
└── train.py
```
- `README.md`: that is this file.
- `/networks/dinknet.py` :  contains all the neural network models the program uses. Here we define the LinkNet and DinkNet for training and comparison.
- `submits`: the folder where the testing prediction will be stored. We also uplaod the prediction images that gain use the best score on AIcorwd in the folder `DinkNet152_0.957 on AICrowd`.
- `weights`: The folder where the optimal weight(.th file) generated during the training phase will be stored. When doing the test, the .th file in this folder will also be used. Given that the size of the .th file that reach our best result on AIcrowd is so large(more than 1G) that it exceeds github's file size limitation, we decide not to upload this file. Please feel free to contact me at yanni.zhang@epfl.ch if you need it.
- `DinkNet152_0.957 on AICrowd.csv`: the file that gets us the best result on AIcrowd. This file is generated from the `/submits/DinkNet152_0.957 on AICrowd` using `mask_to_submission.py`.
- `data_peprocessing.py`: the file that performs data preprocessing, including resizing, rotation, scaling,flip, color space transformation, etc.
- `framework.py`: the file that performs general framework manipulation.
- `loss.py`: the file that defines the loss function(BCE loss, Dice loss)
- `mask_to_submission.py` : the file that transfers the resulting pictures in `/submits/SpecificName` folder to the .csv file.
- `run.py`: the file that integrate data preprocessing, training, and testing.
- `test.py`: the file that applies test-time augmentation when generating the result of images in `/dataset/test_set_images`
- `train.py`: the file that perfom the train

## Usage
- You can run this project by using following command:
```
python run.py
```
Both of the training and testing are porformed on Google Colab GPU(12G). The training phase takes about 2200 seconds, and the testing takes about an hour.
