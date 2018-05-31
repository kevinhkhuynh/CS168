# 2D Retinal Vessel Segmentation using Convolutional Neural Networks

Obtain retina images here:

https://www.isi.uu.nl/Research/Databases/DRIVE/

## Setup

Run the following requirements .sh file to obtain the needed libraries:
sh requirements.sh


Download the DRIVE dataset here: https://www.isi.uu.nl/Research/Databases/DRIVE/

In order to run this code, please set up the directory as follows:
```
CS168
|-- Scripts
|-- Data
|   |-- DRIVE
|   |   |-- test
|   |   |-- training
|-- README.md
```

<hr>

## Usage

### Scripts

Run these scripts from within the ```Scripts``` folder

Preprocess the images
```
python preprocess.py --total_patches 120000 
```

Train the model:

```
python train.py --batch 256 --learning_rate 5e-4 --training_prop 0.9 --max_steps 10000 --checkpoint_step 500 --loss_step 25 
```

Predict on the testing images using the trained model:

```
python test.py --fchu1 512 --format png --out ../Data/DRIVE/tmp/ --inp ../Data/DRIVE/test/ --model ../Data/models/model1/model.ckpt-7999

```

This code is an adaptation of https://github.com/KGPML/Deep-Vessel. However, we built our own neural network architecture on top of this code. We only used this code to preprocess the images and as a backend for the training.
