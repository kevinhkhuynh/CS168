# 2D Retinal Vessel Segmentation using Convolutional Neural Networks

Download the DRIVE dataset here: https://www.isi.uu.nl/Research/Databases/DRIVE/

## Setup

Run the following requirements .sh file to obtain the needed libraries:
```
sh requirements.sh
```

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

## Usage

Run these scripts from within the ```Scripts``` folder

Preprocess the images:
```
python preprocess.py --total_patches 120000 
```

##### train.py

```
Usage: train.py [OPTIONS]

Options:
  --batch BATCH                           Batch Size [Default - 64]
  --fchu1 FCHU1                           Number of hidden units in FC1 layer [Default - 512]
  --learning_rate LEARNING_RATE           Learning rate for optimiser [Default - 5e-4]
  --training_prop TRAINING_PROP           Proportion of data to be used for training data [Default - 0.8]
  --max_steps MAX_STEPS                   Maximum number of iteration till which the program must run [Default - 100]
  --checkpoint_step CHECKPOINT_STEP       Step after which an evaluation is carried out on validation set and model is saved [Default - 50]
  --loss_step LOSS_STEP                   Step after which loss is printed [Default - 5]
  --keep_prob KEEP_PROB                   Keep Probability for dropout layer [Default - 0.5]
  --model_name MODEL_NAME                 Index of the model [Default - '1']
```

Train our model:
```
python train.py --batch 256 --learning_rate 5e-4 --training_prop 0.9 --max_steps 10000 --checkpoint_step 500 --loss_step 25 
```

##### test.py

```
Usage: test.py [OPTIONS]

Options:
  --fchu1 FCHU1    Number of hidden units in FC1 layer. This should be identical to the one used in the model 
                   [Default - 256]
  --out OUT        Directory to put rendered images to
  --inp INP        Directory containing images for testing
  --model MODEL    Path to the saved tensorflow model checkpoint
  --format FORMAT  Format to save the images in. [Available formats: npz, jpg and png]

```
Example 

Predict on the testing images using our trained model:
```
python test.py --fchu1 512 --format png --out ../Data/DRIVE/tmp/ --inp ../Data/DRIVE/test/ --model ../Data/models/model1/model.ckpt-7999
```


Example usage:

```
python train.py --batch 256 --learning_rate 5e-4 --training_prop 0.9 --max_steps 10000 --checkpoint_step 500 --loss_step 25 
```

##### test.py

```
Usage: test.py [OPTIONS]

Options:
  --fchu1 FCHU1    Number of hidden units in FC1 layer. This should be identical to the one used in the model 
                   [Default - 256]
  --out OUT        Directory to put rendered images to
  --inp INP        Directory containing images for testing
  --model MODEL    Path to the saved tensorflow model checkpoint
  --format FORMAT  Format to save the images in. [Available formats: npz, jpg and png]

```

This code is an adaptation of https://github.com/KGPML/Deep-Vessel. However, we built our own neural network architecture on top of this code. We only used this code to preprocess the images and as a backend for the training. The neural network architecture, the requirements.sh, and the code to obtain the error metrics for the predictions is our original work.
