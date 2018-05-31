
# coding: utf-8

# In[ ]:

from __future__ import division
import numpy as np
import pandas as pd
import os
import glob
from skimage import io, color, measure
from skimage.util import img_as_float, img_as_ubyte
import tensorflow as tf
import time
from six.moves import xrange 
import argparse

# In[ ]:
mean_np_img = None
# In[ ]:

PATCH_DIM = 31
BATCH_SIZE = 100 # Must be a perfect square
NUM_CLASSES = 2
OUT_DIR = os.path.abspath("../Data/DRIVE/tmp/model1")
IN_DIR = os.path.abspath("../Data/DRIVE/test")
MODEL_PATH = os.path.abspath("../Data/models/model1/model.ckpt-7999")
FCHU1 = 256
FORMAT = 'npz'

h = int(PATCH_DIM/2)
# We want to access the data chunk by chunk such that each chunk has
# approximately BATCH_SIZE pixels
stride = int(np.sqrt(BATCH_SIZE))

# In[ ]:

def get_path(directory):
    imgs = glob.glob(directory + '/images/*.tif')
    imgs.sort()
    #a = [x.split('/')[-1].split('.')[0] for x in train]

    mask = glob.glob(directory + '/mask/*.gif')
    mask.sort()
    #b = [x.split('/')[-1].split('.')[0] for x in mask]

    gt = glob.glob(directory + '/1st_manual/*.gif')
    gt.sort()
    #c = [x.split('/')[-1].split('.')[0] for x in gt]

    return map(os.path.abspath, imgs), map(os.path.abspath, mask), map(os.path.abspath, gt)

def conv2d(x, W, b, padding="SAME", strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def inference(images, keep_prob, fc_hidden_units1):
    weights = {
        'wc1': tf.get_variable('W0', shape=[6,6,3,48], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'wc2': tf.get_variable('W1', shape=[5,5,48,48], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'wc3': tf.get_variable('W2', shape=[4,4,48,48], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'wc4': tf.get_variable('W3', shape=[2,2,48,48], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'wd1': tf.get_variable('W4', shape=[4*4*48,fc_hidden_units1], initializer=tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('W6', shape=[fc_hidden_units1,NUM_CLASSES], initializer=tf.contrib.layers.xavier_initializer()),
    }
    biases = {
        'bc1': tf.get_variable('B0', shape=[48], initializer=tf.constant_initializer(0.05)),
        'bc2': tf.get_variable('B1', shape=[48], initializer=tf.constant_initializer(0.05)),
        'bc3': tf.get_variable('B2', shape=[48], initializer=tf.constant_initializer(0.05)),
        'bc4': tf.get_variable('B3', shape=[48], initializer=tf.constant_initializer(0.05)),
        'bd1': tf.get_variable('B4', shape=[fc_hidden_units1], initializer=tf.constant_initializer(0.05)),
        'out': tf.get_variable('B5', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.05)),
    }

    # Flattening the 3D image into a 1D array
    x_image = tf.reshape(images, [-1,PATCH_DIM,PATCH_DIM,3])

    # Convolution Layer
    conv1 = conv2d(x_image, weights['wc1'], biases['bc1'], padding="VALID")

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

    # Max-pooling Layer
    pool1 = maxpool2d(conv2, k=2)

    # Convolution Layer
    conv3 = conv2d(pool1, weights['wc3'], biases['bc3'])

    # Max-pooling Layer
    pool2 = maxpool2d(conv3, k=2)

    # Convolution Layer
    conv4 = conv2d(pool2, weights['wc4'], biases['bc4'])

    # Max-pooling Layer
    pool3 = maxpool2d(conv4, k=2)

    # Flatten
    fc1 = tf.reshape(pool3, [-1,weights['wd1'].get_shape().as_list()[0]])

    # Fully connected layer
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Dropout
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    # Final fully connected layer
    out = tf.add(tf.matmul(fc1_drop, weights['out']), biases['out'])
    return out

# In[ ]:

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    Args:
        batch_size: The batch size will be baked into both placeholders.
    Returns:
        images_placeholder: Images placeholder.
    """
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, PATCH_DIM**2*3))
    return images_placeholder


# In[ ]:

def softmax(logits):
    """ Performs softmax operation on logits

        Args:
            logits: logits from inference module
        Output:
            Softmax of logits
    """
    return tf.nn.softmax(logits)

# In[ ]:

def nbd(image, point):
    """ Finds neighborhood around a point in an image

        Args:
            image: Input image
            point: A point around which we would like to find the neighborhood

        Output:
            1d vector of size [PATCH_DIM*PATCH_DIM*3] which is a neighborhood
            aroud the point passed in the parameters list
    """
    i = point[0]
    j = point[1]
    h = int(PATCH_DIM/2)
    return image[i-h:i+h+1,j-h:j+h+1].reshape(-1)


# In[ ]:

def decode(test, mask_test):
    """Segments images in a directory using a given model and saves the images in a particular format
        to an output directory. The ensemble version of the decoder relies on this script to decode
        the same images using different models

        Args:
            test:       Paths to test images
            mask_test:  Paths to corresponding masks
    """

    begin = time.time()
    with tf.Graph().as_default():
        # Generate placeholders for the images and  keep_probability
        images_placeholder = placeholder_inputs(BATCH_SIZE)
        keep_prob = tf.placeholder(tf.float32)


        # Build a Graph that computes predictions from the inference model.
        logits = inference(images_placeholder, keep_prob, FCHU1)
        sm = softmax(logits)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        with tf.Session() as sess:
            saver.restore(sess, MODEL_PATH)

            # Once the model has been restored, we iterate through all images in the test set
            for im_no in xrange(len(test)):

                start_time = time.time()
                print "Working on image %d" % (im_no+1)
                image = io.imread(test[im_no])
                mask = img_as_float(io.imread(mask_test[im_no]))

                # We will start with a completely black image and update it chunk by chunk
                segmented = np.zeros(image.shape[:2])

                # We will use arrays to index the image and mask later
                cols, rows = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
                row_col = np.stack([rows,cols], axis = 2)
                # The neighborhood windows to be fed into the graph
                feed = np.zeros((BATCH_SIZE, PATCH_DIM**2*3))
                # The predicted classes for all the windows that were fed to the graph
                predictions = np.zeros(BATCH_SIZE)

                pixel_count = 0


                i = h+1
                while i < image.shape[0] - h-2:
                    j = h+1
                    while j < image.shape[1] - h-1:
                        # A small check is made to ensure that not all pixels are black

                        # Update i and j by adding stride but take care near the end
                        i_next = min(i+stride, image.shape[0]-h-1)
                        j_next = min(j+stride, image.shape[1]-h-1)

                        if int(np.max(mask[i:i_next,j:j_next])) == 1:

                            pixel_count += BATCH_SIZE # This will not be true for border cases though
                                                      # but we don't care about the progress at the end

                            # Once we get a chunk, we flatten it and map a function that returns
                            # the neighborhood of each point

                            #feed = np.array(map(lambda p: nbd(image, p), row_col[i:i_next, j:j_next].reshape(-1,2)))
                            #print " Feed shape = (%d, %d)" % feed.shape
                            chunk = np.array(map(lambda p: nbd(image, p), row_col[i:i_next, j:j_next].reshape(-1,2)))
                            feed[:len(chunk)] = chunk

                            # Subtract training mean image
                            feed = feed - mean_np_img

                            # Get predictions and draw accordingly on black image
                            predictions = sess.run([sm],
                                           feed_dict={images_placeholder: feed,
                                                      keep_prob: 1.0})
                            predictions = np.asarray(predictions).reshape(BATCH_SIZE, NUM_CLASSES)

                            # Uncomment following line for non-probability plotting
                            #predictions = np.argmax(predictions, axis=1)
                            predictions = predictions[:,1]

                            if not len(chunk) == BATCH_SIZE:
                                predictions = predictions[:len(chunk)]
                            segmented[rows[i:i_next, j:j_next], cols[i:i_next, j:j_next]] = predictions.reshape(i_next-i, j_next-j)

                            # Reset everything after passing feed to feedforward
                            feed = np.zeros((BATCH_SIZE, PATCH_DIM**2*3))
                            predictions = np.zeros(BATCH_SIZE)

                        j += stride
                    i += stride
                segmented = np.multiply(segmented,mask)

                name = test[im_no].split('/')[-1].split('.')[0]
                if FORMAT == 'npz':
                    np.savez(os.path.join(OUT_DIR, name+'.npz'), segmented)
                elif FORMAT == 'jpg' or FORMAT == 'png':
                    segmented = segmented * (1.0/segmented.max())
                    io.imsave(os.path.join(OUT_DIR, name+'.'+FORMAT), segmented)
                else:
                    print "Unknown format. Saving as png."
                    segmented = segmented * (1.0/segmented.max())
                    io.imsave(os.path.join(OUT_DIR, name+'.png'), segmented)

                current_time = time.time()
                print "Time taken - > %f" % (current_time - start_time)
                start_time = current_time

    print "Total time = %f mins" % ((time.time()-begin)/60.0)

def finish_parsing():

    global OUT_DIR, IN_DIR, MODEL_PATH, FCHU1, FORMAT

    parser = argparse.ArgumentParser(description=
                                     'Script to decode images using a single model')
    parser.add_argument("--fchu1", type=int,
                help="Number of hidden units in FC1 layer. This should be identical to the one used in the model [Default - 256]")

    parser.add_argument("--out",
                        help="Directory to put rendered images to")
    parser.add_argument("--inp",
                        help="Directory containing images for testing")
    parser.add_argument("--model",
                        help="Path to the saved tensorflow model checkpoint")
    parser.add_argument("--format",
                        help="Format to save the images in. [Available formats: npz, jpg and png]")

    args = parser.parse_args()

    if args.fchu1 is not None:
        FCHU1 = args.fchu1
        print "New FCHU1 = %d" % FCHU1
    if args.out is not None:
        OUT_DIR = args.out
        print "New OUT_DIR = %s" % OUT_DIR
    if args.inp is not None:
        IN_DIR = args.inp
        print "New IN_DIR = %s" % IN_DIR
    if args.model is not None:
        MODEL_PATH = args.model
        print "New MODEL_PATH = %s" % MODEL_PATH
    if args.format is not None:
        FORMAT = args.format
        print "New FORMAT = %s" % FORMAT

def main():
    finish_parsing()

    global mean_np_img

    mean_img = pd.read_pickle('../Data/mean_img_no_class_bias.pkl')
    mean_np_img = np.asarray(mean_img)

    test, mask_test, _ = get_path(IN_DIR)

    # Make a directory to store the new images in
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    decode(test, mask_test)



if __name__ == "__main__":
    main()
