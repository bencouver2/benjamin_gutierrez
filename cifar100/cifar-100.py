from __future__ import division
from __future__ import absolute_import
"""
Benjamin Gutierrez Garcia Mayo 2018
Codigo de clasificacion del dataset CIFAR-100 usanndo Tensorflow, TFLearn 
Y Tensorboard. 
Disculpas or mezclar ingles y espanol.
http://tflearn.org/getting_started/

Run with python3  on Ubuntu 16.04.4 LTS
Required: python3,pip3,tensorflow, tflearn, tensorboard y paquetes abajo 
importados
"""
import numpy as np
import tensorflow as tf
import urllib.request
import tarfile
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d
from tflearn.layers.conv import max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

#Tensorflow configuration
tf.logging.set_verbosity(tf.logging.INFO)

debug=1

"""
Note that the 'encoding' parameter is only valid for Python 3
From https://www.cs.toronto.edu/~kriz/cifar.html
benjamin@higgs:~/cifar-100$ ls cifar-100-python/
file.txt~  meta  test  train
Each of these files is a Python "pickled" object produced with cPickle. Here is a python3
routine which will open such a file and return a dictionary:
"""

def unpickle(file):
     import pickle
     with open(file, 'rb') as fo:
#    Original encoding from the CIFAR-100 site, 
#    Not good in bytes, you cant use the keys
#         dict = pickle.load(fo, encoding='bytes')
         dict = pickle.load(fo, encoding='latin1')
     return dict


def download_dataset():
    print('Fetching fresh copy of the dataset with urllib2...')
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'  
    urllib.request.urlretrieve(url, './cifar-100-python.tar.gz')  
    target_folder = '.'
    with tarfile.open("./cifar-100-python.tar.gz") as tar:
      
      import os
      
      def is_within_directory(directory, target):
          
          abs_directory = os.path.abspath(directory)
          abs_target = os.path.abspath(target)
      
          prefix = os.path.commonprefix([abs_directory, abs_target])
          
          return prefix == abs_directory
      
      def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
      
          for member in tar.getmembers():
              member_path = os.path.join(path, member.name)
              if not is_within_directory(path, member_path):
                  raise Exception("Attempted Path Traversal in Tar File")
      
          tar.extractall(path, members, numeric_owner=numeric_owner) 
          
      
      safe_extract(tar, target_folder)

""" 
The following are standard preparation for categorical Data.  
X must be change to a to 4 dimensional tensor: 
[records, width, height, channels]
"""

def tensor4d(unpick_data):
#Create a numpy array from unpickled data
    unpick_data_float = np.array(unpick_data, dtype=float) 
    proper4d = unpick_data_float.reshape([-1, 3, 32, 32])
#For an n-D array, if axes are given, their order indicates how the axes are permuted 
    proper4d = proper4d.transpose([0, 2, 3, 1])
    return proper4d
"""
Some algorithms can work with categorical data directly.
This means that categorical data must be converted to a numerical form. 
This is where the integer encoded variable is removed and a new binary variable 
is added for each unique integer value. In the “color” variable example, there are 3 
categories and therefore 3 binary variables are needed. A “1” value is placed in the 
binary variable for the color and “0” values for the other colors.
See https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/

red,	green,	blue
1,		0,	0
0,		1,	0
0,		0,	1
"""
def encoding_one_hot(labels,number_clases):
#   Return a 2-D array with ones on the diagonal and zeros elsewhere.
    return np.eye(number_clases)[labels]

"""
Fetch Dataset
"""

download_dataset()

#Exploration of the dataset dictionary for debugging purposes
if debug==1:
   data = unpickle('cifar-100-python/train')
   len(data)
   data.keys()
   data["data"].shape
   del data

 
""" Ingestion"""

X = tensor4d(unpickle('cifar-100-python/train')['data'])
Y = encoding_one_hot(unpickle('cifar-100-python/train')['fine_labels'],100)
X_test = tensor4d(unpickle('cifar-100-python/test')['data'])
Y_test = encoding_one_hot(unpickle('cifar-100-python/test')['fine_labels'],100)

"""
Data Preprocessing and Data Augmentation
From tutorial http://tflearn.org/getting_started/
"""
# Real-time image preprocessing
img_prep = tflearn.ImagePreprocessing()
# Zero Center (With mean computed over the whole dataset)
img_prep.add_featurewise_zero_center()
# STD Normalization (With std computed over the whole dataset)
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
# Random flip and rotatean image
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=15.)
# Add these methods into an 'input_data' layer
# CIFAR-60 are 60k images 32x32 with 3 fields deep color RGB
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

""" 
Model
http://tflearn.org/models/dnn/
network = ... (some layers) ...
network = regression(network, optimizer='sgd', loss='categorical_crossentropy')

model = DNN(network)
model.fit(X, Y)
"""

network = tflearn.conv_2d(network, 32, 3, strides=1, padding='same',bias=True,bias_init='zeros', 
          weights_init='uniform_scaling',activation='relu')
network = max_pool_2d(network, 2 , strides=None, padding='same')
network = tflearn.conv_2d(network, 64, 3, strides=1, padding='same', activation='relu',
                  bias_init='zeros', weights_init='uniform_scaling', bias=True)
network = tflearn.conv_2d(network, 64, 3 , padding='same', activation='relu', bias=True, 
                  bias_init='zeros', weights_init='uniform_scaling', strides=1)
network = max_pool_2d(network, 2 , strides=None, padding='same')
network = fully_connected(network, 600, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 100, activation='softmax')
# Optimizer, Objective and Metric:
network = tflearn.regression(network, optimizer='adam',learning_rate=0.001, loss='categorical_crossentropy')

"""
Execution of the Model
"""                     

with tf.device('cpu:0'):
    #Possible Arguments see http://tflearn.org/models/dnn/.
    #tensorboard_verbose=3 is needed for Tensorboard. Use Firefox, graphs dont show in Chrome
    model = tflearn.DNN(network, tensorboard_verbose=3)
    model.fit(X, Y, n_epoch=50, run_id='CIFAR100-BEN',shuffle=True, 
              validation_set=(X_test, Y_test), show_metric=True, batch_size=100)

