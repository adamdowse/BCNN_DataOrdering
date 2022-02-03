

from pickletools import optimize
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import time
import os

#https://openreview.net/pdf?id=Sk_P2Q9sG&source=post_page---------------------------
#read through and implement this


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    """Converts to one hot labels"""
    label = tf.one_hot(label,10)
    return tf.cast(image, tf.float32) / 255., label

#load the data and the model
data = tfds.load('mnist', split='train[90%:100%]', shuffle_files=True,as_supervised=True)
data = data.map(normalize_img)

bcnn_80 = tf.keras.models.load_model('/home/saved_model/saved_model_80')
bcnn_60 = tf.keras.models.load_model('/home/saved_model/saved_model_60')
bcnn_40 = tf.keras.models.load_model('/home/saved_model/saved_model_40')
bcnn_20 = tf.keras.models.load_model('/home/saved_model/saved_model_20')

bcnn_80 = tf.keras.Sequential([bcnn_80,tf.keras.layers.Softmax()])
bcnn_60 = tf.keras.Sequential([bcnn_60,tf.keras.layers.Softmax()])
bcnn_40 = tf.keras.Sequential([bcnn_40,tf.keras.layers.Softmax()])
bcnn_20 = tf.keras.Sequential([bcnn_20,tf.keras.layers.Softmax()])




#calculate the true per class model uncertainty
#Take n images and mean the standard deviations for correct class


samples=10
num_images = 1000
i = 0
runs_out = [[],[],[],[],[],[],[],[],[],[]]
mean_outputs = []
std_outputs = []

#take one batch from the data
data = data.batch(1)
for image,label in data.take(num_images):
    i+=1
    #run the same image through the network "samples" times
    true_label = np.argmax(tf.squeeze(label))
    for s in range(samples):
        #add the true label model output to corosponding array
        runs_out[true_label].append(tf.squeeze(bcnn_20(image))[true_label])
    if i % 10 == 0:
        print(i)
    
for a in range(10):
    mean_outputs.append(np.mean(runs_out[a]))
    std_outputs.append(np.std(runs_out[a]))

print(mean_outputs)
print(std_outputs)

plt.bar(range(1,11),mean_outputs,yerr=std_outputs,tick_label=range(10))
plt.legend()

plt.savefig('/home/project/Model_Outputs/class_total_uncertainty')
plt.close()