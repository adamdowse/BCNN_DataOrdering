
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

#Aleatoric uncertainty
#this is the output of each classes distribution
samples=100
i = 0
#take one batch from the data
data = data.batch(1)
for image,label in data.take(5):
    i+=1
    #run the same image through the network "samples" times
    runs_out = []
    for s in range(samples):
        runs_out.append((bcnn_80(image)))

    mean_outputs = tf.math.reduce_mean(runs_out,0)
    std_outputs = tf.math.reduce_std(runs_out,0)
    true_label = np.argmax(tf.squeeze(label))

    plt.bar(range(1,11),tf.squeeze(mean_outputs),yerr=tf.squeeze(std_outputs),label=true_label,tick_label=range(10))
    plt.legend()
    plt.yscale('log')
    plt.savefig('/home/project/Model_Outputs/output_'+str(i))
    plt.close()


   
#inputting a random noise image
for image,label in data.take(1):
    noise = tf.random.normal(shape=(1,28,28,1), mean=0.0, stddev=(20)/(255), dtype=tf.dtypes.float32)
    true_label = np.argmax(tf.squeeze(label))
    noise_std_outputs = []
    noise_mean_outputs = []
    for i in range(10):
        #run the same image through the network "samples" times
        runs_out = []
        noise_out = []
        
        for s in range(samples):
            noise_out.append(bcnn_80(image))
        image = image + noise #add more noise each time

        noise_mean_outputs.append(tf.math.reduce_mean(noise_out,0)[:,true_label])
        noise_std_outputs.append(tf.math.reduce_std(noise_out,0)[:,true_label])


    plt.plot(noise_mean_outputs)
    plt.plot(np.array(noise_mean_outputs) + np.array(noise_std_outputs))
    plt.plot(np.array(noise_mean_outputs) - np.array(noise_std_outputs))
    plt.savefig('/home/project/increasing_noise_80')
    plt.close()

# testing different models together

for image,label in data.take(1):
    out_80 = []
    out_60 = []
    out_40 = []
    out_20 = []
    for s in range(samples):
        out_80.append(bcnn_80(image))
        out_60.append(bcnn_60(image))
        out_40.append(bcnn_40(image))
        out_20.append(bcnn_20(image))

    mean_80 = tf.math.reduce_mean(out_80,0)
    std_80 = tf.math.reduce_std(out_80,0)
    mean_60 = tf.math.reduce_mean(out_60,0)
    std_60 = tf.math.reduce_std(out_60,0)
    mean_40 = tf.math.reduce_mean(out_40,0)
    std_40 = tf.math.reduce_std(out_40,0)
    mean_20 = tf.math.reduce_mean(out_20,0)
    std_20 = tf.math.reduce_std(out_20,0)

    true_label = np.argmax(tf.squeeze(label))

    width=0.2
    plt.bar(np.arange(10) -(width*1.5),  tf.squeeze(mean_80),yerr=tf.squeeze(std_80),label=(str(true_label)+" 80%"),tick_label=range(10), alpha=0.7,width=width)
    plt.bar(np.arange(10) -(width/2),  tf.squeeze(mean_60),yerr=tf.squeeze(std_60),label=(str(true_label)+" 60%"),tick_label=range(10), alpha=0.7,width=width)
    plt.bar(np.arange(10) +(width/2),tf.squeeze(mean_40),yerr=tf.squeeze(std_40),label=(str(true_label)+" 40%"),tick_label=range(10), alpha=0.7,width=width)
    plt.bar(np.arange(10) +(width*1.5),tf.squeeze(mean_20),yerr=tf.squeeze(std_20),label=(str(true_label)+" 20%"),tick_label=range(10), alpha=0.7,width=width)
    plt.legend()
    plt.yscale('log')
    plt.savefig('/home/project/Model_Outputs/output_size')
    plt.close()

