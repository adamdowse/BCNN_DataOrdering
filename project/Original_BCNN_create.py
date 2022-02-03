#Setup
from pickletools import optimize
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import time
import os


def build_BCNN_model(input_shape):
    #2 conv layers and 2 dense layers
    model = tf.keras.Sequential()
    model.add(tfp.layers.Convolution2DFlipout(32, kernel_size=(3,3), padding="same", strides=2, input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tfp.python.layers.Convolution2DFlipout(64, kernel_size=(3, 3), padding="same", strides=2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tfp.python.layers.DenseFlipout(512, activation='relu'))
    model.add(tfp.python.layers.DenseFlipout(10))
    return model

@tf.function
def elbo_loss(labels, logits):
    loss_en = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    loss_kl = tf.keras.losses.KLD(labels, logits)
    loss = tf.reduce_mean(tf.add(loss_en, loss_kl))
    return loss

@tf.function
def train_step(images,labels):
    with tf.GradientTape() as tape:
        logits = bcnn(images, training=True)
        loss = elbo_loss(labels,logits)
    gradients = tape.gradient(loss,bcnn.trainable_variables)
    optimizer.apply_gradients(zip(gradients, bcnn.trainable_variables))
    return loss

def sum_correct_labels(preds,labels):
    #simply sums up the correct pairs to average later
    acc = 0
    for p,l in zip(preds,labels):
        if np.argmax(p) == np.argmax(l):
            acc += 1
    return acc

# Construct a tf.data.Dataset
train_ds,test_ds = tfds.load('cifar10', split=['train[:80%]','train[80%:90%]'], shuffle_files=True,as_supervised=True)
batch_size = 128

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    """Converts to one hot labels"""
    label = tf.one_hot(label,10)
    return tf.cast(image, tf.float32) / 255., label

train_ds = train_ds.map(normalize_img).batch(batch_size)
test_ds = test_ds.map(normalize_img).batch(batch_size)


bcnn = build_BCNN_model([28,28,1])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

#Training time
times = []
accs = []
test_accs = []
losses = []
test_losses = []
for i in range(5):
    tic = time.time()
    #training steps
    acc = 0
    j=0
    loss = 0
    for images,labels in train_ds:
        j += tf.shape(images)[0].numpy()
        loss += train_step(images,labels) 
        preds = bcnn(images)
        acc += sum_correct_labels(preds, labels)
    acc = acc/j
    loss = loss/j
    accs.append(acc)
    losses.append(loss)
    
    #testing steps
    test_acc = 0
    test_loss = 0
    j = 0
    for images,labels in test_ds:
        j += tf.shape(images)[0].numpy()
        test_preds = bcnn(images)
        test_loss += elbo_loss(labels, test_preds)
        test_acc += sum_correct_labels(labels, test_preds)
    test_acc = test_acc/j
    test_loss = test_loss/j
    test_accs.append(test_acc)
    test_losses.append(test_loss)

    tock = time.time()
    train_time = tock-tic
    times.append(train_time)
    
    print("Epoch: {}: loss = {:7.3f} , accuracy = {:7.3f}, val_loss = {:7.3f}, val_acc={:7.3f} time: {:7.3f}".format(i, loss, acc, test_loss, test_acc, train_time))

bcnn.save('saved_model/saved_model_80')


