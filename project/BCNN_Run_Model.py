
from numpy.core.numeric import False_
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Acquisition_functions as af
import Supporting_functions as sf
import Models
import time


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        preds = model(images, training=True)
        #loss = loss_fn(labels, preds)
        batch_loss = loss_fn_noreduction(labels,preds)
        loss = tf.math.reduce_mean(batch_loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_acc_metric(labels, preds)
    return batch_loss

@tf.function
def test_step(images, labels):
    preds = model(images, training=False)
    t_loss = loss_fn(labels,preds)

    test_loss(t_loss)
    test_acc_metric(labels, preds)

class stats:
    thresh = 100
    dataused = []
    lam = 1.01
    class_names = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
var = stats()

name = 'cifar/normalt1'
root = '/home'
train_path = root+'/data/train_test/train/train/'
labels_path = root+'/data/trainLabels.csv'
test_path = root+'/data/tes' 
saved_model_path = root+'/project/saved_model/test1'
board_path = root + '/logs'

#Randomly initalise the difficulty csv
df = sf.init_diffs(labels_path)
print('Initalised Difficulty Dataframe')

#Split Data into train and test
train_df,test_df = sf.split_train_test(df,0.15)
print(train_df.head())
#Convert dataframe into datasets
test_ds = sf.collect_test_data(test_df,train_path,var).shuffle(1000)
print('Initalising Difficulty CSV')



#Load model
model = Models.AlexNet(10)
print('Model Loaded')

#evaluation metrics
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
loss_fn_noreduction = keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
#loss_fn = sf.elbo_loss
#loss_fn_noreduction = sf.elbo_loss_no_reduction

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc_metric = keras.metrics.CategoricalAccuracy()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc_metric = keras.metrics.CategoricalAccuracy()

print('Setup Complete, Starting training')

#Run the epochs
epochs = 15
BATCHES = 32

#Tensorboard Setup
info = '_E' +str(epochs)+'_B'+str(BATCHES)
train_log_dir = board_path + name + info + '/train'
test_log_dir = board_path + name + info + '/test'
cm_log_dir = board_path + name  + info + '/CM'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
cm_summary_writer = tf.summary.create_file_writer(cm_log_dir)



for epoch in range(epochs):
    
    train_ds = sf.collect_train_data('normal',train_df,var,train_path).batch(BATCHES)

    #training step
    for i,batch in enumerate(train_ds):
        #if i % 100 == 0:
        print("Batch = "+ str(i),end="\r")
        batch_loss = train_step(batch[0],batch[1])
        #update df
        train_df =sf.update_diffs_v3(train_df,batch,batch_loss)
    
    #Tensorboard updating
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy',train_acc_metric.result(), step=epoch)
    
    #test step
    for t_images, t_labels in test_ds.batch(BATCHES):
        test_step(t_images,t_labels)
    
    #Tensorboard updating
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step =epoch)
        tf.summary.scalar('accuracy', test_acc_metric.result(), step=epoch)
    
    #Printing to screen
    print('Epoch ',epoch+1,', Loss: ',train_loss.result().numpy(),', Accuracy: ',train_acc_metric.result().numpy(),', Test Loss: ',test_loss.result().numpy(),', Test Accuracy: ',test_acc_metric.result().numpy())
    
    #reset the metrics
    print('Reset States')
    train_loss.reset_states()
    train_acc_metric.reset_states()
    test_loss.reset_states()
    test_acc_metric.reset_states()

    #save the model
    if epoch % 10 ==0:
        model.save(saved_model_path)
        print('Checkpoint saved')
    
#save the diffs to seperate file
#np.savetxt('imagecounts/'+name,dataused)
model.save(saved_model_path)







