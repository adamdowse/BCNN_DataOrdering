
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


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        preds = model(images, training=True)
        loss = loss_fn(labels, preds)
        batch_loss = loss_fn_noreduction(labels,preds)
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


class_names = ['Airplane','automobile','Bird','cat','Deer','Dog','Frog','Horse','Ship','Truck']

class stats:
    thresh = 100
    dataused = [0]
    lam = 1.01

var = stats()


name = 'cifar/var_threshold_t100_l1.01'
root = '/user/HS223/ad00878/PhD/BCNN_DataOrdering/BCNN_DataOrdering'
train_path = 'E:/OneDrive/Documents/Uni/PG/Thesis/Code/cifar/cifar/train'
test_path = 'E:/OneDrive/Documents/Uni/PG/Thesis/Code/cifar/cifar/test' 
saved_model_path = 'project/saved_model/test1'
board_path = 'E:/OneDrive/Documents/Uni/PG/Thesis/Code/logs/'

#Randomly initalise the difficulty csv
df = sf.init_diffs(train_path)
print('Initalised Difficulty Dataframe')

#Convert dataframe into datasets
test_ds = sf.collect_test_data(test_path).shuffle(1000)
print('Initalising Difficulty CSV')

#Load model
model = Models.AlexNet(10)
print('Model Loaded')

#evaluation metrics
optimizer = keras.optimizers.Adam(learning_rate=0.001)
#loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
loss_fn = sf.elbo_loss
loss_fn_noreduction = sf.elbo_loss_no_reduction

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc_metric = keras.metrics.CategoricalAccuracy()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc_metric = keras.metrics.CategoricalAccuracy()

print('Setup Complete, Starting training')

#Run the epochs
epochs = 5
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
    
    train_ds = sf.collect_train_data('normal',df,var).batch(BATCHES)

    #training step
    for i,batch in enumerate(train_ds):
        if i % 100 == 0:
            print("Batch = "+ str(i),end="\r")
        batch_loss = train_step(batch[0],batch[1])
        #update df
        df =sf.update_diffs_v2(df,batch,batch_loss)
    
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
    
#save the diffs to seperate file
#np.savetxt('imagecounts/'+name,dataused)
model.save(saved_model_path)







