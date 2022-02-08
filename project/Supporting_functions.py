import tensorflow as tf
import pandas as pd
import os
import numpy as np
import Acquisition_functions as af
import random


def init_diffs(labels_path,typ='random'): 
    '''Initiate a csv with either random or uniform variables.
    data_path = where the data is to collect file locations
    csv_path = when the output saves the created dataframe
    typ = (random = create random initialisation, equal = all 1s)

    returns the dataframe [id, label, diff, l1, l2, l3]
    '''
    #Open csv file with data labels
    df = pd.read_csv(labels_path)

    if typ=='random':
        nums = [random.random() for i in range(len(df.index))]
        print('Randoms Initilised')
    elif typ == 'equal':
        nums = [1.0 for i in range(len(df.index))]
        print('Ones Initialised')

    df['diff'] = nums
    df['l1'] = nums
    df['l2'] = nums
    df['l3'] = nums
    print('Initial Dataframe Created')
    
    return df

def decode_img(img,x=244,y=244):
    '''
    Converts an image to between 0 and 1 and resizes to x y size
    ''' #Convert to [0,1] and resize
    img = tf.image.decode_png(img, channels=3) #turn image into a 3d rgb
    img = tf.image.convert_image_dtype(img,tf.float32)#range between [0,1]
    return tf.image.resize(img, [x,y])#resize image (for use with alex net)

def get_label(label,labels):
    '''
    labels is a list of strings
    This changes the file paths that include the image labels into one hot definitions
    returns the label in one hot
    '''
    out_label = [0] * len(labels)
    for i in range(len(labels)):
        #print(label.numpy().decode())
        #print(labels[i].numpy().decode())
        if label.numpy().decode() == labels[i].numpy().decode():
            out_label[i] = 1

    return out_label

def process_path(id,label,diff,root,class_names):
    '''
    Convert from [filename,difficulty] to [image, label, filename, difficulty]
    or to [image, label] if include extras is false
    file_path and diff is the inner variables and include extras is exteror

    '''
    label = get_label(label,class_names)
    img = tf.io.read_file(root + str(id.numpy()) +'.png')
    img = decode_img(img)

    return img, label, id, diff

def collect_train_data(name,df,vars,root):
    '''
    name = name of acquisition function
    df = the dataframe
    vars = any other variables that the funcitonmay need eg 'epoch'
    '''
    #This has been chnaged significantly make sure it works
    #Use the acquisition funcitons to order dataframe
    df = af.choose_func(name,df,vars)
    print('Finished Resampling')

    #dataset with [id , label, diff]
    train_ds = tf.data.Dataset.from_tensor_slices((df['id'].values, df['label'].values,df['diff'].values))
    print('Finished Creating Train Dataset')

    class_names = vars.class_names

    # Convert the dataset to [img,label,id,difficulty]
    train_ds = train_ds.map(lambda x,y,z :tf.py_function(func=process_path,inp=[x,y,z,root,class_names],Tout=[tf.float32,tf.int32,tf.int64,tf.float64]))
    return train_ds

def split_train_test(df,test_percentage):
    test_df = df.sample(frac=test_percentage)
    train_df = df.drop(test_df.index)
    return train_df, test_df

def collect_test_data(df,root,vars): 
    #dataset with [id , label, diff]
    test_ds = tf.data.Dataset.from_tensor_slices((df['id'].values, df['label'].values,df['diff'].values))
    test_ds = test_ds.map(lambda x,y,z :tf.py_function(func=process_path,inp=[x,y,z,root,vars.class_names],Tout=[tf.float32,tf.int32,tf.int64,tf.float64]))
    print('Finished Creating Test Dataset')
    #output it [img,label,id,(diff)?]
    return test_ds

def save_diffs(df,path): #Save the difficult dataframe to the csv
    df.to_csv(path, sep=',',index=True)
    return

def update_diffs(df,batch,predictions): #batch update diff metric in the pandas df
    #Updates based on the probabilty of correct predicition
    #Convert batch diffs to list
    #NEED TO ENSURE THIS IS DOING THE RIGHT THING HERE!!!
    batch_size = tf.shape(batch[0]).numpy()
    batch_size = batch_size[0]
    diff =[0] * batch_size
    for i in range(batch_size):
        cat = batch[1][i].numpy().tolist()
        cat = cat.index(1)
        d = predictions[i,cat].numpy() 
        diff[i] =abs(d)

    df = df.set_index('img')

    #Update the difficulties in the pandas dataframe
    for i,f in enumerate(batch[2]):
        #need to convert to a file location
        f_str = f.numpy().decode('utf-8') #opens in bytes mode b'...' so convert it
        #if the batch string is in the pandas file update the difficulty
        if f_str in df.index.values:
            df.loc[f_str,'diff'] = diff[i]
        else:
            print('error: file not found in df')

    df = df.reset_index()
    return df

def update_diffs_v2(df,batch,losses):
    #Use loss result instead of predicted probability to update the df
    #
    #batch_size = batch_size[0]
    diff = losses.numpy() #array of individual img losses
    df = df.set_index('img')
    #updates the df values 
    for i, f in enumerate(batch[2]):
        f_str = f.numpy().decode('utf-8') #img file path name
        if f_str in df.index.values:
            df.loc[f_str,'diff'] = diff[i]
        else:
            print('error: file not found in df')
    df = df.reset_index()
    #print(df)
    #print(df.describe())
    return df

def update_diffs_v3(train_df,batch,losses):
    #train_df = [id,label,diff,l1,l2,l3]
    #batch = [img,label,id,diff] * batchsize
    #losses = [losses] *batchsize

    losses = losses.numpy()
    for i,id  in enumerate(batch[2]):
        train_df.loc[train_df['id']==id.numpy(),'diff'] =losses[i]

    return train_df

def new_diff_col(df):

    #move columns across to store in bank
    df['l3'] = df['l2']
    df['l2'] = df['l1']
    df['l1'] = df['diff']
    df['diff'] = df[['l3','l2','l1']].mean(axis=1)
    
    print(df['diff'].head())
    print(df['l1'].head())
    print(df['l2'].head())
    print(df['l3'].head())
    return df

@tf.function
def elbo_loss(labels, logits):
    loss_en = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    loss_kl = tf.keras.losses.KLD(labels, logits)
    loss = tf.reduce_mean(tf.add(loss_en, loss_kl))
    return loss

@tf.function
def elbo_loss_no_reduction(labels, logits):
    loss_en = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    loss_kl = tf.keras.losses.KLD(labels, logits)
    loss_noreduction = tf.add(loss_en, loss_kl)
    return loss_noreduction

def sum_correct_labels(preds,labels):
    #simply sums up the correct pairs to average later
    acc = 0
    for p,l in zip(preds,labels):
        if np.argmax(p) == np.argmax(l):
            acc += 1
    return acc













