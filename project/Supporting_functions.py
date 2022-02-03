import tensorflow as tf
import pandas as pd
import os
import numpy as np
import Acquisition_functions as af
import random


def init_diffs(data_path,csv_path,typ='random'): 
    '''Initiate a csv with either random or uniform variables.
    data_path = where the data is to collect file locations
    csv_path = when the output saves the created dataframe
    typ = (random = create random initialisation, equal = all 1s)

    returns the dataframe
    '''
    #count and list the number of file
    f_files = [os.path.join(r,file) for r,d,f in os.walk(data_path) for file in f]
    print('Files collected')

    if typ=='random':
        nums = [random.random() for i in range(len(f_files))]
        print('Randoms Initilised')
    elif typ == 'equal':
        nums = [1.0 for i in range(len(f_files))]
        print('Ones Initialised')

    df = pd.DataFrame(data={"img":f_files, "diff":nums, 'l1':nums, 'l2':nums,'l3':nums})
    print('Initial Dataframe Created')

    #save to file
    df.to_csv(csv_path, sep=',', index=False)
    print('csv saved')
    return df

def decode_img(img,x=244,y=244):
    '''
    Converts an image to between 0 and 1 and resizes to x y size
    ''' #Convert to [0,1] and resize
    img = tf.image.decode_png(img, channels=3) #turn image into a 3d rgb
    img = tf.image.convert_image_dtype(img,tf.float32)#range between [0,1]
    return tf.image.resize(img, [x,y])#resize image (for use with alex net)

@tf.function
def get_label(file_path):
    '''
    This changes the file paths that include the image labels into one hot definitions
    currently only works with cifar data
    returns the label in one hot
    '''
    #THINGS TO ADD
    # - Make this more common eg add input labels as list of stings

    #get the file name
    label = tf.strings.split(file_path, os.sep)[-1]
    label = tf.strings.split(label,'_')[-1]
    label = tf.strings.split(label,'.')[0] #should output somthing like ('frog')
    if label == 'airplane':
        label = tf.constant([1,0,0,0,0,0,0,0,0,0])
    elif label =='automobile':
        label =tf.constant([0,1,0,0,0,0,0,0,0,0])
    elif label == 'bird':
        label =tf.constant([0,0,1,0,0,0,0,0,0,0])
    elif label == 'cat':
        label =tf.constant([0,0,0,1,0,0,0,0,0,0])
    elif label == 'deer':
        label =tf.constant([0,0,0,0,1,0,0,0,0,0])
    elif label == 'dog':
        label =tf.constant([0,0,0,0,0,1,0,0,0,0])
    elif label == 'frog':
        label =tf.constant([0,0,0,0,0,0,1,0,0,0])
    elif label == 'horse':
        label =tf.constant([0,0,0,0,0,0,0,1,0,0])
    elif label == 'ship':
        label =tf.constant([0,0,0,0,0,0,0,0,1,0])
    else:
        label =tf.constant([0,0,0,0,0,0,0,0,0,1])

    return label

def process_path(file_path,diff,include_extras = True):
    '''
    Convert from [filename,difficulty] to [image, label, filename, difficulty]
    or to [image, label] if include extras is false
    file_path and diff is the inner variables and include extras is exteror

    '''
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    if include_extras == True:
        return img, label, file_path, diff
    else:
        return img, label


def collect_train_data(name,df,vars):
    '''
    name = name of acquisition function
    df = the dataframe
    vars = any other variables that the funcitonmay need eg 'epoch'
    '''
    #This has been chnaged significantly make sure it works
    #Use the acquisition funcitons to order dataframe
    df = af.choose_func(name,df,vars)
    print('Finished Resampling')

    train_ds = tf.data.Dataset.from_tensor_slices((df('img').values, df('diff').values))
    print('Finished Creating Dataset')

    # Convert the dataset to [img,label,filename,difficulty]
    train_ds = train_ds.map(lambda x,y :process_path(x,y,True))
    return train_ds

def collect_test_data(path): #Output train and val datasets
    #val dataset
    #CHNAGE THIS TO USE OS INSTEAD OF PATHLIB
    val_images_root = pathlib.Path(path)
    val_ds = tf.data.Dataset.list_files(str(val_images_root/'*'))
    val_ds = val_ds.map(lambda x, y: process_path(x,y,False))
    return val_ds

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