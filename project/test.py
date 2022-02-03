import tensorflow as tf

tensor = tf.range(5)

ds = tf.data.Dataset.from_tensor_slices(tensor)

for i in ds:
    print(i)

def func (input):
    return input * input

ds1 = ds.map(func)

for i in ds1:
    print(i)

def func2(input,var):
    return input + var

ds2 =ds.map(func2(,4))
