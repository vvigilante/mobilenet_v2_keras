#!/usr/bin/python3
import tensorflow as tf
import os
import sys
import numpy as np
import re

if len(sys.argv) < 2:
    print("Usage: %s <chekpoint_name.ckpt>" % sys.argv[0])
    sys.exit(1)
cfile = sys.argv[1] #'./mobilenet_v2_0.75_128/mobilenet_v2_0.75_128.ckpt'

'''
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file(file_name=cfile,tensor_name='',all_tensors=False)
print("X")
sys.exit(0)
'''


def get_filename(key):
    """Rename tensor name to the corresponding Keras layer weight name.
    # Arguments
        key: tensor name in TF (determined by tf.variable_scope)
    """

    filename = str(key)
    filename = filename.replace('/Conv_1/', '/conv_18/')
    filename = filename.replace('/Conv/', '/conv_0/')
    filename = filename.replace('/expanded_conv/', '/expanded_conv_0/')
    filename = filename.replace('/', '_')
    filename = filename.replace('MobilenetV2_', '')
    filename = filename.replace('BatchNorm', 'batch_normalization')
    filename = filename.replace('Logits_Conv2d_1c_1x1', 'logits')
    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')
    return filename + '.npy'

def extract_tensors_from_checkpoint_file(filename, output_folder='weights'):
    """Extract tensors from a TF checkpoint file.
    # Arguments
        filename: TF checkpoint file
        output_folder: where to save the output numpy array files
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reader = tf.train.NewCheckpointReader(filename)

    for key in reader.get_variable_to_shape_map():
        # not saving the following tensors
        if key == 'global_step':
            continue
        if 'RMSProp' in key or 'ExponentialMovingAverage' in key:
            continue

        # convert tensor name into the corresponding Keras layer weight name and save
        new = get_filename(key)
        path = os.path.join(output_folder, new)
        arr = reader.get_tensor(key)
        np.save(path, arr)
        print("%s -> %s" % (key, new) )

extract_tensors_from_checkpoint_file(cfile)