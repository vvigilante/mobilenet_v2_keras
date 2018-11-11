#!/usr/bin/python

import sys

if len(sys.argv) < 2:
    print("Usage: %s <MODEL_NAME.h5>" % sys.argv[0])
    sys.exit(1)

import keras
import numpy as np

from keras import backend as K
def relu6(x):
    return K.relu(x, max_value=6)

MODEL_FILE = sys.argv[1] # "mobilenet_v2_0.75_128.h5"

model = keras.models.load_model(MODEL_FILE, {'relu6':relu6} )

shape = model.input.shape
shape = (1, shape[1], shape[2], shape[3])
#print("Input shape:")
#print(shape)

import PIL
from PIL import Image
from imagenet_labels import labels
pic = Image.open("800px-Grosser_Panda.JPG")
pic = pic.resize(shape[1:3], PIL.Image.ANTIALIAS)
x = np.array(pic.getdata()).reshape(shape) / 256.0

y = model.predict(x)
print("Predicted: %s (confidence %.2f)" % (labels[np.argmax(y)], np.max(y)) )