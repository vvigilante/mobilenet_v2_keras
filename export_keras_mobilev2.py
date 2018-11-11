#!/usr/bin/python3

import sys

if len(sys.argv) < 3:
    print("Usage: %s <WIDTH_MULTIPLIER> <INPUT_SIZE>" % sys.argv[0])
    sys.exit(1)



import numpy as np
import os
from mobilenet_v2_keras import MobileNetv2


MULT= float(sys.argv[1]) #0.75
SIZE= int(sys.argv[2]) #128
NCLASS=1001
WEIGHTS_DIR = './weights'
KERAS_OUT = 'mobilenet_v2_%s_%d.h5' % (str(MULT), SIZE)

print("Creating model Mobile V2 %s %d..." % (str(MULT), SIZE))
model = MobileNetv2((SIZE, SIZE, 3), NCLASS, MULT)

print("Copying weights from %s..." % WEIGHTS_DIR)
for l in model.layers:
    print(l.name)
    weights = []
    for w in l.weights:
        print('\t'+w.name)
        weight_name = os.path.basename(w.name).replace(':0', '')
        weight_file = l.name + '_' + weight_name + '.npy'
        weight_arr = np.load(os.path.join(WEIGHTS_DIR, weight_file))
        weights.append(weight_arr)
    l.set_weights(weights)


print("Saving model %s..." % KERAS_OUT)
model.save(KERAS_OUT)
print("Ok.")
