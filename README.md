# Mobilenet v2 Keras port
This code allows to port pretrained imagenet weights from original MobileNet v2 models to a keras model.
You can use this code to convert all the MobileNets from tensorflow to keras, with pretrained weights.

## Usage
- Download a checkpoint from https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
- Set the parameters (width multiplier and input size, e.g. 0.75, 128)
- Run *extract_weights_from_tf_checkpoint.py* to extract the weights from the selected checkpoint
- Run *export_keras_mobilev2.py* to create the h5 model with loaded weights.
- Run *test_keras_mobilev2.py* to load and test the keras model (tensorflow checkpoint no longer required)

### Example
```bash
wget https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.5_160.tgz
tar -xf mobilenet_v2_0.5_160.tgz
rm -rf weights
./extract_weights_from_tf_checkpoint.py mobilenet_v2_0.5_160.ckpt
./export_keras_mobilev2.py 0.5 160

wget https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Grosser_Panda.JPG/800px-Grosser_Panda.JPG
./test_keras_mobilev2.py mobilenet_v2_0.5_160.h5
```

## Credits
- Original MobileNet port to keras: https://github.com/xiaochus/MobileNetV2
- Original paper: [Inverted Residuals and Linear Bottlenecks Mobile Networks for Classification, Detection and Segmentation.](https://arxiv.org/abs/1801.04381)
- Original mechanism for weight conversion: https://github.com/yuyang-huang/keras-inception-resnet-v2

## More notes
Tested on Linux Subsystem for Windows, with Keras 2.2.4, Tensorflow 1.12.0, python 3.5.2.
