# Mobilenet v2 Keras port
This code allows to port pretrained imagenet weights from original mobilenet v2 models to a keras model.

# Usage
- Download a checkpoint from https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
- Set the parameters (width multiplier and input size, e.g. 0.75, 128)
- Run *extract_weights_from_tf_checkpoint.py* to extract the weights from the selected checkpoint
- Run *export_keras_mobilev2.py* to create the h5 model with loaded weights.
- Run *test_keras_mobilev2.py* to load and test the keras model (tensorflow checkpoint no longer required)

# Credits
- Original mobilenet port to keras: https://github.com/xiaochus/MobileNetV2
- Original paper: [Inverted Residuals and Linear Bottlenecks Mobile Networks for Classification, Detection and Segmentation.](https://arxiv.org/abs/1801.04381)
- Original mechanism for weight conversion: https://github.com/yuyang-huang/keras-inception-resnet-v2
