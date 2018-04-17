# coding=utf-8

""" vgg_base.py:
    To build the basic VGG16 part of the SSD net.
    The input shape is n*300*300*3
"""

import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


def vgg_net(images):
    with tf.name_scope('VGG_base'):

        # zero-mean input
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        inputs = images - mean

        # build the vgg16 net
        # block1, the output is 150*150*64
        conv1_1 = conv_layer(inputs, 64, 'conv1_1')
        conv1_2 = conv_layer(conv1_1, 64, 'conv1_2')
        pool1 = max_pooling_layer(conv1_2, 'pool1')

        # block2, the output is 75*75*128
        conv2_1 = conv_layer(pool1, 128, 'conv2_1')
        conv2_2 = conv_layer(conv2_1, 128, 'conv2_2')
        pool2 = max_pooling_layer(conv2_2, 'pool2')

        # block3, the output is 38*38*256
        conv3_1 = conv_layer(pool2, 256, 'conv3_1')
        conv3_2 = conv_layer(conv3_1, 256, 'conv3_2')
        conv3_3 = conv_layer(conv3_2, 256, 'conv3_3')
        pool3 = max_pooling_layer(conv3_3, 'pool3')

        # block4, the output is 19*19*512
        conv4_1 = conv_layer(pool3,512, 'conv4_1')
        conv4_2 = conv_layer(conv4_1, 512, 'conv4_2')
        conv4_3 = conv_layer(conv4_2, 512, 'conv4_3')
        pool4 = max_pooling_layer(conv4_3, 'pool4')

        # block5,no pooling, the output is 19*19*512
        conv5_1 = conv_layer(pool4, 512, 'conv5_1')
        conv5_2 = conv_layer(conv5_1, 512, 'conv5_2')
        conv5_3 = conv_layer(conv5_2, 512, 'conv5_3')

        return conv4_3, conv5_3


# convolution layer, the default kernel size is 3, the default stride is 1
def conv_layer(inputs, num_output, name=None):
    # get the number of channels of the inputs
    num_input = inputs.get_shape()[-1].value
    with tf.name_scope(name):
        weights = tf.get_variable(name + '_w',
                                  shape=[3, 3, num_input, num_output],
                                  dtype=tf.float32,
                                  initializer=tf.glorot_normal_initializer())

        conv = tf.nn.conv2d(inputs, weights, [1, 1, 1, 1], padding='SAME')

        bias = tf.get_variable(name + '_b',
                               shape=[num_output],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0))
        conv_b = tf.nn.bias_add(conv, bias)
        conv_b = tf.nn.relu(conv_b)
        return conv_b


# max-pooling layer, the default kernel size is 2, the default stride is 2
def max_pooling_layer(inputs, name):
    return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
