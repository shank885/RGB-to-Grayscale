# This script contain the model architecture

# importing libraries
import tensorflow as tf

def autoencoder(inputs):    
    # encoder
    model = tf.layers.conv2d(inputs, 128, 2, activation = tf.nn.relu)
    model = tf.layers.max_pooling2d(model, 2, 2, padding = 'same')

    # decoder
    model = tf.image.resize_nearest_neighbor(model, tf.constant([129, 129]))
    model = tf.layers.conv2d(model, 1, 2, activation = None, name = 'outputOfAuto')
    
    return model

