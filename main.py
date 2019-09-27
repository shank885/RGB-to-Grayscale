# RGB to Grayscale Conversion using neural network

# import libraries
import tensorflow as tf
import numpy as np
import cv2
import glob as gl
from utils import model


# find total number of images
path_to_source_images = 'data/color_images/*.jpg'
filenames = gl.glob(path_to_source_images)

num_images = len(filenames)

# loading the input images
dataset = []
for i in range(1, num_images+1):
    img = cv2.imread("data/color_images/color_" +str(i) +".jpg" )
    dataset.append(np.array(img))

dataset_source = np.asarray(dataset)

# loading the output images
dataset_tar = []
for i in range(1, num_images+1):
    img = cv2.imread("data/gray_images/gray_" +str(i) +".jpg", 0)    
    dataset_tar.append(np.array(img))

dataset_target = np.asarray(dataset_tar)
dataset_target = dataset_target[:, :, :, np.newaxis]


ae_inputs = tf.placeholder(tf.float32, (None, 128, 128, 3), name = 'inputToAuto')
ae_target = tf.placeholder(tf.float32, (None, 128, 128, 1))
ae_outputs = model.autoencoder(ae_inputs)

# set training parameters
lr = 0.001
batch_size = 32
epoch_num = 50

# define optimizer and loss function
loss = tf.reduce_mean(tf.square(ae_outputs - ae_target))
train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)

# Intialize the network 
init = tf.global_variables_initializer()
saving_path = 'model/ColorToGray.ckpt'

saver_ = tf.train.Saver(max_to_keep = 3)

batch_img = dataset_source[0:batch_size]
batch_out = dataset_target[0:batch_size]

num_batches = num_images//batch_size

sess = tf.Session()
sess.run(init)


for ep in range(epoch_num):
    batch_size = 0
    for batch_n in range(num_batches):

        _, c = sess.run([train_op, loss], feed_dict = {ae_inputs: batch_img, ae_target: batch_out})
        print("Epoch: {} - cost = {:.5f}" .format((ep+1), c))
            
        batch_img = dataset_source[batch_size: batch_size+32]
        batch_out = dataset_target[batch_size: batch_size+32]
            
        batch_size += 32
    
    saver_.save(sess, saving_path, global_step = ep)

recon_img = sess.run([ae_outputs], feed_dict = {ae_inputs: batch_img})

sess.close()

saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# loading the saved model
saver.restore(sess, 'model/ColorToGray.ckpt-49')

# loading the testing images
filenames = gl.glob('data/input_images/*.jpg')

test_data = []
for file in filenames[0:100]:
    test_data.append(np.array(cv2.imread(file)))

test_dataset = np.asarray(test_data)

batch_imgs = test_dataset
gray_imgs = sess.run(ae_outputs, feed_dict = {ae_inputs: batch_imgs})

# saving the images in the output_images folder
for i in range(gray_imgs.shape[0]):
    cv2.imwrite('data/output_images/' +str(i) +'.jpg', gray_imgs[i])

