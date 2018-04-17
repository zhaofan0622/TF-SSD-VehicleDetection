import tensorflow as tf
import numpy as np
from net.ssd_net import ssd

images = tf.get_variable('input', shape=[1, 300, 300, 3], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
cls_predict, reg_predict = ssd(images, True)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run([cls_predict, reg_predict])

train_writer = tf.summary.FileWriter('D:/SSD/log/', sess.graph)
