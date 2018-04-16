# coding=utf-8

import signal
import sys
import tensorflow as tf
import time
from data.genBatch import GenBatch
from net.ssd_net import ssd, ssd_loss_new
from util import load_vgg_weights
from data.constants import all_anchors_num


flags = tf.app.flags
FLAGS = flags.FLAGS

# configurations
flags.DEFINE_float(
    'weight_decay', 0.0005,
    'The weight decay on the model weights.')
flags.DEFINE_float(
    'learning_rate', 0.01,
    'Initial learning rate.')
flags.DEFINE_float(
    'lr_decay', 0.5,
    'The decay rate of learning rate.')
flags.DEFINE_integer(
    'lr_decay_steps', 1000,
    'The steps of decaying learning rate with a base lr_decay:.')
flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum value used in SGD with momentum.')
flags.DEFINE_integer(
    'max_iter', 10000,
    'The maximum iterative batches for training.')
flags.DEFINE_integer(
    'display_interval', 20,
    'The number of interval batches for showing present training results.')
flags.DEFINE_integer(
    'batch_size', 32,
    'The number of samples in each batch.')
flags.DEFINE_string(
    'checkpoint_path', './model/',
    'The path of saving a checkpoint.')
flags.DEFINE_integer(
    'checkpoint_interval', 1000,
    'The number of interval batches for saving a checkpoint.')
flags.DEFINE_string(
    'image_path', '/media/data1/zhaofan/KITTI/images/training/image_2/',
    'The path of KITTI images.')
flags.DEFINE_string(
    'label_path', '/media/data1/zhaofan/KITTI/labels/training/label_2/',
    'The path of KITTI labels.')
flags.DEFINE_string(
    'log_path', './log/', 'The path to save logs.')
flags.DEFINE_string(
    'gpu_list', '1', 'The list of gpus that will be used')
flags.DEFINE_string(
    'vgg_model', './net/vgg_base.npz',
    'the path where the pre-trained vgg16 model located')
flags.DEFINE_string(
    'ssd_model', None,
    'the path where the pre-trained ssd object detection model located')


def main(_):
    # allow automatically assigned the devices
    config = tf.ConfigProto(allow_soft_placement=True)
    # assign the used gpu
    config.gpu_options.visible_device_list = FLAGS.gpu_list
    sess = tf.Session(config=config)

    # build a BATCH generator
    m_genbatch = GenBatch(FLAGS.image_path, FLAGS.label_path, FLAGS.batch_size, 300, 300)

    # inputs
    images = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 300, 300, 3], name='inputs')
    cls_label = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * all_anchors_num, 2], name='cls_label')
    reg_label = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * all_anchors_num, 4], name='reg_label')
    cls_mask = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * all_anchors_num], name='cls_mask')
    reg_mask = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * all_anchors_num], name='reg_mask')

    # outputs and losses
    cls_predict, reg_predict = ssd(images, True)

    mean_cls_loss, mean_reg_loss, mean_all_loss = ssd_loss_new(cls_predict, reg_predict,
                                                               cls_label, reg_label, cls_mask, reg_mask)
    # classification accuracy
    # correct_prediction = tf.equal(tf.argmax(tf.reshape(cls_predict, [-1, 2]), 1), tf.argmax(cls_label, 1))
    # correct_prediction = tf.cast(correct_prediction, tf.float32)
    # accuracy = tf.reduce_mean(correct_prediction)

    # configure the optimizer
    with tf.name_scope('optimizer'):
        global_step = tf.Variable(0, trainable=False)
        # decayed_learning_rate = learning_rate * lr_decay ^ (global_step / decay_steps)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                                   FLAGS.lr_decay_steps, FLAGS.lr_decay, staircase=True)
        m_optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum,
                                                 name='Momentum').minimize(mean_all_loss, global_step)

    # handle interrupt signal
    m_saver = tf.train.Saver()

    if FLAGS.ssd_model is None:
        sess.run(tf.global_variables_initializer())
        load_vgg_weights(sess, FLAGS.vgg_model)
    else:
        print('Load ssd model weights from %s' % FLAGS.ssd_model)
        m_saver.restore(sess, FLAGS.ssd_model)

    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        print('The saved model is %sckpt' % FLAGS.checkpoint_path)
        m_saver.save(sess, "%s/ckpt" % FLAGS.checkpoint_path, global_step)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('cls_loss', mean_cls_loss)
    tf.summary.scalar('reg_loss', mean_reg_loss)
    tf.summary.scalar('loss', mean_all_loss)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)

    t_start = time.time()
    # iteratively train
    while True:
        batch_image, batch_cls_label, batch_reg_label, batch_cls_mask, batch_reg_mask = m_genbatch.nextbatch(0.5, 0.5)

        _, batch_cls_loss, batch_reg_loss, batch_all_loss, step, summary, lr = sess.run(
            [m_optimizer, mean_cls_loss, mean_reg_loss, mean_all_loss, global_step, merged, learning_rate],
            feed_dict={images: batch_image, cls_label: batch_cls_label, reg_label: batch_reg_label,
                       cls_mask: batch_cls_mask, reg_mask: batch_reg_mask})

        # print and record a log
        if (step % FLAGS.display_interval) == 0:
            t_end = time.time()
            print('step %d, cls_loss is %f, reg_loss is %f, sum_loss is %f, learning rate is %f.' %
                  (step, batch_cls_loss, batch_reg_loss, batch_all_loss, lr))
            print('time consuming: %.0fs\n' % (t_end - t_start))
            t_start = time.time()
            train_writer.add_summary(summary, step)

        # save a snapshot
        if (step % FLAGS.checkpoint_interval) == 0:
            print('Saving a snapshot to %s ckpt' % FLAGS.checkpoint_path)
            m_saver.save(sess, "%s/ckpt" % FLAGS.checkpoint_path, global_step)

        if step >= FLAGS.max_iter:
            break


if __name__ == "__main__":
    tf.app.run(main=main)
