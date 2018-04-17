# coding=utf-8

import sys
import numpy as np
import tensorflow as tf
import time
import cv2
from net.ssd_net import ssd
from data.genBatch import gen_anchors
from util import post_process


flags = tf.app.flags
FLAGS = flags.FLAGS

# configurations
flags.DEFINE_string(
    'image_path', './test_image/000001.png',
    'The path of a single image for testing.')
flags.DEFINE_string(
    'gpu_list', '0', 'The list of gpus that will be used')
flags.DEFINE_string(
    'ssd_model', './model/ckpt-10000',
    'the path where the pre-trained ssd object detection model located')


def main(_):
    # allow automatically assigned the devices
    config = tf.ConfigProto(allow_soft_placement=True)
    # assign the used gpu
    config.gpu_options.visible_device_list = FLAGS.gpu_list
    sess = tf.Session(config=config)

    # generate all default anchors
    all_anchors = gen_anchors()

    # read an image
    ssd_input = tf.placeholder(tf.float32, shape=[None, 300, 300, 3], name='input')
    img = cv2.imread(FLAGS.image_path)
    img_det = img
    if img is None:
        print('Error: failed to read the labels, no such file %s' % FLAGS.image_path)
        sys.exit(0)
    height, width = img.shape[:2]
    s_h = np.ones(1) * height / 300
    s_w = np.ones(1) * width / 300
    img = cv2.resize(img, (300, 300))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = np.reshape(img, [1, 300, 300, 3])
    # outputs, set bn_training as False
    cls_predict, reg_predict = ssd(ssd_input, False)
    cls_predict = tf.nn.softmax(cls_predict)

    # handle interrupt signal
    m_saver = tf.train.Saver()

    if FLAGS.ssd_model is None:
        print('Error: no valid ssd model!')
        sys.exit(0)
    else:
        print('Load ssd model weights from %s' % FLAGS.ssd_model)
        m_saver.restore(sess, FLAGS.ssd_model)

    t_start = time.time()
    # test
    cls_scores, reg_offsets = sess.run([cls_predict, reg_predict], feed_dict={ssd_input: img})

    det_boxes = post_process(all_anchors, cls_scores, reg_offsets, s_w, s_h, score_thr=0.9, nms_thr=0.6)
    t_end = time.time()
    print('time consuming: %.0fs\n' % (t_end - t_start))

    # show results
    for i in det_boxes[0]:
        img_det = cv2.rectangle(img_det, (int(i[0]), int(i[1])), (int(i[0] + i[2]), int(i[1] + i[3])),
                                (0, 0, 255), 2)
    cv2.imshow('Results', img_det)
    cv2.waitKey(0)




if __name__ == "__main__":
    tf.app.run(main=main)
