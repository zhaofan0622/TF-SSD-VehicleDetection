# coding=utf-8

import numpy as np
import tensorflow as tf


# initiate the parameters of vgg_net from a pre-trained model
def load_vgg_weights(sess, vgg_model_path=None):
    if vgg_model_path is None:
        print('Init VGG16 base model with random variables.')
    else:
        weights = np.load(vgg_model_path)
        keys = sorted(weights.keys())
        with tf.variable_scope('', reuse=True):
            for key in keys:
                sess.run(tf.get_variable(key).assign(weights[key]))
        print('Init VGG16 base model from pre-trained weights.')


# adjust a anchor bounding box with the predicted offsets, [x,y,w,h]
def adjust_box_offsets(anchor_box, pre_offsets, s_w, s_h):
    num = anchor_box.shape[0]
    if num == 0:
        return []
    else:
        result = np.zeros([num, 4])
        for i in range(num):
            result[i, 0] = (pre_offsets[i, 0] * anchor_box[i, 2] + anchor_box[i, 0]) * s_w
            result[i, 1] = (pre_offsets[i, 1] * anchor_box[i, 3] + anchor_box[i, 1]) * s_h
            result[i, 2] = (np.exp(pre_offsets[i, 2]) * anchor_box[i, 2]) * s_w
            result[i, 3] = (np.exp(pre_offsets[i, 3]) * anchor_box[i, 3]) * s_h
        return result


# nms: non-maximum suppression
def nms(boxes, overlap_thr=0.5):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thr)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick]


# post_process, return the bounding boxes of objects
def post_process(all_anchors, cls_scores, reg_offsets, s_w, s_h, score_thr=0.6, nms_thr=0.5):
    # get the batch_size
    batch = cls_scores.shape[0]

    result = []
    for i in range(batch):
        ids = np.nonzero(cls_scores[i, :, 1] > score_thr)
        boxes_anchors = np.reshape(all_anchors[ids, :], [-1, 4])
        boxes_offsets = np.reshape(reg_offsets[i, ids, :], [-1, 4])
        boxes = adjust_box_offsets(boxes_anchors, boxes_offsets, s_w[i], s_h[i])
        boxes = nms(boxes, nms_thr)
        result.append(boxes)

    return result

