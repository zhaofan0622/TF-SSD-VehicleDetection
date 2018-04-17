# coding=utf-8

import random
import cv2
import numpy as np

import data.readKITTI as readKITTI
import data.imAugment as imAugment
from data.constants import *
from data.iou import compute_iou


# compute normalized offset between boxG(ground truth) and boxD(default anchor) [x,y,w,h]
def compute_offset(boxG, boxD):
    offset = np.zeros([1, 4])
    # offset_x, offset_dy
    offset[0, :2] = [(boxG[0] - boxD[0]) / boxD[2], (boxG[1] - boxD[1]) / boxD[3]]
    # offset_w, offset_h
    offset[0, 2:] = np.log([boxG[2] / boxD[2], boxG[3] / boxD[3]])
    return offset


# pre-process: generate all anchors [x,y,w,h]
def gen_anchors():
    all_anchors = np.zeros([all_anchors_num, 4])
    count = 0
    for l in range(6):
        for h in range(feature_size[l]):
            for w in range(feature_size[l]):
                # compute the center of the present anchor
                c_x = (float(w) + 0.5) * anchor_steps[l]
                c_y = (float(h) + 0.5) * anchor_steps[l]
                for a in range(anchors_num[l]):
                    w_base = h_base = anchors_size[l][1]
                    # specially, this anchor has only one kind of ratio
                    if a == 0:
                        w_base = h_base = anchors_size[l][0]
                    # adjust the ratio
                    w_base *= np.sqrt(anchors_ratio[l][a])
                    h_base /= np.sqrt(anchors_ratio[l][a])

                    all_anchors[count, :] = [c_x - w_base/2.0, c_y - h_base/2.0, w_base, h_base]
                    count += 1
    return all_anchors


# generate ground truth labels from source labeled bounding boxes
# each image will generate a classification label map with the shape 8732*2 tf.float32
# each image will generate a regression label map with the shape 8732*4 [dx,dy,dw,dh] tf.float32
# notice: the boxes corresponds to the objects on a 300x300 sized image
def gen_labels(boxes, all_anchors, iou_thr=0.5, print_no_match=False):
    boxes = np.asarray(boxes)

    cls_label = np.concatenate((np.ones([all_anchors_num, 1]), np.zeros([all_anchors_num, 1])), 1)
    reg_label = np.zeros([all_anchors_num, 4])

    iou_temp = np.zeros([all_anchors_num, 1])
    for i in range(boxes.shape[0]):
        match_ok = False
        for j in range(all_anchors_num):
            iou_now = compute_iou(boxes[i, :], all_anchors[j, :])
            if iou_now > iou_thr:
                match_ok = True
                cls_label[j, :] = [0., 1.]
                if iou_now > iou_temp[j]:
                    iou_temp[j] = iou_now
                    reg_label[j, :] = compute_offset(boxes[i, :], all_anchors[j, :])
        if match_ok is False and print_no_match is True:
            print('Boring, no matches')
            print('The lost ground truth is :', boxes[i, :])
    return cls_label, reg_label


# generate two masks to weights different parts in the final ssd loss
def gen_masks(cls_label, neg_weight=3.0, reg_weight=1.0):
    pos_mask = cls_label[:, 1]
    neg_mask = 1. - pos_mask
    pos_num = np.sum(pos_mask)
    neg_num = np.sum(neg_mask)

    if pos_num > 0:
        pos_mask = pos_mask / pos_num
    if neg_num > 0:
        neg_mask = neg_mask / neg_num * neg_weight

    return pos_mask + neg_mask, pos_mask * reg_weight


class GenBatch:
    def __init__(self, image_path, label_path,
                 batch_size, new_w, new_h, is_color=True, is_shuffle=True):
        self.image_path, self.label_path = image_path, label_path,
        self.batch_size, self.new_w, self.new_h, self.is_color, self.is_shuffle = \
            batch_size, new_w, new_h, is_color, is_shuffle

        self.readPos = 0

        # read KITTI
        self.image_list = readKITTI.get_filelist(image_path, '.png')
        self.bbox_list = readKITTI.get_bboxlist(label_path, self.image_list)
        if len(self.image_list) > 0 and len(self.image_list) == len(self.bbox_list):
            print("The amount of images is %d" % (len(self.image_list)))

            self.initOK = True
            self.all_anchors = gen_anchors()

            # init the outputs
            self.batch_image = np.zeros([batch_size, new_h, new_w, 3 if self.is_color else 1], dtype=np.float32)
            self.batch_cls_label = np.zeros([batch_size * all_anchors_num, 2], dtype=np.float32)
            self.batch_reg_label = np.zeros([batch_size * all_anchors_num, 4], dtype=np.float32)
            self.batch_cls_mask = np.zeros([batch_size * all_anchors_num], dtype=np.float32)
            self.batch_reg_mask = np.zeros([batch_size * all_anchors_num], dtype=np.float32)
        else:
            print("The amount of images is %d, while the amount of "
                  "corresponding label is %d" % (len(self.image_list), len(self.bbox_list)))
            self.initOK = False

    # generate a new batch
    # mirror_ratio and crop_ratio are used to control the image augmentation,
    # the default zeros means no images augmentation
    # cls_pos_weight and reg_weight are used to generate a mask to compute the final SSD loss
    def nextbatch(self, mirror_ratio=0.0, crop_ratio=0.0):
        if self.initOK is False:
            print("NO successful initiation!.")
            return []
        for i in range(self.batch_size):
            # if a epoch is completed
            if self.readPos >= len(self.image_list)-1:
                self.readPos = 0
                if self.is_shuffle is True:
                    r_seed = random.random()
                    random.seed(r_seed)
                    random.shuffle(self.image_list)
                    random.seed(r_seed)
                    random.shuffle(self.bbox_list)
                    print('Shuffle the data successfully.\n')

            img = cv2.imread(self.image_path + self.image_list[self.readPos])

            bbox = self.bbox_list[self.readPos]

            self.readPos += 1

            # randomly crop under a specified probability
            if crop_ratio > 0 and random.random() < crop_ratio:
                img, bbox = imAugment.imcrop(img, bbox, min(self.new_w, self.new_h))
            
            # check the input image's size and color
            img, bbox = imAugment.imresize(img, bbox, self.new_w, self.new_h, self.is_color)
            
            # horizontally flip the input image under a specified probability
            if mirror_ratio > 0 and random.random() < mirror_ratio:
                img, bbox = imAugment.immirror(img, bbox)

            # generate processed labels
            cls_label, reg_label = gen_labels(bbox, self.all_anchors)

            # generate masks
            cls_mask, reg_mask = gen_masks(cls_label)

            self.batch_image[i, :, :, :] = img.astype(np.float32)
            self.batch_cls_label[i*all_anchors_num:(i+1)*all_anchors_num, :] = cls_label
            self.batch_reg_label[i*all_anchors_num:(i+1)*all_anchors_num, :] = reg_label
            self.batch_cls_mask[i*all_anchors_num:(i+1)*all_anchors_num] = cls_mask
            self.batch_reg_mask[i*all_anchors_num:(i+1)*all_anchors_num] = reg_mask

        return self.batch_image, self.batch_cls_label, self.batch_reg_label, self.batch_cls_mask, self.batch_reg_mask




