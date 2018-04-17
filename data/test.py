# coding=utf-8

# just for testing some functions

from data.readKITTI import get_filelist, get_bboxlist
from data.imAugment import immirror, imresize, imcrop
from data.genBatch import GenBatch, gen_anchors, gen_labels
from data.constants import *
import cv2
import random
import numpy as np

# choose the function that will be tested
TEST_READ_KITTI = True
TEST_DATA_AUGMENT = False
TEST_GENERATE_ANCHOR = False
TEST_GENERATE_LABEL = True
TEST_GENERATE_BATCH = False

IMAGE_DIR = './test_data/image/training/image_2/'
LABEL_DIR = './test_data/label/training/label_2/'
image_list, bbox_list = [], []

if TEST_READ_KITTI is True:
    image_list = get_filelist(IMAGE_DIR, '.png')
    bbox_list = get_bboxlist(LABEL_DIR, image_list)
    if len(image_list) > 0 and len(image_list) == len(bbox_list):
        print('Test: read KITTI successfully.')
    else:
        print('Test: fail to read KITTI successfully!')

if TEST_DATA_AUGMENT is True:
    # source image
    imID = random.randrange(len(image_list))
    img = cv2.imread(IMAGE_DIR + image_list[imID])
    img_ori = cv2.imread(IMAGE_DIR + image_list[imID])
    for i in bbox_list[imID]:
            img_ori = cv2.rectangle(img_ori, (int(i[0]), int(i[1])), (int(i[0] + i[2]), int(i[1] + i[3])),
                                    (0, 0, 255), 2)
    cv2.imshow('Original', img_ori)

    # mirror
    if True:
        img_mirror, bbox_mirror = immirror(img, bbox_list[imID])
        for i in bbox_mirror:
            img_mirror = cv2.rectangle(img_mirror,
                                       (int(i[0]), int(i[1])), (int(i[0] + i[2]), int(i[1] + i[3])),
                                       (0, 0, 255), 2)
        cv2.imshow('Mirror', img_mirror)

    # resize
    if True:
        img_resize, bbox_resize = imresize(img, bbox_list[imID], 300, 300, True)
        for i in bbox_resize:
            img_resize = cv2.rectangle(img_resize, (int(i[0]), int(i[1])), (int(i[0] + i[2]), int(i[1] + i[3])),
                                       (0, 0, 255), 2)
        cv2.imshow('Resize', img_resize)

    # crop
    if True:
        img_crop, bbox_crop = imcrop(img, bbox_list[imID], 200)
        for i in bbox_crop:
            img_crop = cv2.rectangle(img_crop, (int(i[0]), int(i[1])), (int(i[0] + i[2]), int(i[1] + i[3])),
                                     (0, 0, 255), 2)
        cv2.imshow('Crop', img_crop)

    print('Test: data augment successfully.')


if TEST_GENERATE_ANCHOR is True:
    # choose an image
    imID = 6
    img = cv2.imread(IMAGE_DIR + image_list[imID])
    src_img = np.array(img)
    ground_truth = bbox_list[imID]
    height, width = img.shape[:2]
    s_h = np.ones(1) * height / 300
    s_w = np.ones(1) * width / 300
    # resize to 300x300
    img, ground_truth = imresize(img, ground_truth, 300, 300, True)
    all_anchors = gen_anchors()
    for i in range(all_anchors_num):
        bbox = all_anchors[i, :]
        x1 = int(bbox[0] * s_w)
        y1 = int(bbox[1] * s_h)
        x2 = int((bbox[0] + bbox[2]) * s_w)
        y2 = int((bbox[1] + bbox[3]) * s_h)
        img_new = cv2.rectangle(np.array(src_img), (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('Anchors', img_new)
        cv2.waitKey(1)

if TEST_GENERATE_LABEL is True:
    # choose an image
    imID = 8
    img = cv2.imread(IMAGE_DIR + image_list[imID])
    src_img = np.array(img)
    ground_truth = bbox_list[imID]
    height, width = img.shape[:2]
    s_h = np.ones(1) * height / 300
    s_w = np.ones(1) * width / 300

    # resize to 300x300
    img, ground_truth = imresize(img, ground_truth, 300, 300, True)
    all_anchors = gen_anchors()
    cls_label, reg_label = gen_labels(ground_truth, all_anchors, 0.5)

    # Back-Calculate the ground truth from pre-generated labels
    for i in range(len(cls_label)):
        if cls_label[i, 1] == 1:
            offset = reg_label[i, :]
            anchor = all_anchors[i, :]
            x1 = int((offset[0] * anchor[2] + anchor[0]) * s_w)
            y1 = int((offset[1] * anchor[3] + anchor[1]) * s_h)
            x2 = int(np.exp(offset[2]) * anchor[2] * s_w) + x1
            y2 = int(np.exp(offset[3]) * anchor[3] * s_h) + y1
            src_img = cv2.rectangle(src_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Back-Calculate', src_img)
    cv2.waitKey(0)


if TEST_GENERATE_BATCH is True:
    # build a BATCH generator, batch_size is 1
    m_genbatch = GenBatch(IMAGE_DIR, LABEL_DIR, 1, 300, 300)
    batch_image, batch_cls_label, batch_reg_label = [], [], []
    for i in range(15):
        batch_image, batch_cls_label, batch_reg_label, batch_cls_mask, batch_reg_mask = m_genbatch.nextbatch(0.5, 0.5)

    # Back-Calculate the ground truth from pre-generated labels
    img = np.array(batch_image[0, :, :, :]).astype(np.uint8)
    cls_label = np.array(batch_cls_label).astype(np.uint8)
    reg_label = np.array(batch_reg_label).astype(np.float32)
    all_anchors = gen_anchors()
    # Back-Calculate the ground truth from pre-generated labels
    for i in range(len(cls_label)):
        if cls_label[i, 1] == 1:
            offset = reg_label[i, :]
            anchor = all_anchors[i, :]
            x1 = int((offset[0] * anchor[2] + anchor[0]))
            y1 = int((offset[1] * anchor[3] + anchor[1]))
            x2 = int(np.exp(offset[2]) * anchor[2]) + x1
            y2 = int(np.exp(offset[3]) * anchor[3]) + y1
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Back-Calculate', img)
    cv2.waitKey(0)

cv2.waitKey(0)
