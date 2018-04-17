# coding=utf-8
import cv2
import random


# resize input image to specified size and check the color, [x,y,w,h]
def imresize(in_img, in_bbox, out_w, out_h, is_color=True):
    # if the input is the path of an image
    if isinstance(in_img, str):
        in_img = cv2.imread(in_img)

    # get the height and width
    height, width = in_img.shape[:2]
    out_img = cv2.resize(in_img, (out_w, out_h))

    # check and adjust the color
    if is_color is True and in_img.ndim == 2:
        out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
    elif is_color is False and in_img.ndim == 3:
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)

    # re-adjusting the bounding box
    s_h = out_h / height
    s_w = out_w / width
    out_bbox = []
    for i in in_bbox:
        out_bbox.append((i[0]*s_w, i[1]*s_h, i[2]*s_w, i[3]*s_h))
    return out_img, out_bbox


# flip the input image horizontally, [x,y,w,h]
def immirror(in_img, in_bbox):
    # if the input is the path of an image
    if isinstance(in_img, str):
        in_img = cv2.imread(in_img)

    # flip the input image horizontally
    out_img = cv2.flip(in_img, 1)

    # get the height and width
    width = out_img.shape[1]

    # re-adjusting the bounding box
    out_bbox = []
    for i in in_bbox:
        out_bbox.append((width - i[0] - i[2], i[1], i[2], i[3]))
    return out_img, out_bbox


# randomly crop the input image, min_wh is the minimum size, [x,y,w,h]
def imcrop(in_img, in_bbox, min_hw):
    # if the input is the path of an image
    if isinstance(in_img, str):
        in_img = cv2.imread(in_img)

    # get the height and width
    height, width = in_img.shape[:2]

    # if the image is too small, give up cropping
    if height <= min_hw and width <= min_hw:
        return in_img, in_bbox

    # to guarantee the correctness of bounding boxes, the crop should include all objects
    # now, find the minimum rectangle which include all objects
    min_x1, min_y1, min_x2, min_y2 = width-1, height-1, 0, 0
    for i in in_bbox:
        min_x1 = min(min_x1, int(i[0]))
        min_y1 = min(min_y1, int(i[1]))
        min_x2 = max(min_x2, int(i[0] + i[2]))
        min_y2 = max(min_y2, int(i[1] + i[3]))

    # base on the minimum rectangle, randomly crop a region including all objects
    if min_x1 <= 1:
        rand_x1 = 0
    else:
        rand_x1 = random.randint(0, min(min_x1, max(width - min_hw, 1)))
    if min_y1 <= 1:
        rand_y1 = 0
    else:
        rand_y1 = random.randint(0, min(min_y1, max(height - min_hw, 1)))
    if min_x2 >= width or rand_x1 + min_hw >= width:
        rand_x2 = width
    else:
        rand_x2 = random.randint(max(rand_x1+min_hw, min_x2), width)
    if min_y2 >= height or rand_y1 + min_hw >= height:
        rand_y2 = height
    else:
        rand_y2 = random.randint(max(rand_y1+min_hw, min_y2), height)
    
    # crop the region
    out_img = in_img[rand_y1:rand_y2-1, rand_x1:rand_x2-1]

    # re-adjusting the bounding box
    out_bbox = []
    for i in in_bbox:
        out_bbox.append((i[0]-rand_x1, i[1]-rand_y1, i[2], i[3]))

    return out_img, out_bbox
