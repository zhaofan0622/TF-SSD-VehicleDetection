# coding=utf-8
import os
import sys


# get the file list with the specified suffix
def get_filelist(path, ext):
    # get the file list in a specified folder
    filelist_temp = os.listdir(path)
    filelist = []
    # compare the suffix
    for i in filelist_temp:
        if os.path.splitext(i)[1] == ext:
            filelist.append(os.path.splitext(i)[0] + ext)
    if len(filelist) == 0:
        print('Empty image list. The path is %s, the suffix is %s' % (path, ext))
        sys.exit(0)
    return filelist


# read the labeled bounding box: nx4, [x,y,w,h]
def get_bbox(filename):
    bbox = []
    # if exist
    if os.path.exists(filename):
        with open(filename) as fi:
            label_data = fi.readlines()
        # read a new line
        for l in label_data:
            data = l.split()
            # if existing a car
            if data[0] in ['Van', 'Car', 'Truck']:
                bbox.append([float(data[4]), float(data[5]),
                             float(data[6])-float(data[4]), float(data[7])-float(data[5])])
    else:
        print('Error: failed to read the labels, no such file %s' % filename)
        sys.exit(0)
    return bbox


# run get_bbox for a batch files
def get_bboxlist(rootpath, imagelist):
    bboxlist = []
    for i in imagelist:
        bboxlist.append(get_bbox(rootpath + i.split('.')[0] + '.txt'))
    return bboxlist
        

