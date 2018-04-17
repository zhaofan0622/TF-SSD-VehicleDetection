# coding=utf-8


# compute the IOU between two bounding boxes，[x,y,w,h]
def compute_iou(bbox1, bbox2):
    # pre-process to avoid invalid bounding boxes
    bbox1 = [bbox1[0], bbox1[1], max(bbox1[2], 0.01), max(bbox1[3], 0.01)]
    bbox2 = [bbox2[0], bbox2[1], max(bbox2[2], 0.01), max(bbox2[3], 0.01)]

    # compute the intersection area
    left = max(bbox1[0], bbox2[0])
    right = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    bottom = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    top = max(bbox1[1], bbox2[1])
    intersection = 0
    if left < right and top < bottom:
        intersection = (right - left) * (bottom - top)

    # compute the union area
    union = (bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - intersection)

    # return the iou value
    return intersection / union
