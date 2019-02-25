# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
import cv2


def parse_line(line):
    '''
    Given a line from the training/test txt file, return parsed
    pic_path, boxes info, and label info.
    return:
        pic_path: string.
        boxes: shape [N, 4], N is the ground truth count, elements in the second
            dimension are [x_min, y_min, x_max, y_max]
    '''
    s = line.strip().split(' ')
    pic_path = s[0]
    s = s[1:]
    box_cnt = len(s) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = int(s[i*5]), float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3]), float(s[i*5+4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    return pic_path, boxes, labels


def resize_image_and_correct_boxes(img, boxes, img_size):
    # convert gray scale image to 3-channel fake RGB image
    if len(img) == 2:
        img = np.expand_dims(img, -1)
    ori_height, ori_width = img.shape[:2]
    new_width, new_height = img_size
    # shape to (new_height, new_width)
    img = cv2.resize(img, (new_width, new_height))

    # convert to float
    img = np.asarray(img, np.float32)

    # boxes
    # xmin, xmax
    boxes[:, 0] = boxes[:, 0] / ori_width * new_width
    boxes[:, 2] = boxes[:, 2] / ori_width * new_width
    # ymin, ymax
    boxes[:, 1] = boxes[:, 1] / ori_height * new_height
    boxes[:, 3] = boxes[:, 3] / ori_height * new_height

    return img, boxes


def data_augmentation(img, boxes, label):
    '''
    Do your own data augmentation here.
    param:
        img: a [H, W, 3] shape RGB format image, float32 dtype
        boxes: [N, 4] shape boxes coordinate info, N is the ground truth box number,
            4 elements in the second dimension are [x_min, y_min, x_max, y_max], float32 dtype
        label: [N] shape labels, int64 dtype (you should not convert to int32)
    '''
    return img, boxes, label


def process_box(boxes, labels, img_size, class_num, anchors):
    '''
    :param boxes: [None,4]
    :param labels: [None,]
    :param img_size: 416,416
    :param class_num: [[10, 13], [16, 30], [33, 23],
                         [30, 61], [62, 45], [59,  119],
                         [116, 90], [156, 198], [373,326]]
    :param anchors:
    :return:
    '''
    '''
    Generate the y_true label, i.e. the ground truth feature_maps in 3 different scales.
    '''
    anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]

    # convert boxes form:
    # shape: [N, 2]
    # (x_center, y_center)
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    # (width, height)
    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]

    # [13, 13, 3, 5+num_class]
    y_true_13 = np.zeros((img_size[1] // 32,
                          img_size[0] // 32,
                          3, 5 + class_num),
                         np.float32)
    # [26, 26, 3, 5+num_class]
    y_true_26 = np.zeros((img_size[1] // 16,
                          img_size[0] // 16,
                          3, 5 + class_num), np.float32)
    # [52, 52, 3, 5+num_class]
    y_true_52 = np.zeros((img_size[1] // 8,
                          img_size[0] // 8,
                          3, 5 + class_num), np.float32)

    y_true = [y_true_13, y_true_26, y_true_52]

    '''
        下面是，对于每个gt框而言，计算它和每个anchors之间，宽高之间的距离。
        然后计算IOU

        其实就是之前如何得到anchors里面，计算IOU那一套
    '''
    # [N, 1, 2]
    box_sizes = np.expand_dims(box_sizes, 1)
    # broadcast tricks
    # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
    mins = np.maximum(- box_sizes / 2, - anchors / 2)
    maxs = np.minimum(box_sizes / 2, anchors / 2)
    # [N, 9, 2]
    whs = maxs - mins

    # [N, 9]
    iou = (whs[:, :, 0] * whs[:, :, 1]) / (box_sizes[:, :, 0] * box_sizes[:, :, 1]
                                           + anchors[:, 0] * anchors[:, 1]
                                           - whs[:, :, 0] * whs[:, :, 1]
                                           + 1e-10)
    '''
        看这个gt，和哪个anchors框的IOU最大
    '''
    # [N]
    best_match_idx = np.argmax(iou, axis=1)

    ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
    for i, idx in enumerate(best_match_idx):
        '''
            找到了自己属于你哪个anchors的group，已经自己应该被哪个尺度进行预测
        '''
        # idx: 0,1,2 ==> 2; 3,4,5 ==> 1; 6,7,8 ==> 2
        feature_map_group = 2 - idx // 3
        # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
        ratio = ratio_dict[np.ceil((idx + 1) / 3.)]

        '''
            将中心点直接缩放到对应尺度上
        '''
        x = int(np.floor(box_centers[i, 0] / ratio))
        y = int(np.floor(box_centers[i, 1] / ratio))
        # k表示这个gt属于第几个anchors_group中的第几个anchors
        k = anchors_mask[feature_map_group].index(idx)
        c = labels[i]
        # print feature_map_group, '|', y,x,k,c

        y_true[feature_map_group][y, x, k, :2] = box_centers[i] # x,y
        y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i] # w,h
        y_true[feature_map_group][y, x, k, 4] = 1. # confidence
        y_true[feature_map_group][y, x, k, 5+c] = 1. # one_label

    return y_true_13, y_true_26, y_true_52


def parse_data(line, class_num, img_size, anchors, mode):
    '''
    param:
        line: a line from the training/test txt file
        args: args returned from the main program
        mode: 'train' or 'val'. When set to 'train', data_augmentation will be applied.
    '''
    '''
        对于train.txt/valid.txt/test.txt中的每一行
        将其解析成：
            1、一个图路径
            2、多个gt的坐标信息
            3、类别信息。
    '''
    pic_path, boxes, labels = parse_line(line)

    '''
        读取图片，
        resize图片
        根据resize尺寸，将gt的坐标信息也对应到resize后的内容上
    '''
    img = cv2.imread(pic_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img, boxes = resize_image_and_correct_boxes(img, boxes, img_size)

    # 图像增强
    # do data augmentation here
    if mode == 'train':
        img, boxes, labels = data_augmentation(img, boxes, labels)
    '''
        超重要！
        将图像的值放缩到0~1之间
    '''
    # the input of yolo_v3 should be in range 0~1
    img = img / 255.
    '''
        这里：
        boxes:[None,4]
        labels:[None,]
        img_size  = 416,416
        class_num = 80
        anchors = [[10, 13], [16, 30], [33, 23],
                         [30, 61], [62, 45], [59,  119],
                         [116, 90], [156, 198], [373,326]]

    '''
    y_true_13, y_true_26, y_true_52 = process_box(boxes, labels, img_size,
                                                  class_num, anchors)

    return img, y_true_13, y_true_26, y_true_52
