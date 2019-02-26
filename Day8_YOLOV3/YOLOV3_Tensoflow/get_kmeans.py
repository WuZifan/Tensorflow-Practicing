# coding: utf-8
# This script is modified from https://github.com/lars76/kmeans-anchor-boxes

from __future__ import division, print_function

import numpy as np

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """

    # 比较cluster中的每一个宽和高，与box的宽和高的大小
    # 返回值是 [k,]的vector,
    # 两个array，对应位置的元素相互比较，选择较小的那一个。
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    # 保证长宽不为0
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
    # 重合部分？
    intersection = x * y
    # 计算面积
    box_area = box[0] * box[1]
    # 计算每一个cluster的面积
    cluster_area = clusters[:, 0] * clusters[:, 1]
    # 这个IOU定义的好奇怪……
    # 应该是是在计算，仅知道长和宽的情况下，最大的IOU吧= -
    iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU)
    between a numpy array of boxes and k clusters.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    param:
        boxes: numpy array of shape (r, 4)
    return:
    numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0] #有几个

    distances = np.empty((rows, k)) # 保存距离用的
    last_clusters = np.zeros((rows,)) # 上一次的中心点？

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    # 随机选取k个中心点
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            # 计算每一个样本和9个中心点的距离
            # 用1-（自定义IOU计算）
            distances[row] = 1 - iou(boxes[row], clusters)

        # 拿到最近的cluster
        nearest_clusters = np.argmin(distances, axis=1)

        '''
            last_clusters 和 nearest_cluster都是两个长为row的vector
            这里我们来比较，对于每个box而言，如果它这次的中心选的和上次一样
            那么就可以不用继续k_means了。
        '''
        if (last_clusters == nearest_clusters).all():
            break

        '''
            对于所有第i类的样本，用他们的均值作为聚类中心
        '''
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        # 更新聚类中心。
        last_clusters = nearest_clusters

    return clusters


def parse_anno(annotation_path):
    anno = open(annotation_path, 'r')
    result = []
    '''
        对于每一行：xxx/xxx/1.jpg 0 453 369 473 391 1 588 245 608 268

    '''
    for line in anno:
        s = line.strip().split(' ')
        s = s[1:] # 不要绝对路径
        box_cnt = len(s) // 5  # 每5个为一组（label+gt框）
        for i in range(box_cnt):
            # 拿到gt框的坐标
            x_min, y_min, x_max, y_max = float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3]), float(s[i*5+4])
            # 计算宽和高
            width = x_max - x_min
            height = y_max - y_min
            assert width > 0
            assert height > 0
            # 这里返回的是每一个沟通框的宽和高
            result.append([width, height])
    result = np.asarray(result)
    return result


def get_kmeans(anno, cluster_num=9):
    '''

    :param anno: [ [w,h],[w,h],[w,h],[w,h]。。。],
    :param cluster_num: 9
    :return:
    '''
    # 这组参数输入k_means
    # 返回的是cluster_num个宽高信息，作为聚类信息
    # anchors.shape = (9,2)
    anchors = kmeans(anno, cluster_num)
    # 对于每一个box，计算它和9个聚类中心的IOU最大值
    # 然后再求一个均值
    ave_iou = avg_iou(anno, anchors)

    # 按照聚好类的anchors排个序
    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou


if __name__ == '__main__':
    '''
        告诉我们如何得到几种不同的anchors的。

        我们输入的格式是：image_absolute_path box_1 box_2 ... box_n. Box_format: label_index x_min y_min x_max y_max：
        例子：xxx/xxx/1.jpg 0 453 369 473 391 1 588 245 608 268
        
        基本而言它是这么一个流程：
            1、本质上是一个kmeans，那么问题就是如何准备数据，以及计算数据之间的size。
            2、数据准备上，每一个anchors都是用他们的宽和高作为他们特征。
            3、数据计算上，比较有意思的点，是通过计算两者的最大重合面积，算出anchors的之间的IOU
            4、从而来计算anchors的距离。
            5、具体而言 dis(a1,a2) = 1-IOU(a1,a2)

    '''
    annotation_path = "./data/my_data/train.txt"
    anno_result = parse_anno(annotation_path)
    anchors, ave_iou = get_kmeans(anno_result, 9)

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-2]

    print('anchors are:')
    print(anchor_string)
    print('the average iou is:')
    print(ave_iou)

