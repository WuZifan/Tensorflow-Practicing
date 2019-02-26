import tensorflow as tf
import numpy as np
if __name__ == '__main__':

    grid_x = tf.range(13,dtype=tf.int32)
    grid_y = tf.range(13,dtype=tf.int32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)


    k = [[10, 13], [16, 30], [33, 23],
                         [30, 61], [62, 45], [59,  119],
                         [116, 90], [156, 198], [373,326]]
    kk = [k[6:9],k[3:6]]

    ee = [1,2,3,4,5]
    print(ee[2:3])
    a = tf.constant(dtype=tf.float32,value=np.zeros(shape=[2,13,13,3,2]))
    b = tf.constant(dtype=tf.float32,value=kk[0])
    print(a/b)


    def reorg_layer(self, feature_map, anchors):
        '''
        将预测bbox的坐标信息映射回原图尺寸，
        并把不同预测内容的维度分开
        feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3] returned
            from `forward` function
            [ [None,13,13,255],[None,26,26,255],[None,52,52,255]]中的一个
        anchors: shape: [3, 2]
        '''
        '''
            1、拿到现在feature_map的尺寸
            2、拿到缩放倍数。
        '''
        # NOTE: size in [h, w] format! don't get messed up!
        grid_size = feature_map.shape.as_list()[1:3]  # [13, 13]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)

        '''
            注意：
                1、所有的anchors的宽高都是在原始图像上。
                2、现在需要把anchors的尺寸映射到现在这个尺寸的feature-map上。
        '''
        # rescale the anchors to the feature_map
        # NOTE: the anchor is in [w, h] format!
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        '''
            把最后一个维度展开，从255变成[3,85]
        '''
        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        '''
        将最后一维分一下
         split the feature_map along the last dimension
         shape info: take 416x416 input image and the 13*13 feature_map for example:
         box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
         box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
         conf_logits: [N, 13, 13, 3, 1]
         prob_logits: [N, 13, 13, 3, class_num]
        '''
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)

        '''
            中心位置过一下sigmoid，变成0~1值，表示相对cell左上角的偏移
        '''
        box_centers = tf.nn.sigmoid(box_centers)

        '''
            拿到13*13的feature_map上，每个cell左上角的坐标吧
        '''
        # use some broadcast tricks to get the mesh coordinates
        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        # shape: [13, 13, 1, 2]
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        '''
        拿到相对整个feature_map左上角的距离
        并且映射回原图上的位置
        '''
        # get the absolute box coordinates on the feature_map
        box_centers = box_centers + x_y_offset
        # rescale to the original image scale
        box_centers = box_centers * ratio[::-1]

        '''
            尺寸就直接映射回原位置就好
            别忘了宽高的计算公式：anchor*exp(predict_w) = bbox_real_coor
        '''
        # avoid getting possible nan value with tf.clip_by_value
        box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 50) * rescaled_anchors
        # rescale to the original image scale
        box_sizes = box_sizes * ratio[::-1]

        # shape: [N, 13, 13, 3, 4]
        # last dimension: (center_x, center_y, w, h)
        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        '''
            返回处理过的信息：
                x_y_offset：feature_map上每个cell的左上角坐标 shape: [13, 13, 1, 2]
                boxes：bbox在原图上的位置 [N, 13, 13, 3, 4]
                conf_logits：bbox的置信度 [N, 13, 13, 3, 1]
                prob_logits：bbox的分类信息 [N, 13, 13, 3, class_num]

        '''
        # shape:
        # x_y_offset: [13, 13, 1, 2]
        # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        return x_y_offset, boxes, conf_logits, prob_logits
