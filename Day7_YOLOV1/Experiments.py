# import tensorflow as tf

#这个是定义损失函数的，具体可以看论文
'''
self.boundary1 = self.cell_size * self.cell_size * self.num_class  #类似于7*7*20
self.boundary2 = self.boundary1 +\
    self.cell_size * self.cell_size * self.boxes_per_cell  #类似于 7*7*20 + 7*7*2
'''
def loss_layer(self, predicts, labels, scope='loss_layer'):   #定义损失函数，损失函数的具体形似可以查看论文, label的格式为[batch_size, 7, 7, 25]
    with tf.variable_scope(scope):
        # 预测的20类分类结果
        predict_classes = tf.reshape(  #reshape一下，每个cell一个框，变成[batch_size, 7, 7, 20]
            predicts[:, :self.boundary1],
            [self.batch_size, self.cell_size, self.cell_size, self.num_class])
        # 预测的前背景分类结果
        predict_scales = tf.reshape( #reshape一下，7*7*20 ~ 7*7*22, 就是分别找到每个cell的两个框的置信度,这里是两个框，可自定义,变成[batch_size, 7, 7, 2]
            predicts[:, self.boundary1:self.boundary2],
            [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
        # 预测的回归框结果
        predict_boxes = tf.reshape( #reshape，就是分别找到每个cell中两个框的坐标（x_center, y_center, w, h），这里是两个框，可自定义, 变成[batch_size, 7, 7, 2, 4]
            predicts[:, self.boundary2:],  #7 * 7 * 22 ~ 7 * 7 * 30，
            [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])


        '''
        感觉比较明显，如果这个cell不是用来对这个gt做分类的话，后面的都没有意义了…
        '''
        #下面是对label部分进行reshape
        # 每一个gt由一个cell负责标记，这里是找到这个cell是不是负责gt的（任意一个都可以）
        response = tf.reshape(
            labels[..., 0],
            [self.batch_size, self.cell_size, self.cell_size, 1])    #reshape, 就是查看哪个cell负责标记object,是的话就为1 ，否则是0 ，维度形式：[batch_size, 7, 7, 1]
        # 拿到这个cell对应的gt的坐标值（重复两遍）
        boxes = tf.reshape(
            labels[..., 1:5],
            [self.batch_size, self.cell_size, self.cell_size, 1, 4])  #找到这个cell负责的框的位置，其形式为：(x_center,y_center,width,height), 其维度为：[batch_size, 7, 7, 1, 4]
        boxes = tf.tile(
            boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size   # tile() 平铺之意，用于在同一维度上的复制, 变成[batch_size, 7, 7, 2, 4]， 除以image_size就是得到相对于整张图片的比例
        # 拿到这个cell的分类结果
        classes = labels[..., 5:]          #找到这个cell负责的框所框出的类别，有20个类别, 变成[batch_size, 7, 7, 20]，正确的类别对应的位置为1，其它为0

        offset = tf.reshape(
            tf.constant(self.offset, dtype=tf.float32),
            [1, self.cell_size, self.cell_size, self.boxes_per_cell])    #由7*7*2 reshape成 1*7*7*2
        offset = tf.tile(offset, [self.batch_size, 1, 1, 1])  #在第一个维度上进行复制，变成 [batch_size, 7, 7,2]
        offset_tran = tf.transpose(offset, (0, 2, 1, 3))  #维度为[batch_size, 7, 7, 2]

        #offset_tran如下，只不过batch_size=1
        # [[[[0. 0.]
        # [0. 0.]
        # [0. 0.]
        # [0. 0.]
        # [0. 0.]
        # [0. 0.]
        # [0. 0.]]
        #
        # [[1. 1.]
        #  [1. 1.]
        # [1. 1.]
        # [1. 1.]
        # [1. 1.]
        # [1. 1.]
        # [1. 1.]]
        #
        # [[2. 2.]
        #  [2. 2.]
        # [2. 2.]
        # [2. 2.]
        # [2. 2.]
        # [2. 2.]
        # [2. 2.]]
        #
        # [[3. 3.]
        #  [3. 3.]
        # [3. 3.]
        # [3. 3.]
        # [3. 3.]
        # [3. 3.]
        # [3. 3.]]
        #
        # [[4. 4.]
        #  [4. 4.]
        # [4. 4.]
        # [4. 4.]
        # [4. 4.]
        # [4. 4.]
        # [4. 4.]]
        #
        # [[5. 5.]
        #  [5. 5.]
        # [5. 5.]
        # [5. 5.]
        # [5. 5.]
        # [5. 5.]
        # [5. 5.]]
        #
        # [[6. 6.]
        #  [6. 6.]
        # [6. 6.]
        # [6. 6.]
        # [6. 6.]
        # [6. 6.]
        # [6. 6.]]]]
        #

        predict_boxes_tran = tf.stack(   #相对于整张特征图来说，找到相对于特征图大小的中心点，和宽度以及高度的开方， 其格式为[batch_size, 7, 7, 2, 4]
            [(predict_boxes[..., 0] + offset) / self.cell_size,   #self.cell=7
             (predict_boxes[..., 1] + offset_tran) / self.cell_size,
             tf.square(predict_boxes[..., 2]),    #宽度的平方，和论文中的开方对应，具体请看论文
             tf.square(predict_boxes[..., 3])], axis=-1)  #高度的平方，

        iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)   #计算IOU,  其格式为： [batch_size, 7, 7, 2]

        # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)  # Computes the maximum of elements across dimensions of a tensor, 在第四个维度上，维度从0开始算
        object_mask = tf.cast(
            (iou_predict_truth >= object_mask), tf.float32) * response   #其维度为[batch_size, 7, 7, 2]  , 如果cell中真实有目标，那么该cell内iou最大的那个框的相应位置为1（就是负责预测该框），其余为0

        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        noobject_mask = tf.ones_like(
            object_mask, dtype=tf.float32) - object_mask    #其维度为[batch_size, 7 , 7, 2]， 真实没有目标的区域都为1，真实有目标的区域为0

        boxes_tran = tf.stack(     #stack这是一个矩阵拼接的操作， 得到x_center, y_center相对于该cell左上角的偏移值， 宽度和高度是相对于整张图片的比例
            [boxes[..., 0] * self.cell_size - offset,
             boxes[..., 1] * self.cell_size - offset_tran,
             tf.sqrt(boxes[..., 2]),   #宽度开方，和论文对应
             tf.sqrt(boxes[..., 3])], axis=-1)  #高度开方，和论文对应

        # class_loss, 计算类别的损失
        class_delta = response * (predict_classes - classes)
        class_loss = tf.reduce_mean(   #平方差损失函数
            tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
            name='class_loss') * self.class_scale   # self.class_scale为损失函数前面的系数

        # 有目标的时候，置信度损失函数
        object_delta = object_mask * (predict_scales - iou_predict_truth)  #用iou_predict_truth替代真实的置信度，真的妙，佩服的5体投递
        object_loss = tf.reduce_mean(  #平方差损失函数
            tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
            name='object_loss') * self.object_scale

        # 没有目标的时候，置信度的损失函数
        noobject_delta = noobject_mask * predict_scales
        noobject_loss = tf.reduce_mean(       #平方差损失函数
            tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
            name='noobject_loss') * self.noobject_scale

        # 框坐标的损失，只计算有目标的cell中iou最大的那个框的损失，即用这个iou最大的框来负责预测这个框，其它不管，乘以0
        coord_mask = tf.expand_dims(object_mask, 4)  # object_mask其维度为：[batch_size, 7, 7, 2]， 扩展维度之后变成[batch_size, 7, 7, 2, 1]
        boxes_delta = coord_mask * (predict_boxes - boxes_tran)     #predict_boxes维度为： [batch_size, 7, 7, 2, 4]，这些框的坐标都是偏移值
        coord_loss = tf.reduce_mean(  #平方差损失函数
            tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
            name='coord_loss') * self.coord_scale

        tf.losses.add_loss(class_loss)
        tf.losses.add_loss(object_loss)
        tf.losses.add_loss(noobject_loss)
        tf.losses.add_loss(coord_loss)  #将各个损失总结起来

        tf.summary.scalar('class_loss', class_loss)
        tf.summary.scalar('object_loss', object_loss)
        tf.summary.scalar('noobject_loss', noobject_loss)
        tf.summary.scalar('coord_loss', coord_loss)

        tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
        tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
        tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
        tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
        tf.summary.histogram('iou', iou_predict_truth)

if __name__ == '__main__':
    import numpy as np
    print('hehe')

    offset = np.transpose(np.reshape(np.array(  # reshape之后再转置，变成7*7*2的三维数组
        [np.arange(7)] * 7 * 2),
        (2, 7, 7)), (1, 2, 0))

    offset_tran = np.transpose(offset,(1,0,2))

    print(offset.shape)
    # print(offset)
    # print(offset_tran)
    import os
    import yolo.config as cfg
    yolo = os.path.join(cfg.PASCAL_PATH,'VOCdevkit')
    print(os.path.curdir)
    if os.path.exists(yolo):
        print(yolo)

    aa=np.array([[[1,2,3],[4,5,6]],
              [[7, 8, 9], [10, 11, 12]]])
    print(aa[:,::-1,:])

    print(aa[:,:,1])