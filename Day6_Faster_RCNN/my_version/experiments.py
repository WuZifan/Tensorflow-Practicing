import tensorflow as tf
from keras import backend as K
'''
ROI 池化
'''
class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
        
    '''

    '''
    keras自定义的类：
        0、在init中通过输入数据，初始化一些变量
        1、在build中定义训练用参数
        2、在call中，根据传入的x，来定义op，以及执行顺序。
    '''

    def __init__(self, pool_size, num_rois, **kwargs):
        self.dim_ordering = K.image_dim_ordering()
        # 把一个区域ROI区域分割成几份，那么pool_size*pool_size的值就是ROI区域的特征值
        # 值是7
        self.pool_size = pool_size
        # 每次处理几个roi区域。（这个还可以并行的？？）
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        
        这个就和原始tf定义conv操作一样，需要自己定义W和b
        这里就是定义W和b这些参数的地方
        :param input_shape: 
        :return: 
        '''
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels


    def call(self, x, mask=None):
        '''
        
        :param x: [base_layers, input_rois]
                    base_layer，表示原始的特征图【None,None,None,512】 第0维表示batch_size
                    input_rois，表示想看几个ROI，【None,None，4】 第0维表示batch_size
                                                      (num_rois,4) with ordering (x,y,w,h)
        :param mask: 
        :return: 
        '''
        assert (len(x) == 2)
        '''
        这里是定义，如何计算的地方
        对应到tensorflow中，就是定义如何通过 op来计算参数的地方
        '''
        # x[0] is image with shape (None,rows, cols, channels)
        img = x[0]

        # x[1] is roi with shape (None,num_rois,4) with ordering (x,y,w,h)
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):
            '''
            拿到第roi_idx个anchors的坐标
            '''
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # Resized roi of the image to pooling size (7x7)
            # 这里看函数简介就知道，根据输入的shape不同，它会在不同的维度上进行resize
            # 这里就是对于这个anchors进行resize。
            '''
            所以这里其实验证了，只能有一张图。
            因为anchors在不同的图上不一样，而这里做了跨batch的resize。
            唯一的解释就只能是一张图，
            才能保证一个anchors在所有的batch和channel维度上都一样
            '''
            rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :],
                                        (self.pool_size, self.pool_size))
            outputs.append(rs)

        # 这里完了之后应该是 [4,7,7,3]
        final_output = K.concatenate(outputs, axis=0)

        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        # Might be (1, 4, 7, 7, 3)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        # permute_dimensions is similar to transpose
        # 这个是重拍吧，但是这个没有用啊…
        # 难道只是单纯为了得到一个相同的tensor？？
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





'''
    定义分类层
    '''
def classifier_layer(self,base_layers, input_rois, num_rois, nb_classes = 4):
    """Create a classifier layer

        Args:
            base_layers: vgg
            input_rois: `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
            num_rois: number of rois to be processed in one time (4 in here)

        Returns:
            list(out_class, out_regr)
            out_class: classifier layer output
            out_regr: regression layer output
        """

    input_shape = (num_rois, 7, 7, 512)

    pooling_regions = 7

    # out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
    # num_rois (4) 7x7 roi pooling
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    # Flatten the convlutional layer and connected to 2 FC and 2 dropout
    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    # There are two output layer
    # out_class: softmax acivation function for classify the class name of the object
    # out_regr: linear activation function for bboxes coordinates regression
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]


def rpn_to_roi(rpn_layer, regr_layer, C,
               dim_ordering, use_regr=True,
               max_boxes=300, overlap_thresh=0.9):
    """Convert rpn layer to roi bboxes
    
    准确的说，我看了只是对anchors的表示方法修改了一下。
    从[height,weight,4*9],转换成[num_anchors,4]
    然后进行简单的筛选和NMS过滤
    
    将RPN层的结果，转换成感兴趣区域！
    
    这里的尺寸有点问题啊
    上面不是看过了，rpn网络的输出，class的shape应该是（1，feat_height，feat_weight,9）
                                     reg的shape应该是（1，feat_heigh , feat_weight,4*9) 
    Args: (num_anchors = 9)
        rpn_layer: output layer for rpn classification 
            shape (1, feature_map.height, feature_map.width, num_anchors)
            Might be (1, 18, 25, 18) if resized image is 400 width and 300
        regr_layer: output layer for rpn regression
            shape (1, feature_map.height, feature_map.width, num_anchors)
            Might be (1, 18, 25, 72) if resized image is 400 width and 300
        C: config
        use_regr: Whether to use bboxes regression in rpn
        max_boxes: max bboxes number for non-max-suppression (NMS)
        overlap_thresh: If iou in NMS is larger than this threshold, drop the box

    Returns:
        result: boxes from non-max-suppression (shape=(300, 4))
            boxes: coordinates for bboxes (on the feature map)
    """
    regr_layer = regr_layer / C.std_scaling

    anchor_sizes = C.anchor_box_scales  # (3 in here)
    anchor_ratios = C.anchor_box_ratios  # (3 in here)

    assert rpn_layer.shape[0] == 1 # 确定batch——size=1，即，就一张图的

    (rows, cols) = rpn_layer.shape[1:3] # 拿到特征图的尺寸

    curr_layer = 0

    # A.shape = (4, feature_map.height, feature_map.width, num_anchors)
    # Might be (4, 18, 25, 18) if resized image is 400 width and 300
    # A is the coordinates for 9 anchors for every point in the feature map
    # => all 18x25x9=4050 anchors cooridnates
    # 4，高，宽，分类
    A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:
            # anchor_x = (128 * 1) / 16 = 8  => width of current anchor
            # anchor_y = (128 * 2) / 16 = 16 => height of current anchor
            '''
                将定义在原图上的anchors，映射到现在feature_map上。
            '''
            anchor_x = (anchor_size * anchor_ratio[0]) / C.rpn_stride # anchor在feat_map上的高
            anchor_y = (anchor_size * anchor_ratio[1]) / C.rpn_stride # anchor在feat_map上的宽

            # curr_layer: 0~8 (9 anchors)
            # the Kth anchor of all position in the feature map (9th in total)
            regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]  # shape => (18, 25, 4)
            regr = np.transpose(regr, (2, 0, 1))  # shape => (4, 18, 25)

            # Create 18x25 mesh grid
            # For every point in x, there are all the y points and vice versa
            # X.shape = (18, 25)
            # Y.shape = (18, 25)
            # 这个是拿到坐标矩阵，X表示 每个位置的x坐标值
            #                     Y表示 每个位置的y坐标值
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

            # Calculate anchor position and size for each feature map point
            # 找到了原图上的anchor，在现在feature_map上的映射
            # 用左上角点位+长宽表示
            # 这里可以想象成，需要给4个 （18,25）的矩阵填充信息
            # 每一矩阵表示一个坐标信息。

            '''
             curr_layer表示这一种anchors，0~3表示这种anchors的坐标信息
             中间两个 ： ，表示在每个位置上的这种anchors
             
             那么，我们现在就是得出了，在每个位置上，当前这种anchors的四个坐标信息
            '''

            A[0, :, :, curr_layer] = X -   / 2  # Top left x coordinate
            A[1, :, :, curr_layer] = Y - anchor_y / 2  # Top left y coordinate
            A[2, :, :, curr_layer] = anchor_x  # width of current anchor
            A[3, :, :, curr_layer] = anchor_y  # height of current anchor

            # A[:,:,:,curr_layer]保存的就是每一个anchors的原始坐标信息，和预测无关
            # regr里面就是预测的偏移量
            # 这里就是把偏移量用在原始值上面，得到最终的框位置
            # Apply regression to x, y, w and h if there is rpn regression layer
            if use_regr:
                # 输入是这一个anchors在所有点上的坐标信息，以及回归框信息
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

            # 这个沙雕吧…不会写英文注释别写啊
            # 这个 是 avoid xxxx smaller than 1 把……
            # 即，长宽的下限是1，不然没法圈了
            # 但其实这里这里有问题，即默认物体的最小尺寸是16*16，
            # 再小的就不行了…
            # 这个会不会就是多尺度检测思路的由来之一啊…
            # Avoid width and height exceeding 1
            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])

            # Convert (x, y , w, h) to (x1, y1, x2, y2)
            # x1, y1 is top left coordinate
            # x2, y2 is bottom right coordinate
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

            # Avoid bboxes drawn outside the feature map
            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols - 1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows - 1, A[3, :, :, curr_layer])

            curr_layer += 1

    # 位置信息 （18,25,9*4）
    # 相当于有18*25*9个anchors，每个anchors对应4个位置信息
    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))  # shape=(4050, 4)
    # 概率值 （18,25,9）
    # 相当于有18*25*9个anchors，每个anchors对应一个概率值，表示是前景的概率
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))  # shape=(4050,)

    # 预定x1,y1是小的坐标，x2,y2是大的坐标
    # 那么如果不满足这个假设，就删除这个anchors
    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    # Find out the bboxes which is illegal and delete them from bboxes list
    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

    # 非极大值抑制
    # 返回的是两个值，boxes, probs
    # 但是后续只需要boxes了，所以不用probs
    # Apply non_max_suppression
    # Only extract the bboxes. Don't need rpn probs in the later process
    result= non_max_suppression_fast(all_boxes, all_probs,
                                      overlap_thresh=overlap_thresh,
                                      max_boxes=max_boxes)[0]

    return result

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list

    # Process explanation:
    #   Step 1: Sort the probs list
    #   Step 2: Find the larget prob 'Last' in the list and save it to the pick list
    #   Step 3: Calculate the IoU with 'Last' box and other boxes in the list. If the IoU is larger than overlap_threshold, delete the box from list
    #   Step 4: Repeat step 2 and step 3 until there is no item in the probs list
    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs


def calc_iou(R, img_data, C, class_mapping):
    """Converts from (x1,y1,x2,y2) to (x,y,w,h) format

    Args:
        R: bboxes, probs
        img_data: img_data中没有保存图像的数据，保存的是图像的一些参数，比如长宽，起始点，路径之类的
                  即图像的属性
        C：配置类
        class_mapping:
    """
    bboxes = img_data['bboxes']
    (width, height) = (img_data['width'], img_data['height'])
    # get image dimensions for resizing
    (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

    gta = np.zeros((len(bboxes), 4))

    # 这里通过 图像原始的尺寸，和resize之后的尺寸缩放，来确定了resize图像上bbox的坐标位置
    # 之后通过除以16，来确定图像上bbox在feature_map上的坐标
    for bbox_num, bbox in enumerate(bboxes):
        # get the GT box coordinates, and resize to account for image resizing
        # gta[bbox_num, 0] = (40 * (600 / 800)) / 16 = int(round(1.875)) = 2 (x in feature map)
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/C.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/C.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/C.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/C.rpn_stride))

    '''
    上面拿到的是ground_truth的框位置
    下面要计算预测的框，和ground_truth的重合情况吧
    '''

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = [] # for debugging only

    # R.shape[0]: number of bboxes (=300 from non_max_suppression)
    for ix in range(R.shape[0]):
        '''
        拿到第ix个预测框在feature_map上的位置
        '''
        (x1, y1, x2, y2) = R[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1
        '''
        想看一下这个预测框，和哪个ground_truth框的ioc比较高
        '''
        # Iterate through all the ground-truth bboxes to calculate the iou
        for bbox_num in range(len(bboxes)):
            # 计算两个框的iou
            curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                           [x1, y1, x2, y2])

            # Find out the corresponding ground-truth bbox_num with larget iou
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        # 如果最好的iou还达不到最小值，那么这个预测框可以舍弃
        if best_iou < C.classifier_min_overlap:
                continue
        else:
            # 否则，保存这个预测框的坐标值和IOU值
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            # 然后保存这个预测框的类别信息
            if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
                # hard negative example
                cls_name = 'bg'
            elif C.classifier_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        # 这里是做个one-hot吧…………
        class_num = class_mapping[cls_name]
        class_label = len(class_mapping) * [0]
        class_label[class_num] = 1
        y_class_num.append(copy.deepcopy(class_label))
        # 这里？？？
        coords = [0] * 4 * (len(class_mapping) - 1) # bg就不用管坐标？
        labels = [0] * 4 * (len(class_mapping) - 1) # 不要bg那一类？
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = C.classifier_regr_std
            coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
            labels[label_pos:4+label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    # bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
    X = np.array(x_roi)
    # one hot code for bboxes from above => x_roi (X)
    Y1 = np.array(y_class_num)
    # corresponding labels and corresponding gt bboxes
    Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs

def apply_regr_np(X, T):
    """Apply regression layer to all anchors in one feature map

    Args:
        X: shape=(4, 18, 25) the current anchor type for all points in the feature map
        T: regression layer shape=(4, 18, 25)

    Returns:
        X: regressed position and size for current anchor
    """

    '''
    这里就是把预测的偏移值，用到anchors的原始值上面
    得到ROI区域的最终范围
    '''
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w/2.
        cy = y + h/2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X