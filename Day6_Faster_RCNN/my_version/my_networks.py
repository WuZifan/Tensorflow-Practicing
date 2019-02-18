import os
import pandas as pd
import numpy as np
import time

import tensorflow as tf

from sklearn.metrics import average_precision_score

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy

from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers

import my_cfg

class Network:
    def __init__(self,config):

        self.C = config
        self.lambda_rpn_regr = 1.0
        self.lambda_rpn_class = 1.0

        self.lambda_cls_regr = 1.0
        self.lambda_cls_class = 1.0

        self.epsilon = 1e-4
        '''
        构建共享的网络层
        '''
        # input_shape_img = (None, None, 3)
        # input_shape_img = (300, 400, 3)
        batch_input_shape_img = (1,300,400,3)
        # img_input = Input(shape=input_shape_img)
        img_input = Input(batch_shape=batch_input_shape_img)
        roi_input = Input(shape=(None, 4))

        # define the base network (VGG here, can be Resnet50, Inception, etc)
        shared_layers = self.nn_base(img_input, trainable=True)

        vgg_model  = Model(img_input,shared_layers)
        print('num_rois',self.C.num_rois)
        print('img_input',img_input.shape)
        print('vgg_model_output',shared_layers.shape)
        # print('vgg_model',vgg_model.summary())

        '''
        根据共享层定义RPN结构
        '''
        # define the RPN, built on the base layers
        # 这个是定义每个点上面有几个anchors的
        self.num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)  # 9
        print('num_anchors',self.num_anchors)
        rpn = self.rpn_layer(shared_layers, self.num_anchors)
        print('rpn_class',rpn[0].shape)
        print('rpn_reg',rpn[1].shape)

        '''
        输入是：
            shared_layers:VGG的输出【18,25,512】
            roi_input: 一个placeholder，【None,4】表示None个坐标
            C.num_rois:一次输入几个anchors
            nb_classes:类别总数
        '''
        classifier = self.classifier_layer(shared_layers, roi_input, self.C.num_rois,
                                           nb_classes=len(self.C.classes_count))

        print('classifier_class',classifier[0].shape)
        print('classifier_reg',classifier[1].shape)

        model_rpn = Model(img_input, rpn[:2])
        model_classifier = Model([img_input, roi_input], classifier)
        # print('rpn:',model_rpn.summary())
        # print('classifier:',model_classifier.summary())

        # model_classifier.train_on_batch()
        '''
        构建模型，第一个参数是输入，第二个参数是输出吧
        查了中文文档，是的
        '''
        # this is a model that holds both the RPN and the classifier,
        # used to load/save weights for the models
        model_all = Model([img_input, roi_input], rpn[:2] + classifier)

        # Because the google colab can only run
        # the session several hours one time (then you need to connect again),
        # we need to save the model and load the model to continue training
        if not os.path.isfile(self.C.model_path):
            # If this is the begin of the training, load the pre-traind base network such as vgg-16
            try:
                print('This is the first time of your training')
                print('loading weights from {}'.format(self.C.base_net_weights))
                model_rpn.load_weights(self.C.base_net_weights, by_name=True)
                model_classifier.load_weights(self.C.base_net_weights, by_name=True)
            except:
                print('Could not load pretrained model weights. '
                      'Weights can be found in the keras application folder \
                    https://github.com/fchollet/keras/tree/master/keras/applications')

            # Create the record.csv file to record losses, acc and mAP
            record_df = pd.DataFrame(
                columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls',
                         'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
        else:
            # If this is a continued training, load the trained model from before
            print('Continue training based on previous trained model')
            print('Loading weights from {}'.format(self.C.model_path))
            model_rpn.load_weights(self.C.model_path, by_name=True)
            model_classifier.load_weights(self.C.model_path, by_name=True)

            # Load the records
            record_df = pd.read_csv(self.C.record_path)

            r_mean_overlapping_bboxes = record_df['mean_overlapping_bboxes']
            r_class_acc = record_df['class_acc']
            r_loss_rpn_cls = record_df['loss_rpn_cls']
            r_loss_rpn_regr = record_df['loss_rpn_regr']
            r_loss_class_cls = record_df['loss_class_cls']
            r_loss_class_regr = record_df['loss_class_regr']
            r_curr_loss = record_df['curr_loss']
            r_elapsed_time = record_df['elapsed_time']
            r_mAP = record_df['mAP']

            print('Already train %dK batches' % (len(record_df)))

        '''
        定义网络的loss
        '''
        self.set_loss(model_rpn,model_classifier,model_all)


        pass

    '''
    定义基础网络
    '''
    def nn_base(self,input_tensor=None, trainable=False):
        input_shape = (None, None, 3)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        bn_axis = 3

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        return x

    '''
    定义rpn层
    '''
    def rpn_layer(self,base_layers, num_anchors):
        """Create a rpn layer
            Step1: Pass through the feature map from base layer to a 3x3 512 channels convolutional layer
                    Keep the padding 'same' to preserve the feature map's size
            Step2: Pass the step1 to two (1,1) convolutional layer to replace the fully connected layer
                    classification layer: num_anchors (9 in here) channels for 0, 1 sigmoid activation output
                    regression layer: num_anchors*4 (36 in here) channels for computing the regression of bboxes with linear activation
        Args:
            base_layers: vgg in here
            num_anchors: 9 in here

        Returns:
            [x_class, x_regr, base_layers]
            x_class: classification for whether it's an object
            x_regr: bboxes regression
            base_layers: vgg in here
        """
        x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
            base_layers)

        x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                         name='rpn_out_class')(x)
        x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                        name='rpn_out_regress')(x)

        return [x_class, x_regr, base_layers]

    '''
    定义分类层
    '''
    def classifier_layer(self,base_layers, input_rois, num_rois, nb_classes = 4):
        """Create a classifier layer
    
            base_layers:VGG的输出【18,25,512】
            input_rois: 一个placeholder，【None,4】表示None个坐标
            num_rois:一次输入几个anchors
            nb_classes:类别总数

            Args:
                base_layers: vgg
                input_rois: `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
                num_rois: number of rois to be processed in one time (4 in here)

            Returns:
                list(out_class, out_regr)
                out_class: classifier layer output
                out_regr: regression layer output
            """

        input_shape = (num_rois, 7, 7, 512) # 这里应该是（num_rois,None，None，512）的

        pooling_regions = 7

        # out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
        # num_rois (4) 7x7 roi pooling
        # 返回值是这个shape的(1, num_rois, pool_size, pool_size, nb_channels)
        # 说来好笑，这里就是用resize来实现Roi_Pooling,然后返回上面那个size的数据
        out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
        print('out_roi_pool',out_roi_pool.shape)
        # Flatten the convlutional layer and connected to 2 FC and 2 dropout
        # （1,4,7,7,512） 1表示一个样本，是batch shape
        # TimeDistribute从4这个维度上进行
        # flatten (?, 4, 25088)
        out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)

        print('flatten',out.shape)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
        print('Dense',out.shape)
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


    '''
    定义loss
    '''
    def set_loss(self,model_rpn,model_classifier,model_all):

        optimizer = Adam(lr=1e-5)
        optimizer_classifier = Adam(lr=1e-5)
        model_rpn.compile(optimizer=optimizer, loss=[self.rpn_loss_cls(self.num_anchors),
                                                     self.rpn_loss_regr(self.num_anchors)])
        model_classifier.compile(optimizer=optimizer_classifier,
                                 loss=[self.class_loss_cls,
                                       self.class_loss_regr(len(self.C.classes_count) - 1)],
                                 metrics={'dense_class_{}'.format(len(self.C.classes_count)): 'accuracy'})
        model_all.compile(optimizer='sgd', loss='mae')

    '''
    RPN网络的回归loss
    '''
    def rpn_loss_regr(self,num_anchors):
        """Loss function for rpn regression
        Args:
            num_anchors: number of anchors (9 in here)
        Returns:
            Smooth L1 loss function 
                               0.5*x*x (if x_abs < 1)
                               x_abx - 0.5 (otherwise)
        """

        def rpn_loss_regr_fixed_num(y_true, y_pred):
            # x is the difference between true value and predicted vaue
            x = y_true[:, :, :, 4 * num_anchors:] - y_pred

            # absolute value of x
            x_abs = K.abs(x)

            # If x_abs <= 1.0, x_bool = 1
            x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

            return self.lambda_rpn_regr * K.sum(
                y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
                self.epsilon + y_true[:, :, :, :4 * num_anchors])

        return rpn_loss_regr_fixed_num

    '''
    RPN网络的分类loss
    '''
    def rpn_loss_cls(self,num_anchors):
        """Loss function for rpn classification
        Args:
            num_anchors: number of anchors (9 in here)
            y_true[:, :, :, :9]: [0,1,0,0,0,0,0,1,0] means only the second and the eighth box is valid which contains pos or neg anchor => isValid
            y_true[:, :, :, 9:]: [0,1,0,0,0,0,0,0,0] means the second box is pos and eighth box is negative
        Returns:
            lambda * sum((binary_crossentropy(isValid*y_pred,y_true))) / N
        """

        def rpn_loss_cls_fixed_num(y_true, y_pred):
            return self.lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :],
                                                                                                  y_true[:, :, :,
                                                                                                  num_anchors:])) / K.sum(
                self.epsilon + y_true[:, :, :, :num_anchors])

        return rpn_loss_cls_fixed_num

    '''
    总网络的回归loss
    '''
    def class_loss_regr(self,num_classes):
        """Loss function for rpn regression
            Args:
                num_anchors: number of anchors (9 in here)
            Returns:
                Smooth L1 loss function 
                                   0.5*x*x (if x_abs < 1)
                                   x_abx - 0.5 (otherwise)
            """

        def class_loss_regr_fixed_num(y_true, y_pred):
            # 这里y_true是分两部分，前面的 0~4*3 表示是否是bg，每四位一组，用0和1表示
            # y_true 4*3:end为第二部分，表示预测框坐标
            x = y_true[:, :, 4 * num_classes:] - y_pred
            x_abs = K.abs(x)
            x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
            return self.lambda_cls_regr * K.sum(
                y_true[:, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
                self.epsilon + y_true[:, :, :4 * num_classes])

        return class_loss_regr_fixed_num

    '''
    总网络的分类loss
    '''
    def class_loss_cls(self,y_true, y_pred):
        return self.lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))

    '''
    训练模型
    '''
    def train_net(self,record_df,r_curr_loss):
        '''
        
        :param record_df: 
        :param r_curr_loss: 
        :return: 
        '''
        '''
        设置训练参数
        '''
        # Training setting
        total_epochs = len(record_df)
        r_epochs = len(record_df)

        epoch_length = 1000
        num_epochs = 40
        iter_num = 0

        total_epochs += num_epochs

        losses = np.zeros((epoch_length, 5))
        rpn_accuracy_rpn_monitor = []
        rpn_accuracy_for_epoch = []

        if len(record_df) == 0:
            best_loss = np.Inf
        else:
            best_loss = np.min(r_curr_loss)


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

    def __init__(self, pool_size, num_rois, **kwargs):
        self.dim_ordering = K.image_dim_ordering()
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert (len(x) == 2)

        # x[0] is image with shape (rows, cols, channels)
        img = x[0]

        # x[1] is roi with shape (num_rois,4) with ordering (x,y,w,h)
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # Resized roi of the image to pooling size (7x7)
            rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)

        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        # Might be (1, 4, 7, 7, 3)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        # permute_dimensions is similar to transpose
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def rpn_to_roi(rpn_layer, regr_layer, C, dim_ordering, use_regr=True, max_boxes=300, overlap_thresh=0.9):
    """Convert rpn layer to roi bboxes

    Args: (num_anchors = 9)
        rpn_layer: output layer for rpn classification 
            shape (1, feature_map.height, feature_map.width, num_anchors)
            Might be (1, 18, 25, 18) if resized image is 400 width and 300
        regr_layer: output layer for rpn regression
            shape (1, feature_map.height, feature_map.width, num_anchors)
            Might be (1, 18, 25, 72) if resized image is 400 width and 300
        C: config
        use_regr: Wether to use bboxes regression in rpn
        max_boxes: max bboxes number for non-max-suppression (NMS)
        overlap_thresh: If iou in NMS is larger than this threshold, drop the box

    Returns:
        result: boxes from non-max-suppression (shape=(300, 4))
            boxes: coordinates for bboxes (on the feature map)
    """
    regr_layer = regr_layer / C.std_scaling

    anchor_sizes = C.anchor_box_scales  # (3 in here)
    anchor_ratios = C.anchor_box_ratios  # (3 in here)

    assert rpn_layer.shape[0] == 1

    (rows, cols) = rpn_layer.shape[1:3]

    curr_layer = 0

    # A.shape = (4, feature_map.height, feature_map.width, num_anchors)
    # Might be (4, 18, 25, 18) if resized image is 400 width and 300
    # A is the coordinates for 9 anchors for every point in the feature map
    # => all 18x25x9=4050 anchors cooridnates
    A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:
            # anchor_x = (128 * 1) / 16 = 8  => width of current anchor
            # anchor_y = (128 * 2) / 16 = 16 => height of current anchor
            anchor_x = (anchor_size * anchor_ratio[0]) / C.rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1]) / C.rpn_stride

            # curr_layer: 0~8 (9 anchors)
            # the Kth anchor of all position in the feature map (9th in total)
            regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]  # shape => (18, 25, 4)
            regr = np.transpose(regr, (2, 0, 1))  # shape => (4, 18, 25)

            # Create 18x25 mesh grid
            # For every point in x, there are all the y points and vice versa
            # X.shape = (18, 25)
            # Y.shape = (18, 25)
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

            # Calculate anchor position and size for each feature map point
            A[0, :, :, curr_layer] = X - anchor_x / 2  # Top left x coordinate
            A[1, :, :, curr_layer] = Y - anchor_y / 2  # Top left y coordinate
            A[2, :, :, curr_layer] = anchor_x  # width of current anchor
            A[3, :, :, curr_layer] = anchor_y  # height of current anchor

            # Apply regression to x, y, w and h if there is rpn regression layer
            if use_regr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

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

    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))  # shape=(4050, 4)
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))  # shape=(4050,)

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    # Find out the bboxes which is illegal and delete them from bboxes list
    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

    # Apply non_max_suppression
    # Only extract the bboxes. Don't need rpn probs in the later process
    result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

    return result

if __name__ == '__main__':
    my_config = my_cfg.My_Config()
    my_config.classes_count={'Car': 2383, 'Mobile phone': 1108, 'Person': 3745, 'bg': 0}
    my_config.class_mapping={'Car': 0, 'Mobile phone': 1, 'Person': 2, 'bg':3 }

    Network(my_config)







