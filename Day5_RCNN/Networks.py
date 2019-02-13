import os
import config as cfg
from sklearn import svm
import tensorflow as tf
from sklearn.externals import joblib
import tensorflow.contrib.slim as slim
# TODO 这个和tf.nn有什么区别？？
from tensorflow.python.ops import nn_ops
from tflearn.layers.normalization import local_response_normalization

class Alexnet_Net:

    def __init__(self,is_training=True, is_fineturn=False,is_SVM=False):
        '''
        下面这三种状态会带来这么几种组合：
            1、is_training=True,is_fineturn=False,is_SVM=False:
                表示在训练特征提取器阶段。
            2、is_training=True,is_fineturn=True,is_SVM=False:
                表示在fine_turn特征提取器阶段
            3、is_training=False,is_fineturn=True,is_SVM=True:
                表示在用fine_turn好的模型，为SVM做特征向量提取阶段
            
        :param is_training: 设置是否需要训练
        :param is_fineturn: 设置是否属于fineturn阶段，主要影响最后一层fc的输出多少；
                             同时也影响了batch_size的大小，fine-turn的batch_size会大一些？
        :param is_SVM: 设置是否属于SVM提取特征的阶段
        '''
        '''
        1、定义一些训练和fine-turn阶段都用得到的参数
        '''
        self.image_size = cfg.Image_size
        self.batch_size = cfg.F_batch_size if is_fineturn else cfg.T_batch_size
        self.class_num = cfg.F_class_num if is_fineturn else cfg.T_class_num
        self.input_data = tf.placeholder(dtype=tf.float32,shape=[None,self.image_size,self.image_size,3],name='input')

        '''
        2、根据是否是训练阶段，是否需要作为特征提取阶段，来构建模型
        '''
        self.logits = self.build_network(self.input_data,self.class_num,is_svm=is_SVM,is_training=is_training)

        '''
        3、如果是训练阶段，那么再定义训练用label，loss，accuracy等内容
        '''
        if is_training == True:
            self.label = tf.placeholder(tf.float32,[None,self.class_num],name='label')
            self.loss_layer(self.logits,self.label)  # define a function to calculate the loss
            self.accuracy = self.get_accuracy(self.logits,self.label)  # define a function to calculate the accuracy
            '''
            this adds any losses you have added with `tf.add_loss()`
            需要和tf.losses.add_loss联合起来用。
            '''
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss',self.total_loss)

    def build_network(self,input,output,is_svm=False,scope='R-CNN',is_training=True,keep_prob = 0.5):

        with tf.variable_scope(scope):
            # 定义fully_connected和conv2d的共同参数，有：激活函数，初始化方法，正则化方法
            with slim.arg_scope([slim.fully_connected,slim.conv2d],
                                activation_fn = tf.nn.relu,
                                weights_initializer = tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
                                weights_regularizer = slim.l2_regularizer(scale=0.0005)):
                net = slim.conv2d(input,num_outputs=96,kernel_size=11,stride=4,scope='conv_1')
                net = slim.max_pool2d(net,kernel_size=3,stride=2,scope='pool_2')
                # tf.nn.lrn和tflearn的lrn默认参数不同，这里用tflearn的
                net = tf.nn.lrn(net,depth_radius=5,bias=1,alpha=0.0001,beta=0.75)

                net = slim.conv2d(net,256,5,stride=1,scope='conv_3')
                '''
                tensorboard路径内不能有中文
                这里只是单纯写错了吧…没有设置reuse=True的话，
                名字相同会自动帮你重命名的，比如后面加一个 _1啥的
                '''
                net = slim.max_pool2d(net,kernel_size=3,stride=2,scope='pool_2')
                net = tf.nn.lrn(net,depth_radius=5,bias=1,alpha=0.0001,beta=0.75)

                net = slim.conv2d(net,384,3,scope='conv_4')
                net = slim.conv2d(net,384,3,scope='conv_5')
                net = slim.conv2d(net,256,3,scope='conv_6')
                net = slim.max_pool2d(net,3,stride=2,scope='pool_7')
                net = tf.nn.lrn(net,5,1,0.0001,0.75)

                net = slim.flatten(net,scope='flat_32')
                net = slim.fully_connected(net, 4096,activation_fn=tf.nn.tanh,scope='fc_8')
                # 设置了是否是is_training之后，在预测阶段就不会用dropout，
                # 所以不用手动将keep_prob设置为1了
                net = slim.dropout(net,keep_prob=keep_prob,is_training=is_training,scope='dropout9')

                net = slim.fully_connected(net,4096,activation_fn=tf.nn.tanh,scope='fc_10')
                if is_svm:
                    return net
                net = slim.dropout(net,keep_prob=keep_prob,is_training=is_training,scope='dropout11')
                # 如果是要计算cross_entropy的话，不建议将softmax和cross_entropy分开
                # 会有数值问题的。
                # 方法默认的是tf.nn.relu，我们可能需要设置为None，就采用linear_func，
                # 然后用tf.softmax_with_cross_entropy来进行操作。
                #net = slim.fully_connected(net,output,activation_fn=self.softmax(),scope='fc_11')
                net = slim.fully_connected(net,output,activation_fn=None,scope='fc_11')

        return net

    def loss_layer(self,y_pred,y_true):
        """
        这里注意，我们就不要自己算了，直接用它们的方法
        :param y_pred: 
        :param y_true: 
        :return: 
        """
        '''
        https://www.cnblogs.com/guoyaohua/p/8081192.html
        name_scope用于关于op的定义，variable_scope用于关于variable的定义
        所以在封装的比较好的现在，一般用name_scope会好一点？
        '''
        with tf.name_scope('Crossentropy'):
            # 注意，返回的内容是：A 1-D `Tensor` of length `batch_size`
            # 所以还需要手工对它进行tf.reduce_mean的操作
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred)
            cross_entropy = tf.reduce_mean(cross_entropy)
            tf.losses.add_loss(cross_entropy)
            tf.summary.scalar('loss',cross_entropy)

    def get_accuracy(self,y_pred,y_true):
        """
        虽然net的输出没有softmax，但是由于softmax不改变相对大小，所以没差
        :param y_pred: 
        :param y_true: 
        :return: 
        """
        ''' 
         后面那个1表示在第几维上找，
         由于y_pred和y_true都是[batch,num_class]这样的，我们要在num_class这个维度上找
         所以才有后面的1
        '''
        y_pred_max = tf.argmax(y_pred,1)
        y_true_max = tf.argmax(y_true,1)
        correc_pre = tf.equal(y_pred_max,y_true_max) # 返回true or false
        accuracy = tf.reduce_mean(tf.cast(correc_pre,tf.float32))
        return accuracy

class SVM:
    def __init__(self,data):
        self.data = data
        self.data_save_path = cfg.SVM_and_Reg_save
        self.output = cfg.Out_put

    def train(self):
        svms=[]
        data_dirs = os.listdir(self.data_save_path)
        for data_dir in data_dirs: # 这里data_dir也就代表label了
            images, labels = self.data.get_SVM_data(data_dir)
            clf = svm.LinearSVC()
            clf.fit(images,labels)
            svms.append(clf)
            SVM_model_path = os.path.join(self.output, 'SVM_model')
            if not os.path.exists(SVM_model_path):
                os.makedirs(SVM_model_path)
            '''
            这个是sklearn的模型保存方法
            '''
            joblib.dump(clf, os.path.join(SVM_model_path, str(data_dir) + '_svm.pkl'))

class Reg_Net(object):
    def __init__(self,is_training=True):
        self.output_num = cfg.R_class_num
        self.input_data = tf.placeholder(dtype=tf.float32,shape=[None,4096],name='input')
        self.logits = self.build_network(self.input_data,self.output_num,is_training=is_training)

        if is_training:
            self.label = tf.placeholder(dtype=tf.float32,shape=[None,self.output_num],name='input_label')
            self.loss_layer(self.logits,self.label)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss',self.total_loss)

    def build_network(self,input_image,output_num,is_training=True,scope='regression_box',keep_prob=0.5):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.fully_connected],
                                 activation_fn = tf.nn.tanh,
                                 weights_initializer = tf.truncated_normal_initializer(0.0,0.01),
                                 weights_regularizer = slim.l2_regularizer(0.0005)):
                net = slim.fully_connected(input_image,4096,scope = 'fc_1')
                net = slim.dropout(net,keep_prob=keep_prob,is_training=is_training,scope='dropout')
                net = slim.fully_connected(net,output_num,scope='fc_2')

                return net

    def loss_layer(self,y_pred,y_true):
        '''
        感觉这里是精髓
        :param y_pred: [label,四个坐标偏移量] 
        :param y_true: [label,四个坐标偏移量]
        :return: 
        '''
        '''
        1、我认为这里设置的有问题啊
            总的思路，这里的想法是 边框回归loss+前/背景loss之和。
        2、那么边框回归采用的是MSE，就是  tf.reduce_sum(tf.square(y_true[:,1:5]-y_pred[:,1:5]),1)
        3、问题出在前/背景 loss。这显然是一个二分类问题，所以无论输入还是输入，都应该是2才对啊（对应做了one-hot）
        4、不然，网络推断的第一位输出，是一个没有范围的值，100,1000都可能，而真值是概率值0，和1.
        5、只有一位输出的话，没办法做softmax…
        6、所以这里是有问题的啊…
        
        7、当然，不用两位输出也可以，最后一层那么就用sigmoid将输出缩放到0~1之间，
        8、然后再用cross_entropy来计算啊tf.nn.sigmoid_cross_entropy_with_logits
        
        '''

        no_object_loss = tf.reduce_mean(tf.square((1-y_true[:,0])*y_pred[:,0]))
        object_loss = tf.reduce_mean(tf.square(y_true[:,0]*(y_pred[:,0]-1)))

        loss = (tf.reduce_mean(y_true[:,0] * ( tf.reduce_sum(tf.square(y_true[:,1:5]-y_pred[:,1:5])
                                                             ,1) ) )
                + no_object_loss
                + object_loss
                )

        tf.losses.add_loss(loss)
        tf.summary.scalar('loss',loss)



