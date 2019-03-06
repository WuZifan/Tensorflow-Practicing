import tensorflow as tf
import tensorflow.contrib.slim as slim

class Unet(object):

    def __init__(self,is_training=True):
        # 这里可以试试灰度图
        # 即，用256,256,1的图
        self.is_training = is_training
        self.input_tensor = tf.placeholder(dtype=tf.float32,shape=[None,256,256,3],name='input_img')
        self.logits = self.build_net(self.input_tensor)
        # self.predict_output = tf.nn.softmax(self.logits)

        if self.is_training:
            # self.labels = tf.placeholder(dtype=tf.float32,shape=[None,256,256,2])
            self.labels = tf.placeholder(dtype=tf.float32,shape=[None,256,256,1])
            # self.loss_layer(self.logits,self.labels)

            self.loss_layer2(self.logits,self.labels)

            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss',self.total_loss)

            self.accuracy = self.cal_accuracy2(self.logits,self.labels)


    def loss_layer(self,logits,labels):
        '''
        计算输出是256,256,2的accuracy
        :param logits: 
        :param labels: 
        :return: 
        '''
        flat_logits = tf.reshape(logits,[-1,2])
        flat_labels = tf.reshape(labels,[-1,2])

        binary_loss = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,labels=flat_labels)
        binary_loss = tf.reduce_mean(binary_loss)
        tf.losses.add_loss(binary_loss)

        tf.summary.scalar('total_loss',binary_loss)

    def loss_layer2(self,logits,labels):
        '''
        计算输出是256,256,1的loss
        :param logits: 
        :param labels: 
        :return: 
        '''
        flat_logits = tf.reshape(logits,[-1])
        flat_logits = tf.clip_by_value(flat_logits,1e-10,0.999999)

        flat_labels = tf.reshape(labels,[-1])

        binary_loss = -(3*flat_labels*tf.log(flat_logits)+
                        (1-flat_labels)*tf.log(1-flat_logits))

        # binary_loss = flat_logits-flat_labels

        binary_loss = tf.reduce_mean(binary_loss)
        tf.losses.add_loss(binary_loss)
        tf.summary.scalar('total_loss',binary_loss)

    def cal_accuracy(self,logits,labels):
        '''
        计算输出是256,256,2的accuracy
        :param logits: 
        :param labels: 
        :return: 
        '''
        softmaxed_pred = tf.nn.softmax(logits)
        softmaxed_pred = tf.reshape(softmaxed_pred,[-1,2])
        labels = tf.reshape(labels,[-1,2])
        accuracy = tf.equal(tf.argmax(softmaxed_pred,1),tf.argmax(labels,1))

        accuracy = tf.reduce_mean(tf.cast(accuracy,tf.float32))
        return accuracy

    def cal_accuracy2(self,logits,labels):
        '''
        计算输出是256,256,1的accuracy
        :param logits: 
        :param labels: 
        :return: 
        '''
        temp_logits = logits>0.5
        temp_logits = tf.cast(temp_logits,dtype=tf.float32)
        temp_logits = tf.reshape(temp_logits,[-1])

        temp_labels = tf.reshape(labels,[-1])

        accuracy = tf.equal(temp_logits,temp_labels)
        accuracy = tf.reduce_mean(tf.cast(accuracy,tf.float32))
        return accuracy

    def build_net(self,input_tensor):
        # input_tensor = tf.placeholder(shape=[None, 256, 256, 1], dtype=tf.float32)

        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            activation_fn=tf.nn.relu):
            with tf.variable_scope('down_1'):
                d1_conv1 = slim.conv2d(input_tensor, num_outputs=64, kernel_size=3)
                d1_conv2 = slim.conv2d(d1_conv1, num_outputs=64, kernel_size=3)
                d1_max1 = slim.max_pool2d(d1_conv2, kernel_size=2, stride=2)

            with tf.variable_scope('down_2'):
                d2_conv1 = slim.conv2d(d1_max1, num_outputs=128, kernel_size=3)
                d2_conv2 = slim.conv2d(d2_conv1, num_outputs=128, kernel_size=3)
                d2_max1 = slim.max_pool2d(d2_conv2, kernel_size=2, stride=2)

            with tf.variable_scope('down_3'):
                d3_conv1 = slim.conv2d(d2_max1, num_outputs=256, kernel_size=3)
                d3_conv2 = slim.conv2d(d3_conv1, num_outputs=256, kernel_size=3)
                d3_max1 = slim.max_pool2d(d3_conv2, kernel_size=2, stride=2)

            with tf.variable_scope('down_4'):
                d4_conv1 = slim.conv2d(d3_max1, num_outputs=512, kernel_size=3)
                d4_conv2 = slim.conv2d(d4_conv1, num_outputs=512, kernel_size=3)
                d4_drop1 = slim.dropout(d4_conv2,keep_prob=0.5,is_training=self.is_training)
                d4_max1 = slim.max_pool2d(d4_drop1, kernel_size=2, stride=2)

            with tf.variable_scope('down_5'):
                d5_conv1 = slim.conv2d(d4_max1, num_outputs=1024, kernel_size=3)
                d5_conv2 = slim.conv2d(d5_conv1, num_outputs=1024, kernel_size=3)
                d5_drop1 = slim.dropout(d5_conv2,keep_prob=0.5,is_training=self.is_training)

            with tf.variable_scope('up_1'):
                up1_h = d5_drop1.get_shape().as_list()[1] * 2
                up1_w = d5_drop1.get_shape().as_list()[2] * 2

                up1 = self.my_conv2_transpose2(input_tensor=d5_drop1,
                                         output_size=(up1_h, up1_w),
                                         output_channel=512,
                                         kernel_size=(3, 3))

                concate_1 = tf.concat([up1, d4_conv2], axis=3)

                u1_conv1 = slim.conv2d(concate_1, num_outputs=512, kernel_size=3)
                u1_conv2 = slim.conv2d(u1_conv1, num_outputs=512, kernel_size=3)

            with tf.variable_scope('up_2'):
                up2_h = u1_conv2.get_shape().as_list()[1] * 2
                up2_w = u1_conv2.get_shape().as_list()[2] * 2

                up2 = self.my_conv2_transpose2(u1_conv2, (up2_h, up2_w), 256, (3, 3))
                concate_2 = tf.concat([d3_conv2, up2], axis=3)

                u2_conv1 = slim.conv2d(concate_2, num_outputs=256, kernel_size=3)
                u2_conv2 = slim.conv2d(u2_conv1, num_outputs=256, kernel_size=3)
                # print(u2_conv2)
            with tf.variable_scope('up_3'):
                up3_h = u2_conv2.get_shape().as_list()[1] * 2
                up3_w = u2_conv2.get_shape().as_list()[2] * 2

                up3 = self.my_conv2_transpose2(u2_conv2, (up3_h, up3_w), 128, (3, 3))
                concate_3 = tf.concat([d2_conv2, up3], axis=3)

                u3_conv1 = slim.conv2d(concate_3, num_outputs=128, kernel_size=3)
                u3_conv2 = slim.conv2d(u3_conv1, num_outputs=128, kernel_size=3)
                # print(u3_conv2)

            with tf.variable_scope('up_4'):
                up4_h = u3_conv2.get_shape().as_list()[1] * 2
                up4_w = u3_conv2.get_shape().as_list()[2] * 2

                up4 = self.my_conv2_transpose2(u3_conv2, (up4_h, up4_w), 64, (3, 3))
                concate_4 = tf.concat([d1_conv2, up4], axis=3)

                u4_conv1 = slim.conv2d(concate_4, num_outputs=64, kernel_size=3)
                u4_conv2 = slim.conv2d(u4_conv1, num_outputs=64, kernel_size=3)
                u4_conv3 = slim.conv2d(u4_conv2, num_outputs=2, kernel_size=3)
                u4_conv4 = slim.conv2d(u4_conv3, num_outputs=1, kernel_size=1,activation_fn = tf.nn.sigmoid)

                # print(u4_conv3)
                # output_tensor = tf.nn.sigmoid(u4_conv4)

        return u4_conv4

    def my_conv2_transpose2(self,input_tensor, output_size, output_channel, kernel_size, strides=[1, 2, 2, 1],padding='SAME'):
        '''
        
        :param input_tensor: 输入tensor
        :param output_size: 没用了
        :param output_channel: 输出的channel数目
        :param kernel_size: 反卷积的卷积核尺寸
        :param strides: 通过步长来控制反卷积结果的维度是原始维度的几倍 
        :param padding: padding方式
        :return: 
        '''
        input_channel_size = input_tensor.get_shape().as_list()[3]
        input_size_h = input_tensor.get_shape().as_list()[1]
        input_size_w = input_tensor.get_shape().as_list()[2]
        stride_shape = strides

        # 根据步长计算卷积之后的维度
        output_size_h = (input_size_h)*stride_shape[1]
        output_size_w = (input_size_w)*stride_shape[2]
        output_shape = tf.stack([tf.shape(input_tensor)[0],
                                 output_size_h,output_size_w,output_channel])

        k_h, k_w = kernel_size
        # 反卷积的w
        trans_w = tf.Variable(tf.truncated_normal(shape=[k_h, k_w,
                                                         output_channel,
                                                         input_channel_size],
                                                  dtype=tf.float32))
        # 反卷积的bias的shape就是一个输出的shape
        trans_b = tf.Variable(tf.constant(value=0.1, shape=[output_channel], dtype=tf.float32))

        upconv = tf.nn.conv2d_transpose(input_tensor,trans_w,output_shape,stride_shape,
                                        padding)
        output = tf.nn.bias_add(upconv,trans_b)
        output = tf.nn.relu(output)
        output = tf.reshape(output,output_shape)

        return output



if __name__ == '__main__':
    my_unet = Unet(True)
