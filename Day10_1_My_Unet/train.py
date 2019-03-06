import tensorflow as tf
import tensorflow.contrib.slim as slim
import config as cfg
import data_prepare
import model
import cv2
import os
import numpy as np

class Solver(object):

    def __init__(self,net,data):
        self.net = net
        self.data = data # 传进来的是一个字符串，表示数据的位置

        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS # 这个和总的训练步数相同吗？(好像不用啊）
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE

        # 一些保存的内容暂时先放一下
        self.saver = tf.train.Saver(max_to_keep=3)# 默认保存所有变量，最多保存3个文件
        self.ckpt_file=os.path.join(cfg.SAVE_FILE,'unet.ckpt')

        self.summray_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(cfg.SUMMARY_DIR,flush_secs=30)
        ######

        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.train.exponential_decay(learning_rate=self.initial_learning_rate,
                                                        global_step=self.global_step,
                                                        decay_steps=self.decay_steps,
                                                        decay_rate=self.decay_rate,
                                                        staircase=self.staircase,
                                                        name='learning_rate')

        # 这里用GD是因为上面的learning_rate的下降方法和adam的重复了嘛？
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.train_op = slim.learning.create_train_op(self.net.total_loss,
                                                      self.optimizer,
                                                      global_step=self.global_step)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


        self.writer.add_graph(self.sess.graph)

    '''
    直接读取tfrecords文件
    '''

    def read_and_parse(self,filename):
        filename_queue = tf.train.string_input_producer([filename])

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features_dict = {
            'label': tf.FixedLenFeature([], tf.string),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }

        features_res = tf.parse_single_example(serialized_example, features=features_dict)
        p_tf_org_img = tf.decode_raw(features_res['img_raw'], tf.uint8)
        p_tf_org_img = tf.reshape(p_tf_org_img, [cfg.IMG_SIZE, cfg.IMG_SIZE, 3])

        p_tf_lab_img = tf.decode_raw(features_res['label'], tf.uint8)
        p_tf_lab_img = tf.reshape(p_tf_lab_img, [cfg.IMG_SIZE, cfg.IMG_SIZE, 1])

        return p_tf_org_img, p_tf_lab_img

    def train(self):
        # tf_records文件的placeholder
        file_names = tf.placeholder(dtype=tf.string, shape=[None])

        dataset = tf.data.TFRecordDataset(file_names)

        # 这里是调用映射函数的地方…但不知道能不能用自己的方法啊…
        # 是不是需要将自己的方法封装成tensorflow的方法？
        dataset = dataset.map(data_prepare.parser)
        dataset = dataset.repeat(cfg.REPEAT_TIME)
        dataset = dataset.batch(2)

        # 拿到迭代器
        iterator_init = dataset.make_initializable_iterator()
        # 拿到数据
        images, labels = iterator_init.get_next()

        # 拿到验证数据
        val_img,val_label = self.read_and_parse('./valid_palm.tfrecords')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # valid_img_list = np.asarray([])
            # valid_lab_list = np.asarray([])

            # 暂时理解为控制多个线程同时结束的方法
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            '''
            拿到测试数据
            '''
            for i in range(cfg.VALID_NUMS):
                temp_img, temp_lab = sess.run([val_img, val_label])
                temp_img = np.reshape(temp_img,newshape=[1,cfg.IMG_SIZE,cfg.IMG_SIZE,3])
                temp_lab = np.reshape(temp_lab,newshape=[1,cfg.IMG_SIZE,cfg.IMG_SIZE,1])
                if i==0:
                    valid_img_list = temp_img
                    valid_lab_list = temp_lab
                else:
                    valid_img_list = np.append(valid_img_list,temp_img,axis=0)
                    valid_lab_list = np.append(valid_lab_list,temp_lab,axis=0)

            valid_img_list = valid_img_list/255
            # print(np.max(valid_img_list),valid_img_list.shape)
            # print(np.max(valid_lab_list),valid_lab_list.shape)

            coord.request_stop()
            coord.join(threads)


            # 显式初始化
            train_filenames = ['./train_palm.tfrecords']  # 这里是放self.dataset的地方
            sess.run(iterator_init.initializer, feed_dict={file_names: train_filenames})
            for i in range(cfg.MAX_ITER):
                train_img, train_label = sess.run([images, labels])
                train_img, train_label = data_prepare.img_aug(train_img, train_label)

                # print(np.max(train_img),train_img.shape)
                # print(np.max(train_label),train_label.shape)
                #
                # data_prepare.show_img(train_img[0,:,:,:])
                # data_prepare.show_img(train_label[0,:,:,0]*255)
                # break
                train_img = train_img/255

                feed_dict = {self.net.input_tensor:train_img,
                             self.net.labels:train_label}
                tt_loss,_ = self.sess.run([self.net.total_loss,
                                            self.train_op],feed_dict=feed_dict)

                predict_res = self.sess.run([self.net.logits],
                                            feed_dict={self.net.input_tensor: train_img})
                for step, pre_img in enumerate(predict_res[0]):
                    pre_img = data_prepare.predict_to_img(pre_img)
                    cv2.imwrite('./predict_res/' + str(i) + '_' + str(step) + '.png', pre_img)
                for step,llaabb in enumerate(train_label):
                    cv2.imwrite('./predict_res/' + str(i) + '_' + str(step) + 'label.png', llaabb*255)


                valid_dict = {self.net.input_tensor: valid_img_list,
                              self.net.labels: valid_lab_list}
                # pred_valid = self.sess.run([self.net.logits],feed_dict=valid_dict)
                accuracy = self.sess.run([self.net.accuracy], feed_dict=valid_dict)
                # print(accuracy)

                print('step '+str(i)+' loss is '+str(tt_loss),'acc is '+str(accuracy))


                if i%cfg.SAVE_ITER ==0:
                    self.saver.save(self.sess,
                                    save_path=self.ckpt_file,
                                    global_step=self.global_step)
                    # predict_res = self.sess.run([self.net.logits],
                    #                             feed_dict={self.net.input_tensor:valid_img_list})
                    # cv2.imwrite('./predict_res/'+str(i)+'_trainlabel.png',train_label[0,:,:,0]*255)
                    # for step,pre_img in enumerate(predict_res[0]):
                    #     pre_img = data_prepare.predict_to_img(pre_img)
                    #     cv2.imwrite('./predict_res/'+str(i)+'_'+str(step)+'.png',pre_img)


                # 为啥要这样主动保存
                # 不是会随时更新的嘛= -
                if i%cfg.SUMMARY_ITER==0:
                    summary_str,_ = sess.run([self.summray_op,self.train_op],feed_dict=feed_dict)
                    self.writer.add_summary(summary_str,i)

if __name__ == '__main__':
    '''
    生成网络
    '''
    my_unet = model.Unet(True)
    '''
       调用两个方法，生成对应的tf_records文件
       '''
    train_raw_data, valid_raw_data = data_prepare.devide_all_data()
    data_prepare.to_tfrecords(train_raw_data, 'train_palm.tfrecords')
    data_prepare.to_tfrecords(valid_raw_data, 'valid_palm.tfrecords')
    '''
    定义solver
    '''
    unet_solver = Solver(my_unet,'./train_palm.tfrecords')

    '''
    训练
    '''
    unet_solver.train()