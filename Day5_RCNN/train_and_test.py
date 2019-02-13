#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import Networks
import numpy as np
import process_data
import config as cfg
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.externals import joblib
# slim = tf.contrib.slim
flower = {1:'pancy',2:'Tulip'}

class Solver:
    '''
    为了能够用一个框架去执行不同的模型，那么要求模型在定义的时候有：
        1、统一的input_data,input_label输入。
        2、统一的loss输出。
        统一，是指定义的时候用的名字一样就好。
    
    该Solver用来处理四个过程：
        1、原始特征提取器训练过程，类似于用image_net来训练。
        2、finetune过程，在自己的数据集上进行fine-tune
        3、回归过程。
        4、预测过程。
        
    
    '''
    def __init__(self,net,data,is_training=False,is_fineturn=False,is_Reg = False):
        '''
        吐槽一下，是fine-tune，而不是fine-turn
        :param net: 训练网络
        :param data: 训练数据
        :param is_training: 是否是训练过程
        :param is_fineturn: 是否是fine-turn过程
        :param is_Reg: 是否是回归过程
        '''
        self.net = net
        self.data = data
        self.is_Reg = is_Reg
        self.is_fineturn = is_fineturn

        '''
        定义训练用参数
        '''
        self.summary_step = cfg.Summary_iter # 多久更新一些tensorboard依赖参数
        self.save_step = cfg.Save_iter # 多久保存一次模型
        self.max_iter = cfg.Max_iter # 最高迭代次数
        self.staircase = cfg.Staircase # 控制learning-rate的更新是每一步都更新还是若干步更新一次。

        '''
        定义fine-turn是用的参数
        '''
        if is_fineturn:
            # fine-tune过程
            self.initial_learning_rate = cfg.F_learning_rate
            self.decay_step = cfg.F_decay_iter
            self.decay_rate = cfg.F_decay_rate
            self.weight_file = cfg.T_weights_file # 在fineturn的时候，用的是训练时的weights
            self.output_dir = r'./output/fineturn' # 这个为什么不放到cfg里面啊…
        elif is_Reg:
            # 回归过程
            self.initial_learning_rate = cfg.R_learning_rate
            self.decay_step = cfg.R_decay_iter
            self.decay_rate = cfg.R_decay_rate
            # 区分是训练还是预测
            if is_training:
                self.weight_file = None
            else:
                self.weight_file = cfg.R_weights_file
            self.output_dir = r'./output/Reg_box'
        else:
            self.initial_learning_rate = cfg.T_learning_rate
            self.decay_step = cfg.T_decay_iter
            self.decay_rate = cfg.T_decay_rate
            if is_training: # 训练过程
                self.weight_file = None
            else: # 预测过程，用的是fine-turn好的weight
                self.weight_file = cfg.F_weights_file
            self.output_dir = r'./output/train_alexnet'

        # 自定义的方法，保存上面说的那些参数


        '''
        加载参数
        '''
        exclude = ['R-CNN/fc_11']
        # 加载除了exclude之外的参数
        self.variable_to_restore =slim.get_variables_to_restore(exclude=exclude)
        self.restorer = tf.train.Saver(self.variable_to_restore,max_to_keep=1)
        # 加载所有参数
        self.variable_to_save = slim.get_variables_to_restore(exclude=[])
        self.saver = tf.train.Saver(self.variable_to_save,max_to_keep=1)
        # 模型保存的名字
        self.ckpt_file = os.path.join(self.output_dir,'save.ckpt')

        # 和tensorboard有关，保存后可以通过tensorboard来查看
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir) # 之后可以用add_graph()加入图

        # 通过get_variable来获取variable
        # 这里的global_step就是一个计数器，记录现在训练到第几步了
        # 所以感觉用下面的也能定义
        # self.global = tf.Variable(name='global_step',dtype=tf.int32,value=0,trainable=Flase)
        self.global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)
        # 学习率指数衰减
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate,
            self.global_step,
            self.decay_step,
            self.decay_rate,
            self.staircase,
            name='learning_rate'
        )

        '''
        如果是训练过程，需要做好这些准备
        '''
        if is_training:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
                                    .minimize(self.net.total_loss,global_step=self.global_step)

            # 用滑动平均，帮助预测的时候更准
            # 1、得到滑动平均操作符
            self.ema = tf.train.ExponentialMovingAverage(decay=0.99)
            # 2、选定要滑动平均的对象
            self.average_op = self.ema.apply(tf.trainable_variables())
            # 3、指定顺序，绑定操作符
            with tf.control_dependencies([self.optimizer]):
                self.train_op = tf.group(self.average_op)
        '''
        加载运行所需的上下文对象
        '''
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        '''
        视情况加载模型
        '''
        if self.weight_file is not None:
            self.restorer.restore(self.sess,self.weight_file)

        '''
        将模型信息写到tensorboard里面
        '''
        self.writer.add_graph(self.sess.graph)

    def save_cfg(self):
        '''
        保存模型的一些超参数
        :return: 
        '''
        with open(os.path.join(self.output_dir,'config.txt'),'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

    def train(self):
        '''
        训练模型
        :return: 
        '''
        for step in range(1,self.max_iter+1):
            '''
            拿到对应数据
            '''
            if self.is_Reg:
                input,labels = self.data.get_Reg_batch()
            elif self.is_fineturn:
                input,labels = self.data.get_fineturn_batch()
            else:
                input,labels = self.data.get_batch()

            '''
            开始训练
            '''
            feed_dict = {self.net.input_data:input,self.net.label:labels}
            print('Step %d starts' % step)
            if step % self.summary_step ==0:
                print("summary_step!!"+str(step))
                summary,loss,_ = self.sess.run([self.summary_op,self.net.total_loss,self.train_op],
                                               feed_dict=feed_dict)
                self.writer.add_summary(summary,step)
                print("Data_epoch:" + str(self.data.epoch) + " " * 5 + "training_step:" + str(
                    step) + " " * 5 + "batch_loss:" + str(loss))
            else:
                print("train_step!!"+str(step))
                self.sess.run([self.train_op],feed_dict = feed_dict)
            if step % self.save_step == 0:
                print("saving_step!!"+str(step))
                print("saving the model into " + self.ckpt_file)
                # TODO
                # 输出结果是这样 ./output/train_alexnet\save.ckpt-5
                # 那个就是后面restore需要用的参数。
                # 即，模型文件相同的部分
                save_res = self.saver.save(self.sess,self.ckpt_file,global_step=self.global_step)
                print(save_res)

    def predict(self,input_data):
        feed_dict={self.net.input_data:input_data}
        predict_result = self.sess.run(self.net.logits,feed_dict=feed_dict)
        return predict_result

def get_Solvers():
    '''
    这里是用来获取：
        1、特征提取器；2、SVM分类器；3、Reg坐标回归器的地方
    我们先来试试特征提取器
    :return: 
    '''
    '''
    创建文件夹
    '''
    weight_outputs = ['train_alexnet','fineturn','SVM_model','Reg_box']
    for weight_output in weight_outputs:
        output_path = os.path.join(cfg.Out_put,weight_output)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    '''
    1、先来训练alexnet
        1.1 通过模型文件夹来判断之前是否有训练好的模型，
            文件夹为空那么我就训练一波
    '''
    if len(os.listdir(r'./output/train_alexnet'))==0:
        Train_alexnet = tf.Graph()
        with Train_alexnet.as_default():
            Train_alexnet_data = process_data.Train_Alexnet_Data()
            Train_alexnet_net = Networks.Alexnet_Net(is_training=True,is_fineturn=False,is_SVM = False)
            Train_alexnet_solver = Solver(Train_alexnet_net,Train_alexnet_data,is_training=True,is_fineturn=False,is_Reg=False)
            Train_alexnet_solver.train()

    '''
    2、然后用自己的数据集fine-turn
    '''
    # TODO 这个r''是什么意思？

    if len(os.listdir(r'./output/fineturn')) == 0:
        Fineturn = tf.Graph() #图分开有什么好处,参数不会重叠？
        with Fineturn.as_default():
            Fineturn_data = process_data.FineTurn_And_Predict_Data()
            Fineturn_net = Networks.Alexnet_Net(is_training=True,is_fineturn=True,is_SVM=False)
            Fineturn_solver = Solver(Fineturn_net,Fineturn_data,is_training=True,is_fineturn=True,is_Reg=False)
            Fineturn_solver.train()

    '''
    3、拿到碎片的feature-vector，准备训练svm
    '''
    Features = tf.Graph()
    with Features.as_default():
        '''
        3.1 准备SVM训练数据阶段
        '''
        Features_net = Networks.Alexnet_Net(is_training=False,is_fineturn=True,is_SVM=True)
        Features_solver = Solver(Features_net,None,is_training=False,is_fineturn=True,is_Reg=False)
        # 保存的数据是为三种格式：
        #  1、图像碎片的feature_vector；2、图像的数字label；3、图像的4个ground_truth+前背景分类
        # 保存完了加载数据：
        #
        Features_data = process_data.FineTurn_And_Predict_Data(Features_solver,is_svm=True,is_save=True)

        '''
        3.2 训练SVM模型
        问题：SVM模型是和类别数相同的，这个在训练过程中在哪里体现了。
        '''
    svms = []
    if len(os.listdir(r'./output/SVM_model')) == 0:
        SVM_net = Networks.SVM(Features_data)
        SVM_net.train()

    for file in os.listdir(r'./output/SVM_model'):
        svms.append(joblib.load(os.path.join('./output/SVM_model',file)))


    '''
    4、训练回归模型
    '''
    Reg_box = tf.Graph()
    with Reg_box.as_default():
        Reg_box_data = Features_data
        Reg_box_net  =Networks.Reg_Net(is_training = True)
        if len(os.listdir(r'./output/Reg_box'))==0:
            Reg_box_solver = Solver(Reg_box_net,Reg_box_data,is_training=True,is_fineturn=False,is_Reg=True)
            Reg_box_solver.train()
        else:
            Reg_box_solver = Solver(Reg_box_net,Reg_box_data,is_training=False,is_fineturn=False,is_Reg=True)


    # return Fineturn_solver,svms,Reg_box_solver



if __name__ == '__main__':

    Train_alexnet_solver = get_Solvers()
    # process_data.FineTurn_And_Predict_Data()