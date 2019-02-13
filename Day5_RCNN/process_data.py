import os
import cv2
import sys
import math
import codecs
import pickle
import skimage
import numpy as np
import config as cfg
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return if_intersect
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter

def IOU(ver1, vertice2):
    '''
    用于计算两个矩形框的IOU
    :param ver1: 第一个矩形框
    :param vertice2: 第二个矩形框
    :return: 两个矩形框的IOU值
    '''
    # TODO 另一种思路实现IOU计算
    '''
    另一种思路：
        1、拿到两个多边形的顶点坐标。
        2、根据两组坐标，分别画出两组坐标的mask。
        3、统计255的点，就是面积；两组mask的交集就是重合面积，简单算一下就知道面积了。
    '''
    vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    if area_inter:
        area_1 = ver1[2] * ver1[3]
        area_2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False

def clip_pic(img, rect):
    '''

    :param img: 输入的图片
    :param rect: rect矩形框的4个参数
    :return: 输入的图片中相对应rect位置的部分 与 矩形框的一对对角点和长宽信息
    '''
    x, y, w, h = rect[0], rect[1], rect[2], rect[3]
    x_1 = x + w
    y_1 = y + h
    return img[y:y_1, x:x_1, :], [x, y, x_1, y_1, w, h]

class Train_Alexnet_Data(object):
    """
     构建一个获取训练数据的类，主要有：
        1、加载原始的训练数据。
        2、对图像数据做resize，对label做one-hot
        3、用pickle的格式将其保存。
        4、提供get_batch方法
    """
    def __init__(self):
        self.train_batch_size = cfg.T_batch_size
        self.image_size = cfg.Image_size

        self.train_list = cfg.Train_list
        self.train_class_num = cfg.T_class_num
        self.flower17_data =[]
        self.data = cfg.DATA
        # 创建文件夹，存放数据
        if not os.path.isdir(self.data):
            os.makedirs(self.data)

        self.epoch = 0
        self.cursor = 0
        self.load_17flowers()

    def load_17flowers(self,save_name = '17flowers.pkl'):
        # pkl 文件的保存路径
        save_path = os.path.join(self.data,save_name)


        if os.path.isfile(save_path):
            # 如果文件存在
            self.flower17_data = pickle.load(open(save_path,'rb'))
        else:
            # 如果文件不存在
            with codecs.open(self.train_list,'r','utf-8') as f:
                lines = f.readlines()
                for num,line in enumerate(lines):
                    context = line.strip().split(' ')
                    image_path = context[0]
                    index = int(context[1])

                    img = cv2.imread(image_path)
                    img = cv2.resize(img,(self.image_size,self.image_size))
                    img = np.asarray(img,dtype=np.float32)

                    label = np.zeros(self.train_class_num)
                    label[index]=1
                    self.flower17_data.append([img,label])
                    # view_bar("Process train_image of %s" % image_path, num + 1, len(lines))
            pickle.dump(self.flower17_data,open(save_path,'wb'))

    def get_batch(self):
        '''
        拿到每一轮的数据
        :return: 
        '''
        images = np.zeros((self.train_batch_size,self.image_size,self.image_size,3))
        labels = np.zeros((self.train_batch_size,self.train_class_num))
        count = 0

        while count <self.train_batch_size:
            # TODO 还有这里的cursor是个啥
            images[count] = self.flower17_data[self.cursor][0]
            labels[count] = self.flower17_data[self.cursor][1]
            count +=1
            self.cursor +=1
            if self.cursor>=len(self.flower17_data):
                self.cursor=0
                self.epoch+=1
                np.random.shuffle(self.flower17_data)
                print(self.epoch)
        return images,labels

class FineTurn_And_Predict_Data(object):
    """
        本类提供以下三个功能所需要的数据：
            1、Fine-turn：【图像碎片，one-hot的label】
            2、SVM：【图像碎片的特征vector，图像的类别】
            3、Regresson：【图像碎片的特征vector，图像ground_truth+label】
    """
    def __init__(self,solver=None,is_svm=False,is_save=True):
        self.solver = solver
        self.is_svm = is_svm
        self.is_save = is_save

        self.fineturn_list = cfg.Finetune_list # 加载fineturn用的数据列表
        self.image_size = cfg.Image_size
        self.F_class_num = cfg.F_class_num # 分类数目
        # 要回归的项目 数目 https://github.com/Liu-Yicheng/R-CNN/issues/1
        # 解释了为什么回归参数是5
        self.R_class_num = cfg.R_class_num

        self.fineturn_batch_size = cfg.F_batch_size
        self.Reg_batch_size = cfg.R_batch_size

        self.fineturn_save_path = cfg.Fineturn_save # 保存路径
        if not os.path.exists(self.fineturn_save_path):
            os.makedirs(self.fineturn_save_path)

        self.SVM_and_Reg_save_path = cfg.SVM_and_Reg_save # 保存路径
        if not os.path.exists(self.SVM_and_Reg_save_path):
            os.makedirs(self.SVM_and_Reg_save_path)

        '''
            由于fineturn,svm和reg的训练都是通过碎片的形式进行的，所以要判断碎片是前景还是背景
            
        '''
        self.fineturn_threshold = cfg.F_fineturn_threshold # 小于threshold是背景，不然就按照label划分
        self.svm_threshold = cfg.F_svm_threshold # 小于threshold是背景，不然就按label划分
        self.reg_threshold = cfg.F_regression_threshold # 小于threshold是背景，其他是前景

        self.SVM_data_dic = {}
        self.Reg_data =[]
        self.fineturn_data = []

        self.cursor = 0
        self.epoch = 0
        print('lala')
        if self.is_svm:
            print('hehe')
            if len(os.listdir(self.SVM_and_Reg_save_path))==0:
                print('haha')
                self.load_2flowers()
        else:
            # fineturn的时候是走这里进函数
            if len(os.listdir(self.fineturn_save_path)) == 0:
                self.load_2flowers()

        self.load_from_npy()

    def load_2flowers(self):
        '''
        加载数据用的
        :return: 
        '''

        '''
        codecs的好处在于能够处理编码问题
        '''
        with codecs.open(self.fineturn_list,'r','utf-8') as f:
            lines = f.readlines()
            # 每行格式：
            # 2flowers/jpg/0/image_0561.jpg 2 90,126,350,434
            for num,line in enumerate(lines):
                # 对于每一张图片
                labels = []
                labels_bbox = []
                images = []
                context = line.strip().split(' ')

                image_path = context[0] # 图片路径
                ref_rect = context[2].split(',')
                ground_truth = [int(i) for i in ref_rect] # 图片中物体位置
                img = cv2.imread(image_path)
                '''
                这里返回值有两个：img_lbl,和regions，由于我们只用的到第二个，所以不管第一个参数。
                regions是一个dict，key有：
                                    rect：碎片的[min_x,min_y,w,h]坐标，
                                    size：是什么mask_pixel/4的值，反正不是面积，
                                    labels：表示这个碎片由哪几个碎片组成
                所以我们一般只要用regions的rect就好。
                '''
                img_lbl,regions = selectivesearch.selective_search(img,scale=500,sigma=0.9,min_size=10)
                candidate = set()
                for r in regions:
                    '''
                    过滤掉一些不要的regions
                    下面这些的过滤条件真的不重复嘛…………
                    '''
                    if r['rect'] in candidate:
                        continue
                    if r['size'] <200: # 这个size到底是个什么鬼,shi 这个felzenszwalb segmentation的size
                        continue
                    if (r['rect'][2]*r['rect'][3])<500: # 真·面积小于500
                        continue
                    # 切割图片
                    proposal_img,proposal_vertice = clip_pic(img,r['rect'])

                    if len(proposal_img) == 0: # 有可能会切不出来嘛？？
                        continue
                    x,y,w,h = r['rect']
                    if w==0 or h ==0: # 前面不都验证过两个乘积要大于500才能往下嘛？？？？
                        continue
                    [a,b,c] = np.shape(proposal_img)
                    if a==0 or b==0 or c==0: # ？？？？
                        continue

                    # 把碎片resize，并加入candidate
                    resized_proposal_img = cv2.resize(proposal_img,(self.image_size,self.image_size))
                    candidate.add(r['rect'])
                    # 把碎片变成np的array数据类型
                    img_float = np.asarray(resized_proposal_img,dtype='float32')

                    '''
                    不同类型的过程，训练用的data不同
                    '''
                    if self.is_svm:
                        '''
                        1、拿到碎片的特征值。
                           过这里是默认了 is_svm=True的时候，solver一定不是None嘛…
                        2、另外，images后面append的两个参数不同步，一个是array，一个是featuresmap呀
                        '''
                        feature = self.solver.predict([img_float])
                        images.append(feature[0])
                    else:
                        images.append(img_float)

                    # 计算真值和碎片框的iou
                    iou_val  = IOU(ground_truth,proposal_vertice)
                    # 计算碎片框的中心点和长宽
                    px = float(proposal_vertice[0]) + float(proposal_vertice[4]/2.0)
                    py = float(proposal_vertice[1]) + float(proposal_vertice[5]/2.0)
                    ph = float(proposal_vertice[5])
                    pw = float(proposal_vertice[4])
                    # ground的中心点和长宽
                    gx = float(ground_truth[0])
                    gy = float(ground_truth[1])
                    gw = float(ground_truth[2])
                    gh = float(ground_truth[3])

                    index = int(context[1]) # 分类的label
                    if self.is_svm:
                        # 准备SVM数据的地方
                        if iou_val<self.svm_threshold:
                            # 表示是一个负样本
                            labels.append(0)
                        else:
                            labels.append(index)

                        # 保存为回归的数据
                        # 计算偏移量
                        label = np.zeros(5)
                        label[1:5] = [(gx-px)/pw,(gy-py)/ph, np.log(gw/pw), np.log(gh/ph)]
                        if iou_val<self.reg_threshold:
                            label[0] = 0
                        else:
                            label[0] = 1 # 这里是1，是为了判断是前景还是背景
                        labels_bbox.append(label)

                    else:
                        # 到这里才是为fine_turn准备数据的地方
                        # 并用了one_hot
                        label = np.zeros(self.F_class_num)
                        if iou_val < self.fineturn_threshold:
                            label[0] = 1
                        else:
                            label[index] = 1
                        labels.append(label)

                '''
                提供数据的持久化操作
                这个.npy格式的有什么好啊…
                可能是想着，原来就是np的array了，那么保存的话，也就直接用numpy来保存就好了
                '''
                if self.is_save:
                    if self.is_svm:
                        # 这个保存是在循环中进行的，每张完整图片保存一次
                        # 这样，就保证label一样的数据，被保存到一个文件夹中
                        if not os.path.exists(os.path.join(self.SVM_and_Reg_save_path,str(context[1]))):
                            os.makedirs(os.path.join(self.SVM_and_Reg_save_path,str(context[1])))
                        np.save((os.path.join(self.SVM_and_Reg_save_path,
                                              str(context[1]),
                                              context[0].split('/')[-1].split('.')[0].strip())
                                                    + '_data.npy'),
                                [images, labels, labels_bbox]) # 三种数据：图像特征值，图像的非one_hot的label，图像的前背景+坐标
                    else:
                        np.save((os.path.join(self.fineturn_save_path,
                                              context[0].split('/')[-1].split('.')[0].strip()) +
                                              '_data.npy'),
                                [images, labels])

    def load_from_npy(self):
        """
            从持久化的数据中来获取数据        
        :return: 
        """
        if self.is_svm:
            data_set  =self.SVM_and_Reg_save_path # 数据保存地址
            data_dirs = os.listdir(data_set) # 下面的所有文件
            for data_dir in data_dirs: # ./FlowerData/SVM_and_Reg下所有文件夹
                SVM_data = []
                data_list = os.listdir(os.path.join(data_set,data_dir))
                # 注意 ./FlowerData/Fineturn 和 ./FlowerData/SVM_and_Reg的目录结构不同
                # Fineturn不需要按照label分文件夹保存，而SVM_需要。
                for ind,d in enumerate(data_list): # ./FlowerData/SVM_and_Reg/2下所有文件
                    i,l,k = np.load(os.path.join(data_set,data_dir,d))
                    for index in range(len(i)): # ./FlowerData/SVM_and_Reg/2/image_0561_data.npy 内所有数据
                        SVM_data.append([i[index],l[index]])
                        self.Reg_data.append([i[index],k[index]])
                self.SVM_data_dic[data_dir] = SVM_data # 这里的data_dir就是label值
        else:
            data_set = self.fineturn_save_path
            data_list = os.listdir(data_set)
            for ind,d in enumerate(data_list):
                i,l = np.load(os.path.join(data_set,d))
                for index in range(len(i)):
                    self.fineturn_data.append([i[index],l[index]])


    def get_fineturn_batch(self):
        images = np.zeros((self.fineturn_batch_size,self.image_size,self.image_size,3))
        labels = np.zeros((self.fineturn_batch_size,self.F_class_num))
        count = 0
        while count<self.fineturn_batch_size:
            images[count] = self.fineturn_data[self.cursor][0]
            labels[count] = self.fineturn_data[self.cursor][1]
            count +=1
            self.cursor +=1
            if self.cursor >= len(self.fineturn_data):
                self.cursor = 0
                self.epoch += 1
                np.random.shuffle(self.fineturn_data)
                print('epoch:',self.epoch)
        return images,labels

    def get_SVM_data(self,data_dir):
        images = []
        labels = []
        for index in range(len(self.SVM_data_dic[data_dir])):
            images.append(self.SVM_data_dic[data_dir][index][0])
            labels.append(self.SVM_data_dic[data_dir][index][1])
        return images,labels

    def get_Reg_batch(self):
        images = np.zeros((self.Reg_batch_size, 4096))
        labels = np.zeros((self.Reg_batch_size, self.R_class_num))
        count = 0
        while (count < self.Reg_batch_size):
            images[count] = self.Reg_data[self.cursor][0]
            labels[count] = self.Reg_data[self.cursor][1]
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.Reg_data):
                self.cursor = 0
                self.epoch += 1
                np.random.shuffle(self.Reg_data)
        return images,labels