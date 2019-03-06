import cv2
import numpy as np
import random
import os
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import config as cfg
import time

'''
拿到增强所需要的参数
'''
def get_aug_dict():
    '''
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True
    '''
    r_rotation_range = random.random() / 5
    r_width_shift_range = random.random() / 20
    r_height_shift_range = random.random() / 20
    r_shear_range = random.random() / 20
    r_zoom_range = random.random() / 20
    r_horizontal_flip = random.choice([True, False])

    data_gen_args = dict(rotation_range=r_rotation_range,
                         width_shift_range=r_width_shift_range,
                         height_shift_range=r_height_shift_range,
                         shear_range=r_shear_range,
                         zoom_range=r_zoom_range,
                         horizontal_flip=r_horizontal_flip)
    return data_gen_args

'''
将label【height,weight,3】
变成【height,weight,2】
'''
def process_label(label_img):
    h,w,channel = label_img.shape
    last_label = np.zeros(shape=[h,w],dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if max(label_img[i,j,:])>0:
                last_label[i,j]=1
    last_label = np.uint8(last_label)
    last_label = np.reshape(last_label,(h,w,1))
    # 只需要返回一个维度就好
    # last_label_reverse = 1-last_label
    # output = np.concatenate([last_label,last_label_reverse],axis = 2)
    return last_label

'''
读取原始数据
将数据分为训练和测试两部分
'''
def devide_all_data():
    all_data = []
    for fi in os.listdir('./my_train_data/'):
        temp_path = './my_train_data/' + fi + '/'

        org_img = cv2.imread(temp_path + 'img.png')
        lab_img = cv2.imread(temp_path + 'label.png')
        lab_img = process_label(lab_img)

        all_data.append((org_img, lab_img))

    valid_choice = random.sample(list(range(95)), 15)
    valid_data = [data for i, data in enumerate(all_data) if i in valid_choice]
    train_data = [data for i, data in enumerate(all_data) if i not in valid_choice]

    return train_data, valid_data


'''
读取keras的数据
'''
def devide_all_data2():
    train_img = './ipython_file/train/image/'
    train_lab = './ipython_file/train/label/'
    all_res = []
    for img_path, lab_path in zip(os.listdir(train_img), os.listdir(train_lab)):
        org_img = cv2.imread(train_img + img_path)

        org_lab = cv2.imread(train_lab + lab_path)
        org_lab = process_label(org_lab)

        all_res.append((org_img, org_lab))

    valid_img = './ipython_file/valid/image/'
    valid_lab = './ipython_file/valid/label/'
    all_res_val = []
    for img_path, lab_path in zip(os.listdir(valid_img), os.listdir(valid_lab)):
        org_img = cv2.imread(valid_img + img_path)

        org_lab = cv2.imread(valid_lab + lab_path)
        org_lab = process_label(org_lab)

        all_res_val.append((org_img, org_lab))

    return all_res, all_res_val

'''
将数据保存成tfrecords的格式。
'''
def to_tfrecords(raw_data, record_name):
    writer = tf.python_io.TFRecordWriter(record_name)
    for temp_org_img, temp_lab_img in raw_data:

        temp_org_img = cv2.resize(temp_org_img,(cfg.IMG_SIZE,cfg.IMG_SIZE))
        temp_org_img = Image.fromarray(temp_org_img)
        # temp_org_img = temp_org_img.resize((cfg.IMG_SIZE,cfg.IMG_SIZE))
        temp_org_img_raw = temp_org_img.tobytes()

        temp_lab_img = cv2.resize(temp_lab_img,(cfg.IMG_SIZE,cfg.IMG_SIZE))
        temp_lab_img = Image.fromarray(temp_lab_img)
        # temp_lab_img = temp_lab_img.resize((cfg.IMG_SIZE,cfg.IMG_SIZE))
        temp_lab_img_raw = temp_lab_img.tobytes()

        feature_dict = {
            'label': tf.train.Feature(bytes_list=
                                      tf.train.BytesList(value=[temp_lab_img_raw])),
            'img_raw': tf.train.Feature(bytes_list=
                                        tf.train.BytesList(value=[temp_org_img_raw]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

        writer.write(example.SerializeToString())

    writer.close()

'''
直接读取tfrecords文件
'''
def read_and_parse(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)

    features_dict = {
        'label': tf.FixedLenFeature([], tf.string),
        'img_raw': tf.FixedLenFeature([], tf.string)
    }

    features_res = tf.parse_single_example(serialized_example,features=features_dict)
    p_tf_org_img = tf.decode_raw(features_res['img_raw'], tf.uint8)
    p_tf_org_img = tf.reshape(p_tf_org_img, [cfg.IMG_SIZE, cfg.IMG_SIZE, 3])

    p_tf_lab_img = tf.decode_raw(features_res['label'], tf.uint8)
    p_tf_lab_img = tf.reshape(p_tf_lab_img, [cfg.IMG_SIZE, cfg.IMG_SIZE, 2])

    return p_tf_org_img, p_tf_lab_img


'''
希望通过tf.data的API来读取文件的时候，
传入参数是record
'''
def parser(record):
    p_feature_dict = {
        'label': tf.FixedLenFeature([], tf.string),
        'img_raw': tf.FixedLenFeature([], tf.string)
    }
    p_features_res = tf.parse_single_example(record,
                                             features=p_feature_dict)

    p_tf_org_img = tf.decode_raw(p_features_res['img_raw'], tf.uint8)
    p_tf_org_img = tf.reshape(p_tf_org_img, [cfg.IMG_SIZE, cfg.IMG_SIZE, 3])

    p_tf_lab_img = tf.decode_raw(p_features_res['label'], tf.uint8)
    p_tf_lab_img = tf.reshape(p_tf_lab_img, [cfg.IMG_SIZE, cfg.IMG_SIZE, 1])

    return p_tf_org_img, p_tf_lab_img

'''
数据增强综合方法
'''
def img_aug(imgs,labels):
    aug_dict = get_aug_dict()
    # print(aug_dict)
    my_aug = My_Aug(**aug_dict)

    aug_imgs = my_aug.process_batch(imgs)
    aug_labs = my_aug.process_batch(labels)
    return aug_imgs,aug_labs

'''
展示图片
'''
def show_img(img):
    plt.imshow(img)
    plt.show()

def predict_to_img(img):
    '''
    :param img: 256,256,2 after softmax 
    :return: 
    '''
    # last_layer = np.zeros(shape=(256,256,1))
    # print(np.sum(np.float32(img>0.5)))
    layer_1 = np.float32(img>0.5)*255

    # layer_1 = np.reshape(layer_1,newshape=(256,256,1))
    # layer_2 = np.uint8(img[:,:,1]*255)
    # layer_2 = np.reshape(layer_2,newshape=(256,256,1))
    # print('last_layer:',last_layer.shape)
    # print('layer_1:',layer_1.shape)
    # print('layer_2:',layer_2.shape)
    return layer_1

'''
数据增强类
'''
class My_Aug():
    def __init__(self,
                 rotation_range,
                 height_shift_range,
                 width_shift_range,
                 shear_range,
                 zoom_range,
                 horizontal_flip):
        self.roration_range = rotation_range
        self.height_shift_range = height_shift_range
        self.width_shift_range = width_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip

    def _roration_img(self, img):
        '''
            旋转图像
        '''
        # 默认图像是灰度图像
        h, w,channel = img.shape
        M = cv2.getRotationMatrix2D((h / 2, w / 2), self.roration_range, 1)
        img = cv2.warpAffine(img, M, (h, w))
        img = np.reshape(img,(h,w,channel))
        return img

    def _shift_img(self, img):
        '''
        平移变换
        '''
        h, w,channel = img.shape
        M = np.float32([[1, 0, self.width_shift_range], [0, 1, self.height_shift_range]])
        img = cv2.warpAffine(img, M, (h, w))
        img = np.reshape(img,(h,w,channel))
        return img

    def _shear_img(self, img):
        '''
        x方向的剪切变换
        '''
        h, w,channel = img.shape
        M = np.float32([[1, np.tan(self.shear_range), 0], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (h, w))
        img = np.reshape(img,(h,w,channel))
        return img

    def _zoom_img(self, img):
        '''
        缩放
        '''
        h, w,channel = img.shape
        zoom_value = 1 + self.zoom_range
        M = np.float32([[zoom_value, 0, 0], [0, zoom_value, 0]])
        img = cv2.warpAffine(img, M, (h, w))
        img = np.reshape(img,(h,w,channel))
        return img

    def _flip_img(self, img):
        h,w,channel = img.shape
        img =cv2.flip(img, 1)
        img = np.reshape(img,(h,w,channel))
        return img

    def process_img(self, img):
        '''
            按顺序处理图片：
                1、旋转
                2、平移
                3、剪切
                4、缩放
                5、水平翻转
        '''
        img = self._roration_img(img)
        img = self._shift_img(img)
        img = self._shear_img(img)
        img = self._zoom_img(img)
        # if self.horizontal_flip:
        #     img = self._flip_img(img)
        return img

    def process_batch(self, imgs):

        # processed_imgs = np.asarray([self.process_img(img) for img in imgs])
        res=[]
        for img in imgs:
            temp_res = self.process_img(img)
            res.append(temp_res)

        processed_imgs = np.asarray(res)

        return processed_imgs

    def img_show(self, img):
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    '''
    调用两个方法，生成对应的tf_records文件
    '''
    train_raw_data, valid_raw_data = devide_all_data()
    to_tfrecords(train_raw_data, 'train_palm.tfrecords')
    to_tfrecords(valid_raw_data, 'valid_palm.tfrecords')

    # tf_records文件的placeholder
    file_names = tf.placeholder(dtype=tf.string, shape=[None])

    dataset = tf.data.TFRecordDataset(file_names)

    # 这里是调用映射函数的地方…但不知道能不能用自己的方法啊…
    # 是不是需要将自己的方法封装成tensorflow的方法？
    dataset = dataset.map(parser)
    dataset = dataset.repeat(cfg.REPEAT_TIME)
    dataset = dataset.batch(20)

    # 拿到迭代器
    iterator_init = dataset.make_initializable_iterator()
    # 拿到数据
    images, labels = iterator_init.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 显式初始化
        train_filenames=['./train_palm.tfrecords']
        sess.run(iterator_init.initializer,feed_dict={file_names:train_filenames})
        start_time = time.time()
        for i in range(10):
            train_img,train_label = sess.run([images,labels])
            train_img, train_label = img_aug(train_img, train_label)
            print(train_label.shape)
            break



        end_time = time.time()
        t_i_list = []
        train_img,train_label = img_aug(train_img,train_label)
        end_time1 = time.time()
        print('sess:',end_time-start_time)
        print('aug:',end_time1-end_time)

