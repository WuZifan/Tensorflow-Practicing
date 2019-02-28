# 这个是用来练习TF_Record和TF_Data api的。

* 1） 代码参考：https://www.jianshu.com/p/eec32f6c5503

* 2） TF读取数据原理参考：https://zhuanlan.zhihu.com/p/27238630

* 3） TF_DATA API参考：https://zhuanlan.zhihu.com/p/30751039

* 4） TF_RECORD参考：https://blog.csdn.net/miaomiaoyuan/article/details/56865361
	这里注意，在读取TF_RECORD，并进行图片解码的时候，会遇到：
		img = tf.decode_raw(features['img_raw'], tf.uint8)
	请让最后的tf.uint8这个类型的最后的位数，和图片原来类型相同
	（比如原来是np.float32,那请用tf.uint32解码）

* 5）TF_RECORD和python读取对比：https://zhuanlan.zhihu.com/p/27481108
