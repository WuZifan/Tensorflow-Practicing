import numpy as np

if __name__ == '__main__':
    aa= np.array([ [0,2],
                    [2,3],
                    [3,4],
                    [4,5],
                    [5,6]])
    bb=np.array([2,2])

    print(aa[:,0])

    res = np.minimum(aa[:,0],bb[0])
    print(res)

    print(np.count_nonzero(res == 0 ))

    print(aa[:,0]*aa[:,1])

    anchors = np.array([
             [ 10,13],
             [ 16,  30.],
             [ 33,  23.],
             [ 30,  61.],
             [ 62,  45.],
             [ 59, 119.],
             [116,  90.],
             [156, 198.],
             [373, 326.]])
    bb=np.array([[[2,3]]])
    res = np.maximum(bb,anchors)

    print(bb.shape,anchors.shape,res.shape)
    print(res)

    aaa=[1,2,3,4]
    print(aaa.index(2))