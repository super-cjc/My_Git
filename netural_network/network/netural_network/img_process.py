import numpy as np
from scipy.signal import convolve2d
import cv2
import os
"""
dt/dx = Img[x,y+1]-Img[x,y-1]/2
dt/dy = Img[x+1,y]-Img[x-1,y]/2
???为什么不选取
dt/dx = Img[x,y+1]-Img[x,y]/2
这个更接近的是[x,y+0.5]的导数
"""
class Img_process(object):
    def __init__(self,img):
        """
        输入的img是图像,卷积核使用的是三维
        """
        self.img = img
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_array = np.array(self.img)
        #获取图像二维数组的长宽,进行扩展
        self.row = len(self.img_array)    #行数
        self.col = len(self.img_array[0]) #列数
    def expend_img(self):
        """对图像边缘进行扩展"""
        img_array_expand = np.zeros((self.row+2,self.col+2,3)) #对图像边缘进行扩展
        img_array_expand[1:self.row+1,1:self.col+1] = self.img_array
        return img_array_expand

    def array2img(self,img_list,filename):
        """将二维数组转化为图像"""

        if os.path.isfile(filename):
            print("已存在该名字的图片,请重新进行命名")
        else :
            cv2.imwrite(filename,img_list)

    def mean_filter(self):
        """均值滤波"""
        mean_array = np.array([[1/9,1/9,1/9],
                               [1/9,1/9,1/9],
                               [1/9,1/9,1/9]])
        img_mean = np.zeros((self.row,self.col,3))
        for i in range(3):
            """convolve2d函数自带扩展图片功能，默认是以0填充"""
            img_mean[:,:,i] = convolve2d(self.img_array[:,:,i], mean_array, mode='same')
        return img_mean

    def verital_filter(self):
        """垂直方向特征"""
        vertical_array = np.array([[1, 1, 1],
                                   [0, 0, 0],
                                   [-1, -1, -1]])
        img_verital = np.zeros((self.row,self.col,3))
        for i in range(3):
            img_verital[:,:,i] = convolve2d(self.img_array[:,:,i], vertical_array, mode='same')
        return img_verital

    def horizontial_filter(self):
        """水平方向特征"""
        horizontal_array = np.array([[1,0,-1],
                                   [1,0,-1],
                                   [1,0,-1]])
        img_horizontal = np.zeros((self.row,self.col,3))
        for i in range(3):
            img_horizontal = convolve2d(self.img_array[:,:,i], horizontal_array, mode='same')
        return img_horizontal



if __name__ == '__main__':
    img = cv2.imread('../pic/human/Lena.png')
    PIC = Img_process(img)
    PIC.array2img(PIC.horizontial_filter(),'../pic/handle/horizontial_filter1.png')
    pass
