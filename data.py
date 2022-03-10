# -*- coding: utf-8 -*-
# 作者：和光同尘

import os
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
transform=transforms.Compose([
    transforms.ToTensor()   #这里Transforms一下，相当于归一化了
])

class MyDataset(Dataset):
    def __init__(self,path):
        self.path=path  #path="/Users/lisuya/Documents/数据集/VOCdevkit/VOC2012"
        ##获得/Users/lisuya/Documents/数据集/VOCdevkit/VOC2012/SegmentationClass文件夹下的图片名称的列表
        self.name = os.listdir(os.path.join(path,'SegmentationClass'))

    def __len__(self):
        return len(self.name)   #返回name列表的长度

    def __getitem__(self, index):
        segment_name = self.name[index] #取出每一个SegmentationClass的名字
        segment_path = os.path.join(self.path,'SegmentationClass',segment_name) #取出SegmentationClass下每一个图片的完整路径
        image_path = os.path.join(self.path,'JPEGImages',segment_name.replace('png','jpg')) #取出JPEGImages下每一张图片的完整路径
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transform(image),transform(segment_image)

if __name__ == '__main__':
    data = MyDataset('/Users/lisuya/Documents/数据集/VOCdevkit/VOC2012')
    print(data[0][0].shape)
    print(data[0][1].shape)