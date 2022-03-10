# -*- coding: utf-8 -*-
# 作者：和光同尘

# 写一个工具类来使所有图片的大小一致，神经网络的输入需要图片的大小一致
from PIL import Image


# 等比缩放的代码
def keep_image_size_open(path, size=(200, 200)):
    img = Image.open(path)  # 读入图片
    image = img.resize(size)
    image.show()
    temp = max(img.size)  # 获取图片的最长边
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))  # (temp,temp)表示所做的mask是一个正方形（边长为temp）(0,0,0)颜色为黑色
    #mask.show()
    #mask.paste(img, (0, 0))  # (0,0)表示左上角，将mask对齐图片的左上角进行粘贴
    # img.paste(mask,(0,0))
    # img.show()
    mask.paste(img, (0, 0))
    mask.show()
    mask = mask.resize(size)
    mask.show()
    return mask


if __name__ == '__main__':
    path = '/Users/lisuya/Documents/数据集/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg'
    image = keep_image_size_open(path)
    image.show()