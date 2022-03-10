# -*- coding: utf-8 -*-
# 作者：和光同尘

#只是测试一下代码
import torch

y = torch.tensor([0,1,2])
y_hat = torch.tensor([[0.1,0.3,0.6],
                      [0.3,0.2,0.5],
                      [0.2,0.8,0.9]])
print(y_hat[[0,2,1],y])