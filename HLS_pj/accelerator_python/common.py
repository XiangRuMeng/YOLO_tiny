import numpy as np
import torch
import torch.nn.functional as F

from config import *

###################
# conv2d
###################
def conv(input, dir_name, shape, stride=1, padding=0,
         weight_file="w.bin", bias_file="b.bin"):
    """
    dir_name: 权重所在目录，eg. "BasicConv1/"
    shape: 卷积核的 NCHW, eg. (8, 1, 1,4)
    stride: 步长
    padding: 填充
    weight_file,bias_file: 权重和偏置文件名
    """
    # 1 load weight and bias, 转为 float32
    weight = torch.from_numpy(np.fromfile(root + dir_name + weight_file, dtype=np.float32)).float()
    weight = weight.reshape(shape)
    bias   = torch.from_numpy(np.fromfile(root + dir_name + bias_file, dtype=np.float32)).float()
    #print(weight.dtype, bias.dtype)
    # 2 compute
    output = F.conv2d(input, weight, bias, stride, padding)
    # 3 store output
    return output

###################
# Pooling
###################
def maxpool(input, kernel_size, stride=None, padding=0):
    output = F.max_pool2d(input, kernel_size, stride, padding)
    return output

###################
# others
###################
def act(input):
    output = F.leaky_relu(input, negative_slope=0.1, inplace=False)
    return output

def upsample(input):
    output = F.interpolate(input, scale_factor=2.0, mode="nearest")
    #output = F.upsample(input, scale_factor=2.0, mode="nearest")
    return output

def concat(in1, in2, dim=1):
    output = torch.cat((in1, in2), dim=dim)
    return output
