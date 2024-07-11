import numpy as np
import torch
import os
import argparse

# 用于查看模型内部权重的前缀定位字符串
def show_prefix(state):
    #print(state)
    for name in state:# 打印每一层的Key
        print(name)

# 将卷积块(Conv2d+bn+leakyrelu)中的相邻的Conv2d和BN权重融合
def conv_bn_merge(pre,state):
    eps=1e-5
    conv_weight = state[pre+"conv.weight"].cpu().numpy()
    bn_weight   = state[pre+"bn.weight"].cpu().numpy()
    bn_bias     = state[pre+"bn.bias"].cpu().numpy()
    bn_mean     = state[pre+"bn.running_mean"].cpu().numpy()
    bn_var      = state[pre+"bn.running_var"].cpu().numpy()
    # merge
    fused_weight= conv_weight*bn_weight.reshape(-1,1,1,1)/np.sqrt(bn_var.reshape(-1,1,1,1)+eps)
    fused_bias  = bn_bias-bn_mean*bn_weight/np.sqrt(bn_var+eps)
    return fused_weight,fused_bias # 返回融合后的 weight 和 bias

# 将残差块中相邻的Conv2d和BN权重融合
def resblock_merge(pre,i,dir,state):
    w1,b1=conv_bn_merge(pre + "conv1.",state)
    w2,b2=conv_bn_merge(pre + "conv2.",state)
    w3,b3=conv_bn_merge(pre + "conv3.",state)
    w4,b4=conv_bn_merge(pre + "conv4.",state)
    w1.tofile(dir+"\\ResBlock{}\\w1.bin".format(i))
    b1.tofile(dir+"\\ResBlock{}\\b1.bin".format(i))
    w2.tofile(dir+"\\ResBlock{}\\w2.bin".format(i))
    b2.tofile(dir+"\\ResBlock{}\\b2.bin".format(i))
    w3.tofile(dir+"\\ResBlock{}\\w3.bin".format(i))
    b3.tofile(dir+"\\ResBlock{}\\b3.bin".format(i))
    w4.tofile(dir+"\\ResBlock{}\\w4.bin".format(i))
    b4.tofile(dir+"\\ResBlock{}\\b4.bin".format(i))

# 导出整个网络的参数
def save_folded_weights(dir,state):
    w1, b1 = conv_bn_merge("backbone.conv1.", state)
    w2, b2 = conv_bn_merge("backbone.conv2.", state)
    w3, b3 = conv_bn_merge("backbone.conv3.", state)

    w1.tofile(dir+"\\BasicConv1\\w.bin")
    b1.tofile(dir+"\\BasicConv1\\b.bin")
    w2.tofile(dir+"\\BasicConv2\\w.bin")
    b2.tofile(dir+"\\BasicConv2\\b.bin")
    w3.tofile(dir+"\\BasicConv3\\w.bin")
    b3.tofile(dir+"\\BasicConv3\\b.bin")

    resblock_merge("backbone.resblock_body1.", 1,dir, state)
    resblock_merge("backbone.resblock_body2.", 2,dir, state)
    resblock_merge("backbone.resblock_body3.", 3,dir, state)

    # conv_for_P5 bn融合
    w, b = conv_bn_merge("conv_for_P5.", state)
    w.tofile(dir+"\\conv_forP5\\w.bin")
    b.tofile(dir+"\\conv_forP5\\b.bin")
    # yolo_headP4
    w1, b1 = conv_bn_merge("yolo_headP4.0.", state)
    w2 = state['yolo_headP4.1.weight'].cpu().numpy() # head中 单Conv2D 的w2和b2
    b2 = state['yolo_headP4.1.bias'].cpu().numpy()
    w1.tofile(dir+"\\yolo_headP4\\w1.bin")
    b1.tofile(dir+"\\yolo_headP4\\b1.bin")
    w2.tofile(dir+"\\yolo_headP4\\w2.bin")
    b2.tofile(dir+"\\yolo_headP4\\b2.bin")
    # yolo_headP5
    w1, b1 = conv_bn_merge("yolo_headP5.0.", state)
    w2 = state['yolo_headP5.1.weight'].cpu().numpy()
    b2 = state['yolo_headP5.1.bias'].cpu().numpy()
    w1.tofile(dir+"\\yolo_headP5\\w1.bin")
    b1.tofile(dir+"\\yolo_headP5\\b1.bin")
    w2.tofile(dir+"\\yolo_headP5\\w2.bin")
    b2.tofile(dir+"\\yolo_headP5\\b2.bin")
    # upsample
    w, b = conv_bn_merge("upsample.upsample.0.", state)
    w.tofile(dir+"\\upsample\\w.bin")
    b.tofile(dir+"\\upsample\\b.bin")


# Create folders to save the weights
def mk_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    os.mkdir(dir+"\\BasicConv1")
    os.mkdir(dir+"\\BasicConv2")
    os.mkdir(dir+"\\BasicConv3")
    os.mkdir(dir+"\\conv_forP5")
    os.mkdir(dir +"\\ResBlock1")
    os.mkdir(dir +"\\ResBlock2")
    os.mkdir(dir +"\\ResBlock3")
    os.mkdir(dir+"\\upsample")
    os.mkdir(dir+"\\yolo_headP4")
    os.mkdir(dir+"\\yolo_headP5")

if __name__=='__main__':
    """ 
    # 如果想用输入的方式指定路径，就用此注释代码
    parser = argparse.ArgumentParser() # 创建 ArgumentParser 对象
    parser.add_argument("-i",help="original weight path") # 添加参数选项 "-i" 和 "-o"(in,out)
    parser.add_argument("-o",help="folded weight path")
    args=parser.parse_args() # 解析命令行参数
    weight_path=args.i
    folded_weight_path=args.o
    """
    weight_path = "../model_data/yolov4_tiny_weights_voc.pth"  # 需要配置的变量: 训练好的模型路径
    folded_weight_path = "./folded_weight" # 导出权重的存储路径
    print("Input model path is ",weight_path)
    print("Output weight path is ", folded_weight_path)
    state=torch.load(weight_path) # load model
    show_prefix(state)
    mk_dir(folded_weight_path)    # create folders
    save_folded_weights(folded_weight_path,state) # save the weights
