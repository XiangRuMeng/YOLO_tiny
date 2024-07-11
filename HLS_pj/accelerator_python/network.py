import torch
import os

from module import *

# kernel shape
config = [# Backbone
          (32, 3, 3, 3),  # BasicConv1
          (64, 32, 3, 3), # BasicConv2

          64, # ResBlock1~3
          128,
          256,

          (512, 512, 3, 3),  # BasicConv3
          (256, 512, 1, 1),  # conv_forP5
          ]

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def Backbone(input):
    print("==== Backbone ====")

    dir_name = "BasicConv1"
    ofm = BasicConv(input, dir_name, shape=config[0], stride=2) # K3 S2 P1
    print("layer 0, BasicConv1, ", ofm.shape)

    dir_name = "BasicConv2"
    ofm = BasicConv(ofm, dir_name, shape=config[1], stride=2)   # K3 S2 P1
    print("layer 1, BasicConv2,", ofm.shape)

    dir_name = "ResBlock1"
    ofm, _ = Resblock(ofm, dir_name, in_channel=config[2])
    print("layer 2, ResBlock1, ", ofm.shape)
    dir_name = "ResBlock2"
    ofm, _ = Resblock(ofm, dir_name, in_channel=config[3])
    print("layer 3, ResBlock2, ", ofm.shape)
    dir_name = "ResBlock3"
    ofm, feat1 = Resblock(ofm, dir_name, in_channel=config[4])
    print("layer 4, ResBlock3, ", ofm.shape)

    dir_name = "BasicConv3"
    ofm = BasicConv(ofm, dir_name, shape=config[5], stride=1)  # K3 S1 P1
    print("layer 5, BasicConv3,", ofm.shape)
    feat2 = ofm
    return feat1, feat2

def YOLOv4_tiny(input, num_classes=20, random=False, Save=False):
    """
        num_classes: 分类数，VOC数据集为 20， COCO数据集为 80
        random: 用随机输入测试, 默认 False
        Save: 是否保存 out0 和 out1
    """
    if random:
        random_input = torch.rand((1, 3, 416, 416)) * 2 - 1  # [-1, 1]
        feat1, feat2 = Backbone(random_input)
        print("feat1 shape is, ", feat1.shape)
        print("feat2 shape is, ", feat2.shape)
    else:
        feat1, feat2 = Backbone(input)
        print("feat1 shape is, ", feat1.shape)
        print("feat2 shape is, ", feat2.shape)

    print("==== Neck and Head ====")

    dir_name = "conv_forP5"
    P5 = BasicConv(feat2, dir_name, shape=config[6], stride=1)
    print("layer 6, conv_forP5", P5.shape)

    dir_name = "yolo_headP5"
    out0 = yolo_head(P5, dir_name, in_ch=256, out_ch=512, num_classes=num_classes)
    print("yolo_head P5, ", out0.shape)

    dir_name = "upsample"
    P5_Upsample = Conv_Upsample(P5, dir_name, in_channel=256)
    print("Conv_Upsample, ", P5_Upsample.shape)
    P4 = torch.cat([P5_Upsample, feat1], dim=1)

    dir_name = "yolo_headP4"
    out1 = yolo_head(P4, dir_name, in_ch=384, out_ch=256, num_classes=num_classes)
    print("yolo_head P4, ", out1.shape)

    # save out0 and out1
    if Save:
        # tensor -> numpy-> bin
        out0.numpy().tofile(result_dir + "/out0.bin")
        out1.numpy().tofile(result_dir + "/out1.bin")
        print("out0 and out1 are saved.")

    return out0, out1


def read_img_bin(path, shape=(1, 3, 416, 416)):
    """
    path: 测试图片 img.bin 的路径
    shape: 图片shape (1, 3, 416, 416)
    """
    # load data, 转为 float32
    data = torch.from_numpy(np.fromfile(path, dtype=np.float32)).float()
    data = data.reshape(shape)
    print("Load image data from ", path)
    return data

if __name__ == "__main__":
    # 1 测试图片数据读取
    img_data = read_img_bin(path="../yolo_test/img.bin")
    # 2 网络推理
    out0, out1 = YOLOv4_tiny(input=img_data, num_classes=20, random=False, Save=True)




