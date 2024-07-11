from common import *


def BasicConv(input, dir_name, shape, stride=1, weight_file="/w.bin", bias_file="/b.bin"):
    #print("shape[3]//2 = ",shape[3]//2)
    ofm = conv(input, dir_name, shape, stride=stride, padding=shape[3]//2,
               weight_file=weight_file, bias_file=bias_file) # shape[3] -> kernel size
    ofm = act(ofm)
    return ofm


def Resblock(input, dir_name, in_channel):
    conv1_shape = (in_channel, in_channel, 3, 3)
    conv2_shape = (in_channel//2, in_channel//2, 3, 3)
    conv3_shape = (in_channel//2, in_channel//2, 3, 3)
    conv4_shape = (in_channel, in_channel, 1, 1)
    ofm = BasicConv(input, dir_name, conv1_shape, stride=1, weight_file="/w1.bin", bias_file="/b1.bin") # conv1
    route = ofm
    ofm = torch.split(ofm, in_channel//2, dim=1)[1]
    ofm = BasicConv(ofm, dir_name, conv2_shape, stride=1, weight_file="/w2.bin", bias_file="/b2.bin") # conv2
    route1 = ofm
    ofm = BasicConv(ofm, dir_name, conv3_shape, stride=1, weight_file="/w3.bin", bias_file="/b3.bin") # conv3
    ofm = torch.cat([ofm, route1], dim=1)
    ofm = BasicConv(ofm, dir_name, conv4_shape, stride=1, weight_file="/w4.bin", bias_file="/b4.bin") # conv4
    feat = ofm
    ofm = torch.cat([route, ofm], dim=1)
    ofm = maxpool(ofm, kernel_size=2, stride=2, padding=0)
    return ofm, feat


def Conv_Upsample(input, dir_name, in_channel):
    out_channel = in_channel//2
    shape = (out_channel, in_channel, 1, 1)
    ofm = BasicConv(input, dir_name, shape, stride=1)
    ofm = upsample(ofm)
    return ofm

def yolo_head(input, dir_name, in_ch, out_ch, num_classes):
    """
    in_ch: head的 input 的通道数
    out_ch: head 卷积块的输出通道数
    """
    shape1 = (out_ch, in_ch, 3, 3)
    shape2 = (3*(5+num_classes), out_ch, 1, 1)
    ofm = BasicConv(input, dir_name, shape1, stride=1, weight_file="/w1.bin", bias_file="/b1.bin")
    ofm = conv(ofm, dir_name, shape2, stride=1, weight_file="/w2.bin", bias_file="/b2.bin")
    return ofm


if __name__ == "__main__":
    BasicConv(0, "BasicConv1", (3, 3, 3, 3))