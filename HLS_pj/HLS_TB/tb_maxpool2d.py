import torch
import torch.nn.functional as F
import numpy as np
import time
import os

# 支持任意 kernel size 的最大池化验证

# 存放目录
root_path        = './maxpool2d/'
#  check if the folder exists and create it if it doesn't
if not os.path.exists(root_path):
    os.makedirs(root_path)
    print(f"Folder '{root_path}' created successfully.")
else:
    print(f"Folder '{root_path}' already exists.")

inputs_path      = root_path + "inputs.txt"
results_path     = root_path + "results.txt"
hls_path     = root_path + "hls.txt"
diff_path    = root_path + "diff.txt"
#=====================================================================
# 层相关参数定义
IN_CHANNELS  = 4
IN_SIZE      = 416

# k2 s2 p0
K_SIZE       = 2
STRIDE_H     = K_SIZE
STRIDE_W     = STRIDE_H
PADDING      = 0

OUT_CHANNELS = IN_CHANNELS
OUT_SIZE_H   = IN_SIZE//STRIDE_H
OUT_SIZE_W   = IN_SIZE//STRIDE_W
print("kernel size is ({}, {})".format(K_SIZE, K_SIZE))

BATCH_SIZE = 1
#=======================================================
def generate_and_compute():
# (1) generate input
    inputs = np.random.random((BATCH_SIZE, IN_CHANNELS, IN_SIZE, IN_SIZE)) # 取值 [0~1)
    inputs = inputs*2 - 1 #[-1, 1]
    inputs = inputs.astype(np.float32)# float64 to float32
    #print(inputs)

    with open(inputs_path, 'w') as file:
        for n in range(BATCH_SIZE):
            for c in range(IN_CHANNELS):
                for h in range(IN_SIZE):
                    for w in range(IN_SIZE):
                        value = inputs[n][c][h][w]
                        file.write(str(value)+"\n")

    input_lines = BATCH_SIZE*IN_CHANNELS*IN_SIZE*IN_SIZE
    print("共有{}行 input".format(input_lines))

    # (2) compute results
    # input NCHW
    # output NCHW
    inputs  = torch.tensor(inputs)
    print(inputs.shape)
    outputs = F.max_pool2d(input=inputs, kernel_size=K_SIZE, stride=(STRIDE_H,STRIDE_W), padding=PADDING)
    print(outputs.dtype, outputs.shape)

    # (4) record results,保存结果（10进制）
    N_outputs = outputs.shape[0]
    C_outputs = outputs.shape[1]
    H_outputs = outputs.shape[2]
    W_outputs = outputs.shape[3]

    with open(results_path,'w') as file:
        for n in range(N_outputs):
            for c in range(C_outputs):
                for h in range(H_outputs):
                    for w in range(W_outputs):
                        value = outputs[n][c][h][w].numpy()
                        # 写入文件
                        file.write(str(value) + "\n")

    output_lines = N_outputs*C_outputs*H_outputs*W_outputs
    print("共有{}行 output".format(output_lines))
    print(time.strftime("%Y-%m-%d %H:%M:%S") + "| success！") # 打印时间戳

precent_limit = 1 # 误差允许低于 limit %
abs_limit     = 1 # 绝对值差值允许低于 abs_limit
def check_diff(): # HLS仿真结果与软件计算结果差值计算
    with open(hls_path, 'r') as file1:
        with open(results_path, 'r') as file2:
            with open(diff_path, 'w') as file3:
                for c in range(OUT_CHANNELS):
                    for h in range(OUT_SIZE_H):
                        for w in range(OUT_SIZE_W):
                            value_pytorch = float(file2.readline())
                            value_hls     = float(file1.readline())
                            diff          = value_pytorch - value_hls
                            # 差值/原值，百分比
                            precent = (diff * 100) / value_pytorch
                            if ((abs(diff) > abs_limit) or (precent > precent_limit)):
                                file3.write(str(c*OUT_SIZE_H*OUT_SIZE_W + h*OUT_SIZE_W + w + 1) + str(" : ") + str(
                                    diff) + "  " + str(precent) + "% \n")

    print(time.strftime("%Y-%m-%d %H:%M:%S") + "| success！")  # 打印时间戳

if __name__ == "__main__":
    mode = input("请输入要进行的操作:\n "
                 "0:生成激励并计算参考输出\n "
                 "1:进行结果对比\n")
    if mode=="0":
        generate_and_compute()
    elif mode=="1":
        check_diff()
    else:
        print("输出的值错误，请重新输入")