# Copyright (c) AlphaBetter. All rights reserved.
import os

import torch
from PIL import Image
from torch import nn
from torchvision import transforms as transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）


# 加载模型
Infer_model = Net()  # 获得网络结构
Infer_model.load_state_dict(torch.load("model_Mnist.pth"))  # 加载最后训练的参数

# 设置为评估模式
Infer_model.eval()

# 图片预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # 预处理


def predict_image(image_path: str):
    """推理单张图片并返回预测结果"""
    image = Image.open(image_path)
    image = image.resize((28, 28))  # 输入尺寸与网络的输入保持一致
    image = image.convert("L")  # 转为灰度图，保持通道数与网络输入一致
    image = transform(image)

    # 增加一个维度来模拟batch size (1, 1, 28, 28)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = Infer_model(image)  # 得到推理结果

    # 返回函数的最大值的下标
    pred = torch.argmax(output)
    return pred.item()


def process_images_in_folder(folder_path: str):
    """批量处理文件夹中的所有图片"""
    # 获取文件夹中的所有图片文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(folder_path, filename)
            pred = predict_image(image_path)
            print(f"Prediction for {filename}: {pred}")


# 设置图片文件夹路径
folder_path = "./results/inference/0"  # 修改为你自己的图片文件夹路径
process_images_in_folder(folder_path)

folder_path = "./results/inference/1"  # 修改为你自己的图片文件夹路径
process_images_in_folder(folder_path)

folder_path = "./results/inference/8"  # 修改为你自己的图片文件夹路径
process_images_in_folder(folder_path)
