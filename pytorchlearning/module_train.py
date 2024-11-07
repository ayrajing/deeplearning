import shutil
import os
# 清理旧的日志文件
log_dir = "./logs_train"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

import torch
from torch.utils.tensorboard import SummaryWriter
from model import *
import torchvision

import torch.nn as nn
from torch.utils.data import DataLoader# 准备数据集
train_data = torchvision.datasets.CIFAR10("../dataset",train=True, transform=torchvision.transforms.ToTensor(),
                                         download=True)
test_data = torchvision.datasets.CIFAR10("../dataset",train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
# len()获取数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=64, drop_last=True)

#创建网络模型
tudui = Tudui()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2  # 学习率  
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0

# 记录测试的次数
total_test_step = 0

# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("/Users/jingjing/data/MNIST/raw/pytorchlearning/logs_train") 
# 开始训练循环
for i in range(epoch):
    print("-------------第 {} 轮训练开始------------".format(i+1))
    #训练步骤开始
    tudui.train() # 设置模型为训练模式
    for data in train_dataloader:
        imgs, targets = data
        output = tudui(imgs)  # 前向传播，输入数据，输出预测结果
        loss = loss_fn(output, targets)  # 计算损失

        #优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新参数
        total_train_step = total_train_step + 1
        if total_train_step % 100 ==0:
            print("训练次数：{}， Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss",loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()  # 设置模型为评估模式
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 禁用梯度计算
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)  # 前向传播，输入数据，输出预测结果
            loss = loss_fn(outputs, targets)  # 计算损失
            total_test_loss = total_test_loss + loss.item()  # 累加损失
            accuracy = (outputs.argmax(1) == targets).sum()  # 计算准确率
            total_accuracy = total_accuracy + accuracy  # 累加准确率
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, i)  # 使用epoch序号i作为x轴
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, i)
    total_test_step = total_test_step + 1

    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")
writer.flush()  # 确保数据被写入
writer.close()

