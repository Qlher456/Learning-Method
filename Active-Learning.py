import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random

# 超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 100
output_dir = "./training_plots"  # 折线图保存目录
initial_labeled_size = 1000  # 初始标注样本数量
query_size = 500  # 每次查询的样本数量

# 创建保存目录
os.makedirs(output_dir, exist_ok=True)

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# 初始标注数据
initial_labeled_indices = random.sample(range(len(train_dataset)), initial_labeled_size)
labeled_data = torch.utils.data.Subset(train_dataset, initial_labeled_indices)
unlabeled_data = torch.utils.data.Subset(train_dataset,
                                         list(set(range(len(train_dataset))) - set(initial_labeled_indices)))

train_loader = torch.utils.data.DataLoader(labeled_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 训练和测试函数
def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100. * correct / total
    return running_loss / len(loader), accuracy


def test(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100. * correct / total
    return running_loss / len(loader), accuracy


# 主动学习选择不确定样本的函数
def query_uncertain_samples(model, unlabeled_data, num_samples):
    model.eval()
    uncertain_samples = []

    # 使用模型计算未标注样本的不确定性（最大熵法）
    for idx in range(len(unlabeled_data)):
        image, _ = unlabeled_data[idx]
        image = image.unsqueeze(0).to(device)  # 扩展为批次大小为1
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        uncertainty = -torch.sum(probs * torch.log(probs))  # 计算熵
        uncertain_samples.append((idx, uncertainty.item()))

    # 按照不确定性排序，选择前num_samples个
    uncertain_samples.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [x[0] for x in uncertain_samples[:num_samples]]

    return selected_indices


# 记录训练过程
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

# 主动学习循环
for epoch in range(num_epochs):
    # 训练阶段
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    test_loss, test_acc = test(model, test_loader, criterion)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f"Epoch [{epoch + 1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # 主动学习：选择最不确定的样本进行标注
    uncertain_indices = query_uncertain_samples(model, unlabeled_data, query_size)

    # 将选择的样本加入标注集
    labeled_indices = list(set(initial_labeled_indices) | set(uncertain_indices))
    labeled_data = torch.utils.data.Subset(train_dataset, labeled_indices)
    train_loader = torch.utils.data.DataLoader(labeled_data, batch_size=batch_size, shuffle=True)

    # 更新未标注数据
    unlabeled_indices = list(set(range(len(train_dataset))) - set(labeled_indices))
    unlabeled_data = torch.utils.data.Subset(train_dataset, unlabeled_indices)

    # 绘制并保存折线图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 2), train_losses, label="Train Loss")
    plt.plot(range(1, epoch + 2), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 2), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, epoch + 2), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_plot.png")
    plt.savefig(plot_path)
    plt.close()  # 关闭当前画布
