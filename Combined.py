import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt

# 超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 50
pretrain_epochs = 20
output_dir = "./training_plots"
initial_label_percent = 0.1  # 初始标注数据比例
query_percent_per_cycle = 0.05  # 每次主动学习的查询比例
num_cycles = 5

# 创建保存目录
os.makedirs(output_dir, exist_ok=True)

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
unlabeled_indices = list(range(len(train_dataset)))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义SimCLR的基础模型
class SimCLR(nn.Module):
    def __init__(self, backbone):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z


# 定义分类器
class Classifier(nn.Module):
    def __init__(self, feature_extractor):
        super(Classifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        with torch.no_grad():
            features, _ = self.feature_extractor(x)
        return self.classifier(features)


# 定义对比学习损失
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        sim_matrix.fill_diagonal_(-float('inf'))

        positive_pairs = self.cos_sim(z_i, z_j) / self.temperature
        loss = -positive_pairs.mean() + torch.logsumexp(sim_matrix, dim=-1).mean()
        return loss


# 自监督预训练
def pretrain(model, loader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = torch.cat([images, transforms.RandomHorizontalFlip()(images)], dim=0)
            images = images.to(device)
            optimizer.zero_grad()

            _, z = model(images)
            loss = criterion(z[:len(images) // 2], z[len(images) // 2:])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Pretrain Epoch [{epoch + 1}/{epochs}] Loss: {running_loss / len(loader):.4f}")


# 主动学习采样
def active_sample(unlabeled_indices, model, loader, query_percent, device):
    model.eval()
    uncertainties = []
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            prob = torch.softmax(outputs, dim=1)
            uncertainty = -torch.sum(prob * torch.log(prob + 1e-5), dim=1)  # 信息熵
            uncertainties.extend(uncertainty.cpu().numpy())

    # 选择最不确定的样本
    num_query = int(len(unlabeled_indices) * query_percent)
    query_indices = np.argsort(uncertainties)[-num_query:]
    return query_indices


# 主训练过程
def train_model(model, labeled_loader, optimizer, criterion, device, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in labeled_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Train Epoch [{epoch + 1}/{epochs}] Loss: {running_loss / len(labeled_loader):.4f}")


# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = models.resnet18(pretrained=False)
backbone.fc = nn.Identity()
simclr_model = SimCLR(backbone).to(device)
classifier_model = Classifier(simclr_model).to(device)

# 自监督预训练
optimizer = optim.Adam(simclr_model.parameters(), lr=learning_rate)
criterion = NTXentLoss()
pretrain(simclr_model, train_loader, optimizer, criterion, pretrain_epochs, device)

# 主动学习
labeled_indices = np.random.choice(unlabeled_indices, int(len(unlabeled_indices) * initial_label_percent), replace=False)
unlabeled_indices = list(set(unlabeled_indices) - set(labeled_indices))

labeled_loader = torch.utils.data.DataLoader(Subset(train_dataset, labeled_indices), batch_size=batch_size, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier_model.parameters(), lr=learning_rate)

for cycle in range(num_cycles):
    print(f"Active Learning Cycle {cycle + 1}/{num_cycles}")
    train_model(classifier_model, labeled_loader, optimizer, criterion, device, num_epochs)

    # 采样新的数据
    query_indices = active_sample(unlabeled_indices, classifier_model, torch.utils.data.DataLoader(
        Subset(train_dataset, unlabeled_indices), batch_size=batch_size, shuffle=False), query_percent_per_cycle, device)
    labeled_indices.extend([unlabeled_indices[i] for i in query_indices])
    unlabeled_indices = [i for i in unlabeled_indices if i not in query_indices]
    labeled_loader = torch.utils.data.DataLoader(Subset(train_dataset, labeled_indices), batch_size=batch_size, shuffle=True)

print("Active Learning Completed!")
