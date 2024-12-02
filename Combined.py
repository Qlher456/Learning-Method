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
pretrain_epochs = 20
num_epochs = 100
initial_label_percent = 0.1
query_percent_per_cycle = 0.05
num_cycles = 5
output_dir = "./training_plots"

os.makedirs(output_dir, exist_ok=True)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# 模型定义
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

class Classifier(nn.Module):
    def __init__(self, feature_extractor):
        super(Classifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        with torch.no_grad():
            features, _ = self.feature_extractor(x)
        return self.classifier(features)

# 损失函数
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        z = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.mm(z, z.t())
        labels = torch.arange(z_i.size(0)).repeat(2).to(z.device)
        loss = nn.CrossEntropyLoss()(similarity_matrix / self.temperature, labels)
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
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            prob = torch.softmax(outputs, dim=1)
            uncertainty = -torch.sum(prob * torch.log(prob + 1e-5), dim=1)
            uncertainties.extend(uncertainty.cpu().numpy())

    num_query = int(len(unlabeled_indices) * query_percent)
    query_indices = np.argsort(uncertainties)[-num_query:]
    return [unlabeled_indices[i] for i in query_indices]

# 训练与测试
def train_model(model, loader, optimizer, criterion, device):
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
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return running_loss / len(loader), accuracy

def test_model(model, loader, criterion, device):
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
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return running_loss / len(loader), accuracy

# 主动学习循环
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = models.resnet18(pretrained=False)
backbone.fc = nn.Identity()
simclr_model = SimCLR(backbone).to(device)
classifier_model = Classifier(simclr_model).to(device)

# 预训练
optimizer = optim.Adam(simclr_model.parameters(), lr=learning_rate)
criterion = NTXentLoss()
pretrain(simclr_model, torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True), optimizer, criterion, pretrain_epochs, device)

# 主动学习
labeled_indices = np.random.choice(len(train_dataset), int(len(train_dataset) * initial_label_percent), replace=False).tolist()
unlabeled_indices = list(set(range(len(train_dataset))) - set(labeled_indices))

train_loader = torch.utils.data.DataLoader(Subset(train_dataset, labeled_indices), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

optimizer = optim.Adam(classifier_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for cycle in range(num_cycles):
    print(f"Active Learning Cycle {cycle + 1}/{num_cycles}")
    train_loss, train_acc = train_model(classifier_model, train_loader, optimizer, criterion, device)
    test_loss, test_acc = test_model(classifier_model, test_loader, criterion, device)

    print(f"Cycle {cycle + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Cycle {cycle + 1}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    query_indices = active_sample(unlabeled_indices, classifier_model, torch.utils.data.DataLoader(Subset(train_dataset, unlabeled_indices), batch_size=batch_size, shuffle=False), query_percent_per_cycle, device)
    labeled_indices.extend(query_indices)
    unlabeled_indices = [i for i in unlabeled_indices if i not in query_indices]
    train_loader = torch.utils.data.DataLoader(Subset(train_dataset, labeled_indices), batch_size=batch_size, shuffle=True)
