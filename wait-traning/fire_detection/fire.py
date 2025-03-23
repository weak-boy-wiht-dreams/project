import torch
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import shutil
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自动分类图像文件
def organize_dataset(source_path):
    fire_dir = os.path.join(source_path, 'fire')
    not_fire_dir = os.path.join(source_path, 'not_fire')
    
    os.makedirs(fire_dir, exist_ok=True)
    os.makedirs(not_fire_dir, exist_ok=True)

    for filename in os.listdir(source_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            if filename.startswith('1_'):  # 文件名以1_开头表示火灾图像
                shutil.move(os.path.join(source_path, filename), os.path.join(fire_dir, filename))
            elif filename.startswith('0_'):  # 文件名以0_开头表示非火灾图像
                shutil.move(os.path.join(source_path, filename), os.path.join(not_fire_dir, filename))

# 数据集路径
train_dir = './train'
val_dir = './val'

# 对训练和验证数据集进行分类整理
organize_dataset(train_dir)
organize_dataset(val_dir)

# 加载模型
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

# 设置数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = ImageFolder(root=train_dir, transform=transform)
val_dataset = ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total * 100
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# 验证模型
model.eval()
val_correct = 0
val_total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        val_correct += (predicted == labels).sum().item()
        val_total += labels.size(0)

val_acc = val_correct / val_total * 100
print(f"Validation Accuracy: {val_acc:.2f}%")

# 保存训练后的模型
with open('resnet_model.pkl', 'wb') as f:
    pickle.dump(model, f)
