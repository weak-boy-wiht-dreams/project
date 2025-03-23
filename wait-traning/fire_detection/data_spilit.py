import os
import shutil
import random

# 设置源数据集路径
source_path = './data/Fire-Detection'

# 设置目标文件夹路径
train_dir = './train'
val_dir = './val'

# 创建目标文件夹
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 获取原始数据集中的文件夹（0 和 1）
for folder in ['0', '1']:
    source_folder = os.path.join(source_path, folder)
    
    # 获取文件夹中的所有图像文件
    images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # 打乱文件顺序
    random.shuffle(images)
    
    # 划分训练集和验证集的比例（例如：80%训练集，20%验证集）
    split_index = int(0.8 * len(images))  # 80% 用于训练
    
    # 划分训练集和验证集
    train_images = images[:split_index]
    val_images = images[split_index:]
    
    # 将训练集和验证集的图像复制到目标文件夹
    for image in train_images:
        shutil.copy(os.path.join(source_folder, image), os.path.join(train_dir, folder + '_' + image))
    
    for image in val_images:
        shutil.copy(os.path.join(source_folder, image), os.path.join(val_dir, folder + '_' + image))

print("数据集划分完成！")
