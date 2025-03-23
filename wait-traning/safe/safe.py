# safe_with_csv.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os
import logging
from typing import Tuple

# 配置日志
logging.basicConfig(level=logging.INFO)

def generate_and_save_data(num_samples: int, seq_length: int, csv_path: str):
    """生成并保存模拟数据到CSV"""
    data_list = []
    
    for sample_id in range(num_samples):
        fault_type = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
        base_current = 10 + np.random.normal(0, 0.2)
        base_voltage = 220 + np.random.normal(0, 2)
        
        for t in range(seq_length):
            # 生成带物理意义的数据
            if fault_type == 1:  # 接触不良
                current = base_current * (0.8 + 0.4 * np.sin(t/5))
                voltage = base_voltage * (0.9 + 0.2 * np.random.rand())
            elif fault_type == 2:  # 电池故障
                current = base_current * (0.6 + 0.1 * t/seq_length)
                voltage = base_voltage * (0.7 + 0.1 * np.random.rand())
            else:  # 正常
                current = base_current + np.random.normal(0, 0.1)
                voltage = base_voltage + np.random.normal(0, 0.5)
            
            data_list.append({
                "sample_id": sample_id,
                "timestep": t,
                "current(A)": current,
                "voltage(V)": voltage,
                "label": fault_type
            })
    
    # 保存为CSV
    df = pd.DataFrame(data_list)
    df.to_csv(csv_path, index=False)
    logging.info(f"数据已保存至 {csv_path}，共 {len(df)} 行")
    return df

def load_and_preprocess(csv_path: str, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """从CSV加载并预处理数据"""
    df = pd.read_csv(csv_path)
    
    # 验证数据完整性
    assert set(df.columns) == {"sample_id", "timestep", "current(A)", "voltage(V)", "label"}, "CSV列名不匹配"
    
    # 按样本ID分组
    grouped = df.groupby("sample_id")
    
    # 转换为三维数组 (samples, timesteps, features)
    X = []
    y = []
    for sample_id, group in grouped:
        # 确保时间步顺序正确
        group = group.sort_values("timestep")
        
        # 检查时间步数量
        if len(group) != seq_length:
            logging.warning(f"样本 {sample_id} 时间步不完整，已跳过")
            continue
        
        # 提取特征和标签
        features = group[["current(A)", "voltage(V)"]].values
        label = group["label"].iloc[0]  # 所有时间步标签相同
        
        X.append(features)
        y.append(label)
    
    return np.array(X), np.array(y)

class LSTMFaultDetector(nn.Module):
    """带注意力机制的LSTM模型"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1, bias=False),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden]
        
        # 注意力机制
        attn_weights = self.attention(lstm_out)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return self.fc(context)

def train_and_save(csv_path: str = "battery_data.csv"):
    # 生成或加载数据
    if not os.path.exists(csv_path):
        logging.info("生成模拟数据...")
        generate_and_save_data(
            num_samples=5000,
            seq_length=60,
            csv_path=csv_path
        )
    
    # 加载并预处理
    X, y = load_and_preprocess(csv_path, seq_length=60)
    logging.info(f"数据形状: X={X.shape}, y={y.shape}")
    
    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, 2)).reshape(X.shape)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 创建DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test), 
        torch.LongTensor(y_test)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, pin_memory=True
    )

    # 初始化模型
    model = LSTMFaultDetector(
        input_size=2,
        hidden_size=128,
        num_layers=2,
        num_classes=3
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 2.0], device=model.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    # 训练循环
    best_acc = 0.0
    for epoch in range(100):
        model.train()
        total_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)

        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(model.device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
        
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        scheduler.step(accuracy)
        
        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            save_data = {
                'state_dict': model.state_dict(),
                'scaler': scaler,
                'input_shape': (60, 2),
                'class_weights': criterion.weight.cpu().numpy(),
                'metadata': {
                    'accuracy': accuracy,
                    'epoch': epoch
                }
            }
            torch.save(save_data, "best_model.pth")
            logging.info(f"Epoch {epoch}: 最佳模型已保存，准确率 {accuracy:.4f}")

        # 打印分类报告
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}")
            print(f"Train Loss: {total_loss / len(train_dataset):.4f}")
            print(classification_report(all_labels, all_preds, 
                                      target_names=['正常', '接触不良', '电池故障']))

if __name__ == "__main__":
    # 检查CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info(f"使用GPU加速: {torch.cuda.get_device_name(0)}")
    
    # 创建输出目录
    os.makedirs("data", exist_ok=True)
    
    # 训练流程
    train_and_save(csv_path="data/battery_data.csv")
    logging.info("训练完成，最佳模型已保存为 best_model.pth")