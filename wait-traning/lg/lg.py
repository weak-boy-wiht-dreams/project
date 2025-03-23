import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 生成数据
def generate_data(n_samples: int = 1000, n_features: int = 3, n_classes: int = 2):
    """生成模拟数据，n_samples 样本数量，n_features 特征数量，n_classes 类别数量"""
    X = np.random.rand(n_samples, n_features)  # 随机生成特征
    y = np.random.randint(0, n_classes, n_samples)  # 随机生成标签
    return X, y

# 2. 训练逻辑回归模型
def train_logistic_regression(X, y):
    """训练逻辑回归模型"""
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

# 3. 保存模型
def save_model(model, model_path: str):
    """保存训练好的模型到文件"""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

# 4. 加载模型
def load_model(model_path: str):
    """加载保存的模型"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# 5. 使用模型进行预测
def predict(model, X):
    """使用训练好的模型进行预测"""
    return model.predict(X)

# 6. 保存数据集,方便后续测试
def save_dataset(X, y, dataset_path: str):
    """保存生成的数据集到文件"""
    data = np.column_stack((X, y))  # 合并特征和标签
    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(X.shape[1])] + ["label"])
    df.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path}")

# 示例流程
if __name__ == "__main__":
    # 生成数据
    X, y = generate_data(n_samples=1000, n_features=3, n_classes=2)
    
    # 保存数据集
    save_dataset(X, y, 'dataset.csv')

    # 训练模型
    model = train_logistic_regression(X, y)

    # 保存模型
    save_model(model, 'logistic_regression_model.pkl')

    # 加载模型
    loaded_model = load_model('logistic_regression_model.pkl')

    # 使用加载的模型进行预测
    sample_data = np.array([[0.5, 0.2, 0.8]])  # 新的样本数据
    prediction = predict(loaded_model, sample_data)

    print(f"Prediction for sample data: {prediction}")
