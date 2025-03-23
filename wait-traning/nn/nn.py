import numpy as np
from sklearn.neural_network import MLPClassifier  # 导入神经网络分类器
import pickle

# 创建一个简单的数据集
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # 输入特征
y = np.array([0, 1, 0, 1])  # 目标标签

# 初始化并训练一个简单的神经网络模型
model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000)
model.fit(X, y)

# 保存模型为 pkl 文件
with open("nn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Neural network model has been saved as nn_model.pkl")
