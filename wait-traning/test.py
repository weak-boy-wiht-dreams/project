import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

# 加载模型
with open('pretrained/logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 假设这是需要进行预测的单个样本
input_data = np.array([210.0, 9.0, 33.0, 1980.0, 0.0])  # 5个特征

# 调整为二维数组 (1, 5) - 一个样本，5个特征
input_data = input_data.reshape(1, -1)

# 进行预测
prediction = model.predict(input_data)

print(f"Prediction: {prediction}")
