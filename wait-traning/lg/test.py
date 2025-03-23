import numpy as np
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self):
        # 创建一个简单的逻辑回归模型并训练它
        self._model = LogisticRegression()
        # 假设我们有一个训练集
        X_train = np.array([[210, 9, 33, 1980, 0], [220, 10, 35, 2200, 1]])  # 5个特征
        y_train = np.array([0, 1])  # 预测标签
        
        # 训练模型
        self._model.fit(X_train, y_train)

    def predict(self, inputs):
        # 确保数据是二维的
        inputs = inputs.reshape(1, -1)  # 调整输入数据的形状，确保它是 (1, n_features)
        # 使用预测函数
        prediction = self._model.predict(inputs)
        return prediction

# 创建模型实例
model = LogisticRegressionModel()

# 假设这是需要进行预测的单个样本
single_sample = np.array([210, 9, 33, 1980, 0])  # 5个特征的输入
print(single_sample)
# 进行预测
prediction = model.predict(single_sample)

print(f"Prediction: {prediction}")
