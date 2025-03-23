import joblib
import pandas as pd

# 加载保存的 XGBoost 模型
model = joblib.load('gbm_model.pkl')
x_test=pd.read_csv("data_test.csv")
x_test = x_test.iloc[:, 1:]
# 假设你有新的测试数据 x_test
y_pred = model.predict(x_test)

# 输出预测结果
print("Predictions:", y_pred)
