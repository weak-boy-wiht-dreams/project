import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib  # 用于保存和加载模型

# 设置列名
col_names = ["ID", "K1K2电压源", "电流电压源", "断路器电压", "电流电压", "THDV-M", "THDI-M", "label"]

# 读取训练数据
data = pd.read_csv("data_train.csv", names=col_names)

# 准备特征和标签数据
dataset_X = data[["K1K2电压源", "电流电压源", "断路器电压", "电流电压", "THDV-M", "THDI-M"]].values
dataset_Y = data[["label"]].values
dataset_Y = dataset_Y.reshape(len(dataset_Y))  # 转换为一维数组

# 拆分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(dataset_X, dataset_Y, test_size=0.2, random_state=42)

# 使用 GradientBoostingClassifier 训练模型
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

# 训练模型
model.fit(x_train, y_train)

# 使用 joblib 保存模型
joblib.dump(model, 'gbm_model.pkl')  # 保存为pkl文件

print("模型训练完成并保存为 'gbm_model.pkl'")
