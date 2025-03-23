import numpy as np
from models.base_model import BaseModel
from typing import Dict, Any

class LogisticRegressionModel(BaseModel):
    #特征字段
    #REQUIRED_FEATURES = ['voltage', 'current', 'temperature']
    #特征顺序
    #FEATURE_ORDER = ['voltage', 'current', 'temperature']
    REQUIRED_FEATURES = ['voltage', 'current', 'temperature', 'power', 'status']
    FEATURE_ORDER = ['voltage', 'current', 'temperature', 'power', 'status']
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._validate_feature_order()

    def _validate_feature_order(self):
        """确保特征顺序一致性"""
        if len(self.FEATURE_ORDER) != len(set(self.FEATURE_ORDER)):
            raise ValueError("特征顺序配置包含重复项")

    def preprocess(self, raw_data: Dict) -> np.ndarray:
        # 特征存在性检查
        missing = set(self.REQUIRED_FEATURES) - set(raw_data.keys())
        if missing:
            raise ValueError(f"缺少必要特征: {missing}")
        
        # 按预定顺序构建特征数组
        try:
            data = np.array([
                float(raw_data[self.FEATURE_ORDER[0]]),
                float(raw_data[self.FEATURE_ORDER[1]]),
                float(raw_data[self.FEATURE_ORDER[2]]),
                float(raw_data[self.FEATURE_ORDER[3]]),
                float(raw_data[self.FEATURE_ORDER[4]])
            ])
            # 返回二维数组，即使只有一个样本
            return data.reshape(1, -1)  # 将其转换为二维数组 (1, n_features)
        except (ValueError, TypeError) as e:
            raise ValueError("特征值格式错误") from e

    def postprocess(self, prediction) -> Dict:
        # 处理numpy数据类型
        class_label = int(prediction[0])
        '''
        # 获取概率值
        if hasattr(self._model, 'predict_proba'):
            proba = self._model.predict_proba(prediction.reshape(1, -1))[0]
            probabilities = {
                str(cls): float(prob) 
                for cls, prob in enumerate(proba)
            }
        else:
            probabilities = None
        '''
        return {
            "predicted_status": class_label,
            #"probabilities": probabilities,
            #"model_type": "logistic_regression"
        }

