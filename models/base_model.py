import pickle
import numpy as np
import asyncio
from abc import ABC ,abstractmethod
from typing import Any,Dict
#用于记录日志的库
import logging
#方便文件路径处理的库
from pathlib import Path


logger=logging.getLogger(__name__)

class BaseModel(ABC):
    _model_cache={}#缓存加载过的模型

    def __init__(self,model_path :str):
        #判断有无属性的函数
        if not hasattr(self,'_model'):
            self._model=self.load_model(model_path)
        self._validate_model()

    @classmethod
    def load_model(cls,model_path:str):
        #resolve（）强转换为promise对象
        canonical_path=str(Path(model_path).resolve())

        if canonical_path in cls._model_cache:
            logger.debug(f"复用缓存模型：{canonical_path}")
            return cls._model_cache[canonical_path]
        
        try:
            with open(model_path,'rb') as f:
                model=pickle.load(f)
            cls._model_cache[canonical_path]=model
            logger.info(f"成功加载模型: {canonical_path}")
            return model

        except Exception as e:
            logger.error(f"模型加载失败，请检查路径等信息{str(e)}")
            raise

    def _validate_model(self):
        """验证模型有效性"""
        if not hasattr(self._model, "predict"):
            raise ValueError("模型必须实现 predict 方法")
        #if not hasattr(self._model, "predict_proba"):
            #logger.warning("模型未实现概率预测方法")

    @abstractmethod
    def preprocess(self, raw_data: Dict) -> Any:
        """数据预处理（子类实现）"""
        pass

    @abstractmethod
    def postprocess(self, prediction) -> Dict:
        """结果后处理（子类实现）"""
        pass

    async def async_predict(self,raw_data:Dict)->Dict:
        '''异步预测'''
        """
        异步执行模型预测，支持批量数据和单个数据预测
        :param raw_data: 原始数据
        :return: 预测结果或错误信息
        """
        try:
            inputs=self.preprocess(raw_data)
            #获取当前事件的循环对象
            loop=asyncio.get_event_loop()
            print(raw_data)
            print(inputs)
            #支持批量预测
            if isinstance(inputs,list):#检查是否是list类型
                print('1')
                predictions=await loop.run_in_executor(
                    None,
                    lambda:self._model.predict(np.array(inputs))
                )
                return [self.postprocess(p) for p in predictions]
            else:#如果是单个数据
                print('2')
                print(inputs)

                #input_data = np.array([210.0, 9.0, 33.0, 1980.0, 0.0])
                #input_data = input_data.reshape(1, -1)  # 确保它是二维数组形状 (1, 5)
                #print(input_data.shape) 
                #prediction = self._model.predict(input_data)
                #print("预测结果",prediction)
                #print(self._model)
                
                prediction = await loop.run_in_executor(
                    None,
                    lambda: self._model.predict(inputs)
                    #lambda: self._model.predict(inputs)
                )
                
                
                return self.postprocess(prediction)
                
        except Exception as e:
            print("3")
            logger.error(f"预测失败: {str(e)}", exc_info=True)
            return {"error": str(e)}

