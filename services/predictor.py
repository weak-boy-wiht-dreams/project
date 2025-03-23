import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
from services.cache import TTLCache
import logging
import time
import json

logger = logging.getLogger(__name__)

class ConcurrentPredictor:
    def __init__(self, config: Dict):
        self.config = config
        self.data_cache = TTLCache(
            maxsize=config.get('cache_maxsize', 1000),
            ttl=config.get('cache_ttl', 300)
        )
        self.model_executor = ThreadPoolExecutor(
            max_workers=config.get('model_workers', 8)
        )
        self.data_semaphore = asyncio.Semaphore(
            config.get('max_concurrent_sources', 50)
        )

    async def process_data_source(
        self, 
        data_source: Dict,
        model_map: Dict[str, any]
    ) -> List[Dict]:
        """处理单个数据源的多模型预测"""
        async with self.data_semaphore:
            # 缓存最新数据
            self.data_cache.set(data_source['id'], data_source['data'])
             
            tasks = []
            for model_id, model in model_map.items():
                task = asyncio.create_task(
                    self._predict_with_retry(
                        model=model,
                        data=data_source['data'],
                        ds_id=data_source['id'],
                        model_id=model_id
                    ) 
                )
                tasks.append(task)
                
            results = await asyncio.gather(*tasks)
            return [res for res in results if res is not None]

    async def _predict_with_retry(self, model, data, ds_id, model_id, retries=3):
        """带重试机制的预测"""
        for attempt in range(retries):
            try:
                result = await asyncio.wait_for(
                    model.async_predict(data),
                    timeout=self.config.get('predict_timeout', 5.0)
                )
                return self._format_result(ds_id, model_id, result)
            except asyncio.TimeoutError:
                logger.warning(f"模型预测超时: {model_id} (尝试 {attempt+1}/{retries})")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"预测失败: {str(e)}")
                return self._format_error(ds_id, model_id, str(e))
        
        return self._format_error(ds_id, model_id, "超过最大重试次数")

    def _format_result(self, ds_id, model_id, result):
        return {
            "data_source_id": ds_id,
            "model_id": model_id,
            "timestamp": time.time(),
            "result": result
        }

    def _format_error(self, ds_id, model_id, error_msg):
        return {
            "data_source_id": ds_id,
            "model_id": model_id,
            "timestamp": time.time(),
            "error": error_msg
        }