import time
#字典子类实现插入顺序
from collections import OrderedDict
from typing import Optional
from typing import Any
#本身可以说是一个先进先出的算法实现
class TTLCache:
    def __init__(self, maxsize=1000, ttl=300):
        self.maxsize = maxsize#缓存最大容量，默认1000
        self.ttl = ttl#缓存过期事件
        self._cache = OrderedDict()

    def set(self, key, value):
        self._cache[key] = (time.time(), value)# 存储数据，时间戳和数据一起存储
        self._cache.move_to_end(key)# 存储数据，时间戳和数据一起存储
        self._check_size()# 检查缓存是否超过最大容量，如果超过，则删除最老的数据

    def get(self, key) -> Optional[Any]:
        if key not in self._cache:
            return None
            
        timestamp, value = self._cache[key]
        if time.time() - timestamp > self.ttl:
            del self._cache[key]
            return None
            
        self._cache.move_to_end(key)
        return value

    def _check_size(self):
        while len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)