from aiohttp import web
import json
import logging
from services.auth import AuthValidator
from services.predictor import ConcurrentPredictor
from typing import Dict,Any
import asyncio

logger = logging.getLogger(__name__)

class WebSocketServer:
    def __init__(self, config: Dict, model_map: Dict):
        self.config = config
        self.model_map = model_map
        self.predictor = ConcurrentPredictor(config)
        self.auth = AuthValidator(config)
        self.connections = set()

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        # 认证检查
        if not await self.auth.validate_request(request):
            return web.Response(text="Unauthorized", status=401)

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.connections.add(ws)
        
        try:
            async for msg in ws:
                await self._handle_message(ws, msg)
        finally:
            self.connections.remove(ws)
            
        return ws

    async def _handle_message(self, ws, msg):
        if msg.type == web.WSMsgType.TEXT:
            try:
                payload = json.loads(msg.data)
                if not self._validate_payload(payload):
                    await ws.send_json({"error": "Invalid payload format"})
                    return
                    
                tasks = [
                    self.predictor.process_data_source(ds, self.model_map)
                    for ds in payload['data_sources']
                ]
                
                for future in asyncio.as_completed(tasks):
                    results = await future
                    for result in results:
                        await ws.send_json(result)
                        
            except json.JSONDecodeError:
                await ws.send_json({"error": "Invalid JSON format"})
            except Exception as e:
                logger.error(f"处理消息失败: {str(e)}")
                await ws.send_json({"error": "Internal server error"})

    def _validate_payload(self, payload: Dict) -> bool:
        required_fields = {'data_sources': list}
        return all(
            isinstance(payload.get(k), v)
            for k, v in required_fields.items()
        )

    async def broadcast(self, message: Dict):
        """广播消息给所有连接"""
        for ws in self.connections:
            if not ws.closed:
                await ws.send_json(message)