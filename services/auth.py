from aiohttp import web
from typing import Dict

class AuthValidator:
    def __init__(self, config: Dict):
        self.api_keys = config.get('api_keys', [])
        self.enable_auth = config.get('enable_auth', False)

    async def validate_request(self, request: web.Request) -> bool:
        if not self.enable_auth:
            return True
            
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return False
            
        token = auth_header[7:].strip()
        return token in self.api_keys