import yaml
from aiohttp import web
from models.logistic_model import LogisticRegressionModel
from models.nn_model import NNModel
from services.websocket_server import WebSocketServer
import logging
from typing import Any,Dict
def load_config() -> Dict:
    with open('configs/settings.yaml', encoding="utf-8") as f:
        return yaml.safe_load(f)

def create_app() -> web.Application:
    # 初始化配置
    config = load_config()
    
    # 配置日志
    logging.basicConfig(level=config.get('log_level', 'INFO'))
    print("config['models'] 的键列表:", config['models'].keys())
    # 初始化模型
    model_map = {
        "lr_charge_status": LogisticRegressionModel(
            config['models']['logistic_regression_path']
        ),
        "nn_charge_status":NNModel(
            config['models']['nn_path']
        )

    }
    
    # 创建WebSocket服务
    ws_server = WebSocketServer(config, model_map)
    
    # 配置路由
    app = web.Application()
    app.router.add_get('/ws', ws_server.websocket_handler)
    
    return app

if __name__ == '__main__':
    web.run_app(create_app(), 
                port=8080,
                access_log_format='%a %t "%r" %s %b "%{User-Agent}i"')