import asyncio
import json
import aiohttp  # 使用 aiohttp 库来建立 WebSocket 客户端连接

async def send_test_data():
    uri = "ws://localhost:8080/ws"  # WebSocket 服务器的 URI
    token = "your-secure-api-key"  # 密钥

    # 假设你的新模型需要5个特征（`voltage`, `current`, `temperature`, `power`, `status`）
    data_sources = [
        {
            "id": "charging_station_1",
            "data": {
                "voltage": 220,   # 电压
                "current": 10,    # 电流
                "temperature": 35,  # 温度
                "power": 2200,     # 功率（假设添加的额外特征）
                "status": 1        # 状态（假设添加的额外特征）
            }
        },
        {
            "id": "chargi   ng_station_2",
            "data": {
                "voltage": 210,
                "current": 9,
                "temperature": 33,
                "power": 1980,     # 功率
                "status": 0        # 状态
            }
        }
    ]

    # 设置 WebSocket 连接时的认证头
    headers = {
        'Authorization': f'Bearer {token}'  # 如果是 Bearer Token 认证
    }

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(uri, headers=headers) as ws:
            # 持续发送数据并接收返回的预测结果
            while True:
                # 构建发送的消息
                message = {
                    "data_sources": data_sources
                }

                # 向服务器发送测试数据
                await ws.send_json(message)
                print("Sent data:", json.dumps(message))

                # 持续接收从服务器返回的预测结果
                msg = await ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    print("Received response:", msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print("Error in receiving message.")
                    break

                await asyncio.sleep(1)  # 控制发送间隔时间，避免发送过于频繁

# 运行客户端
asyncio.get_event_loop().run_until_complete(send_test_data())

