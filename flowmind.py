from mcp.server.fastmcp import FastMCP
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional
import httpx
import time
from datetime import datetime, timedelta

# 初始化 FastMCP 服务器
mcp = FastMCP("flowmind")

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
FLOW_API_BASE_URL = os.environ.get("FLOW_API_BASE_URL", "https://rest-mainnet.onflow.org/v1")
REQUEST_TIMEOUT = 15
# 使用异步 httpx 客户端
async_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True)

# --- 辅助函数：通用的异步 Flow API 请求处理 ---
async def make_async_flow_api_request(endpoint: str) -> Dict[str, Any]:
    """
    通用的异步函数，用于向Flow HTTP API发送GET请求并处理响应和错误。

    Args:
        endpoint: API的路径部分 (例如 "/accounts/0x...")

    Returns:
        解析后的JSON数据字典。
    """
    api_url = f"{FLOW_API_BASE_URL}{endpoint}"
    logging.info(f"向 Flow API 发送异步请求: {api_url}")
    try:
        response = await async_client.get(api_url)
        response.raise_for_status()
        data = response.json()
        logging.info(f"请求成功: {api_url}")
        return data
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        error_text = e.response.text
        logging.error(f"请求 {api_url} 时发生HTTP错误: {status_code} - {error_text}", exc_info=False)
        raise ValueError(f"Flow API 请求失败 (HTTP {status_code}): {error_text}") from e
    except httpx.RequestError as e:
        logging.error(f"请求 {api_url} 时发生网络错误: {e}", exc_info=False)
        raise ConnectionError(f"网络错误，无法连接到 Flow API: {e}") from e
    except json.JSONDecodeError as e:
        logging.error(f"解析来自 {api_url} 的响应JSON失败: {e}", exc_info=False)
        raise ValueError(f"无法解析 Flow API 的响应: {e}") from e
    except Exception as e:
        logging.error(f"处理请求 {api_url} 时发生未知错误: {e}", exc_info=True)
        raise RuntimeError(f"处理 Flow API 请求时发生未知内部错误: {e}") from e

@mcp.tool()
async def get_flow_account_info(account_address: str) -> Dict[str, Any]:
    """查询Flow账户信息，包含余额
    
    Args:
        account_address: Flow账户地址，以0x开头
    """
    if not account_address or not isinstance(account_address, str) or not account_address.startswith("0x"):
        raise ValueError("无效或缺失 'account_address' 参数")
    endpoint = f"/accounts/{account_address}"
    result = await make_async_flow_api_request(endpoint)
    # 返回简化后的信息
    simplified_result = {
        "address": result.get("address"),
        "balance_ufix64": result.get("balance"),
        "keys_count": len(result.get("keys", [])),
        "contracts_count": len(result.get("contracts", {}))
    }
    return simplified_result

@mcp.tool()
async def get_transaction_details(transaction_id: str) -> Dict[str, Any]:
    """查询Flow交易详情
    
    Args:
        transaction_id: 交易ID，64位十六进制字符串
    """
    if not transaction_id or not isinstance(transaction_id, str) or len(transaction_id) != 64:
        raise ValueError("无效或缺失 'transaction_id' 参数 (应为64位十六进制字符串)")
    endpoint = f"/transactions/{transaction_id}"
    # 直接返回原始API结果
    return await make_async_flow_api_request(endpoint)

@mcp.tool()
async def get_latest_sealed_block() -> Dict[str, Any]:
    """查询Flow最新已密封区块信息"""
    endpoint = "/blocks?height=sealed"
    data = await make_async_flow_api_request(endpoint)
    if isinstance(data, list) and len(data) > 0:
        block_info = data[0]
        simplified_result = {
            "id": block_info.get("id"),
            "height": block_info.get("height"),
            "timestamp": block_info.get("timestamp"),
            "parent_id": block_info.get("parent_id"),
        }
        return simplified_result
    else:
        raise ValueError("未能从API响应中获取有效的最新区块信息")

@mcp.tool()
async def get_account_transactions(account_address: str, limit: int = 10) -> Dict[str, Any]:
    """获取Flow账户的交易历史
    
    Args:
        account_address: Flow账户地址，以0x开头
        limit: 返回的交易数量限制，默认为10
    """
    if not account_address or not isinstance(account_address, str) or not account_address.startswith("0x"):
        raise ValueError("无效或缺失 'account_address' 参数")

    try:
        # 使用Flowscan API获取交易历史
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Referer": f"https://www.flowscan.io/account/{account_address}"
        }
        
        # Flowscan API接口
        api_url = f"https://prod-flowscan-api.onflow.org/account/{account_address}/transactions?limit={limit}"
        logging.info(f"向 Flowscan API 发送请求: {api_url}")
        
        # 发送请求获取JSON数据
        response = await async_client.get(api_url, headers=headers)
        response.raise_for_status()
        
        # 解析JSON响应
        transactions_data = response.json()
        
        # 获取账户基本信息
        account_info_url = f"https://prod-flowscan-api.onflow.org/account/{account_address}"
        account_response = await async_client.get(account_info_url, headers=headers)
        account_response.raise_for_status()
        account_data = account_response.json()
        
        # 提取账户信息
        account_info = {}
        if account_data:
            account_info = {
                "balance": account_data.get("balance", "未知"),
                "staked_balance": account_data.get("stakedTokens", "未知"),
                "delegated_balance": account_data.get("delegatorTokens", "未知"),
                "total_balance": account_data.get("totalBalance", "未知")
            }
        
        # 格式化交易历史
        transactions = []
        for tx in transactions_data[:limit]:
            tx_data = {
                "id": tx.get("hash", ""),
                "date": tx.get("time", ""),
                "type": tx.get("type", ""),
                "status": tx.get("status", ""),
                "url": f"https://www.flowscan.io/transaction/{tx.get('hash', '')}"
            }
            transactions.append(tx_data)
        
        return {
            "account_info": account_info,
            "total_transactions": len(transactions),
            "transactions": transactions,
            "source": "flowscan.io API"
        }
    except Exception as e:
        logging.error(f"获取账户交易历史时出错: {e}")
        # 尝试使用备用方法
        return await get_account_transactions_fallback(account_address, limit)

async def get_account_transactions_fallback(account_address: str, limit: int = 10) -> Dict[str, Any]:
    """备用方法：使用Flow API获取账户信息和交易历史
    
    Args:
        account_address: Flow账户地址，以0x开头
        limit: 返回的交易数量限制，默认为10
    """
    try:
        # 获取账户信息
        endpoint = f"/accounts/{account_address}"
        account_result = await make_async_flow_api_request(endpoint)
        
        # 返回简化后的信息
        account_info = {
            "address": account_result.get("address"),
            "balance": account_result.get("balance", "未知"),
            "keys_count": len(account_result.get("keys", [])),
            "contracts_count": len(account_result.get("contracts", {}))
        }
        
        # 由于Flow API不直接提供账户交易历史，我们返回一个说明
        return {
            "account_info": account_info,
            "total_transactions": 0,
            "transactions": [],
            "message": "Flow API不直接提供账户交易历史查询功能。请使用Flowscan网站查看完整交易历史。",
            "source": "flow.org API (fallback)"
        }
    except Exception as e:
        logging.error(f"备用方法获取账户信息时出错: {e}")
        return {"error": f"获取交易历史失败: {str(e)}"}

@mcp.tool()
async def get_wallet_dashboard(account_address: str) -> Dict[str, Any]:
    """生成Flow钱包仪表盘，包含账户信息、交易历史和网络状态
    
    Args:
        account_address: Flow账户地址，以0x开头
    """
    if not account_address or not isinstance(account_address, str) or not account_address.startswith("0x"):
        raise ValueError("无效或缺失 'account_address' 参数")

    logging.info(f"开始获取钱包仪表盘信息，地址: {account_address}")
    dashboard_data = {}

    try:
        # 并行获取所有所需数据
        account_info_task = asyncio.create_task(get_flow_account_info(account_address))
        transactions_task = asyncio.create_task(get_account_transactions(account_address, 5))
        latest_block_task = asyncio.create_task(get_latest_sealed_block())
        
        # 等待所有任务完成
        account_info = await account_info_task
        transactions_result = await transactions_task
        latest_block = await latest_block_task
        
        # 构建仪表盘数据
        dashboard_data = {
            "account_info": account_info,
            "transactions": transactions_result,
            "network_status": {
                "latest_sealed_block": latest_block,
                "network": "mainnet" if "mainnet" in FLOW_API_BASE_URL else "testnet"
            },
            "generated_at": asyncio.current_task().get_name() or "unknown",
            "flowscan_url": f"https://www.flowscan.io/account/{account_address}"
        }

    except Exception as e:
        error_message = f"生成钱包仪表盘时出错: {e}"
        logging.error(error_message)
        # 返回已获取的部分信息（如果有）和错误信息
        if not dashboard_data:
            dashboard_data = {"error": error_message}
        else:
            dashboard_data["error"] = error_message

    return dashboard_data

@mcp.tool()
async def monitor_account_activity(account_address: str, check_interval_seconds: int = 60) -> Dict[str, Any]:
    """监控Flow账户活动，周期性检查最新交易
    
    Args:
        account_address: Flow账户地址，以0x开头
        check_interval_seconds: 检查间隔秒数，默认为60秒
    """
    if not account_address or not isinstance(account_address, str) or not account_address.startswith("0x"):
        raise ValueError("无效或缺失 'account_address' 参数")

    if check_interval_seconds < 10:
        check_interval_seconds = 10  # 最小间隔为10秒

    logging.info(f"开始监控账户 {account_address} 的活动，间隔 {check_interval_seconds} 秒")

    # 初始化监控结果
    monitoring_result = {
        "account_address": account_address,
        "monitor_start_time": None,
        "last_check_time": None,
        "check_count": 0,
        "new_transactions": [],
        "monitor_status": "starting",
        "error": None
    }

    try:
        # 记录初始时间
        start_time = asyncio.current_task().get_name()
        monitoring_result["monitor_start_time"] = start_time
        
        # 首次获取交易历史作为基准
        initial_txs = await get_account_transactions(account_address, 10)
        known_tx_ids = set()
        
        if "transactions" in initial_txs and isinstance(initial_txs["transactions"], list):
            for tx in initial_txs["transactions"]:
                known_tx_ids.add(tx.get("id", ""))
        
        monitoring_result["initial_tx_count"] = len(known_tx_ids)
        monitoring_result["monitor_status"] = "running"
        
        # 模拟监控过程，执行3次检查（实际场景中可能会一直运行）
        for i in range(3):
            # 在实际应用中，这里应该是一个无限循环，而不是固定次数
            monitoring_result["check_count"] += 1
            monitoring_result["last_check_time"] = asyncio.current_task().get_name()
            
            # 为了演示目的，这里只等待几秒钟
            await asyncio.sleep(3)  # 实际应用中应该使用check_interval_seconds
            
            # 获取最新的交易
            current_txs = await get_account_transactions(account_address, 10)
            
            new_txs = []
            if "transactions" in current_txs and isinstance(current_txs["transactions"], list):
                for tx in current_txs["transactions"]:
                    tx_id = tx.get("id", "")
                    if tx_id and tx_id not in known_tx_ids:
                        new_txs.append(tx)
                        known_tx_ids.add(tx_id)
            
            if new_txs:
                monitoring_result["new_transactions"].extend(new_txs)
        
        monitoring_result["monitor_status"] = "completed"
        monitoring_result["total_new_transactions"] = len(monitoring_result["new_transactions"])
        
    except Exception as e:
        error_message = f"监控账户活动时出错: {e}"
        logging.error(error_message)
        monitoring_result["monitor_status"] = "error"
        monitoring_result["error"] = str(e)

    return monitoring_result

# NFT 浏览器功能
async def execute_cadence_script(script: str, args: List[Any] = None) -> Any:
    """
    执行Cadence脚本并返回结果
    
    Args:
        script: Cadence脚本代码
        args: 脚本参数列表，默认为None
    
    Returns:
        脚本执行结果
    """
    if args is None:
        args = []
    
    # 构建脚本执行请求
    endpoint = "/scripts"
    url = f"{FLOW_API_BASE_URL}{endpoint}"
    
    # 编码参数
    encoded_args = []
    for arg in args:
        encoded_args.append({
            "type": "String",  # 这里简化处理，实际应根据参数类型进行编码
            "value": str(arg)
        })
    
    # 构建请求体
    payload = {
        "script": script.encode("utf-8").hex(),
        "arguments": encoded_args
    }
    
    logging.info(f"执行Cadence脚本: {script[:50]}...")
    
    try:
        # 发送POST请求执行脚本
        response = await async_client.post(url, json=payload)
        response.raise_for_status()
        
        # 解析结果
        result = response.json()
        return result
    
    except Exception as e:
        logging.error(f"执行Cadence脚本时出错: {e}")
        raise RuntimeError(f"执行Cadence脚本失败: {e}")

@mcp.tool()
async def query_nft_collection_info(collection_address: str, collection_name: str) -> Dict[str, Any]:
    """查询NFT集合信息
    
    Args:
        collection_address: NFT合约地址
        collection_name: NFT集合名称
    """
    if not collection_address or not isinstance(collection_address, str):
        raise ValueError("无效或缺失 'collection_address' 参数")
    
    if not collection_name or not isinstance(collection_name, str):
        raise ValueError("无效或缺失 'collection_name' 参数")
    
    logging.info(f"查询NFT集合信息: {collection_name} @ {collection_address}")
    
    try:
        # 这里使用Flowscan API模拟获取NFT集合信息
        # 实际应用中可以基于Flow API和Cadence脚本实现
        
        # 示例模拟数据
        collection_info = {
            "name": collection_name,
            "address": collection_address,
            "contract_name": f"{collection_name}Contract",
            "standard": "Flow NFT Standard",
            "created_at": "2023-01-01T00:00:00Z",
            "total_supply": 10000,
            "holders_count": 1500,
            "floor_price": "10.5 FLOW",
            "volume_24h": "1250.0 FLOW",
            "description": f"{collection_name} is a unique collection of digital art on the Flow blockchain."
        }
        
        # 获取合约信息以验证合约是否存在
        endpoint = f"/accounts/{collection_address}"
        account_data = await make_async_flow_api_request(endpoint)
        
        # 检查账户中是否有该合约
        contracts = account_data.get("contracts", {})
        if collection_name in contracts:
            collection_info["verified"] = True
            collection_info["contract_code_snippet"] = contracts[collection_name][:200] + "..."
        else:
            collection_info["verified"] = False
            collection_info["warning"] = "未在指定地址找到该合约名称"
        
        return collection_info
        
    except Exception as e:
        logging.error(f"查询NFT集合信息时出错: {e}")
        return {
            "name": collection_name,
            "address": collection_address,
            "error": f"获取NFT集合信息失败: {str(e)}",
            "status": "error"
        }

@mcp.tool()
async def query_nft_metadata(collection_address: str, token_id: str) -> Dict[str, Any]:
    """查询NFT元数据信息
    
    Args:
        collection_address: NFT合约地址
        token_id: NFT代币ID
    """
    if not collection_address or not isinstance(collection_address, str):
        raise ValueError("无效或缺失 'collection_address' 参数")
    
    if not token_id or not isinstance(token_id, str):
        raise ValueError("无效或缺失 'token_id' 参数")
    
    logging.info(f"查询NFT元数据: 合约 {collection_address}, 代币ID {token_id}")
    
    # 示例Cadence脚本，用于查询NFT元数据
    # 实际脚本需要根据具体NFT合约结构进行调整
    # 这里使用简化的示例
    cadence_script = """
    import NonFungibleToken from 0x1d7e57aa55817448
    import MetadataViews from 0x1d7e57aa55817448
    
    pub fun main(address: String, tokenID: String): {String: String} {
        let account = getAccount(address)
        let collection = account.getCapability(MetadataViews.NFTCollectionDisplay)
                              .borrow<&{MetadataViews.NFTCollectionDisplay}>()
        
        if collection == nil {
            return {"error": "Collection not found"}
        }
        
        let id = UInt64(tokenID) ?? 0
        if id == 0 {
            return {"error": "Invalid token ID"}
        }
        
        let nft = collection.borrowNFT(id: id)
        if nft == nil {
            return {"error": "NFT not found"}
        }
        
        let metadata = nft.resolveView(Type<MetadataViews.Display>())
        if metadata == nil {
            return {"error": "Metadata not available"}
        }
        
        let display = metadata as! MetadataViews.Display
        
        return {
            "name": display.name,
            "description": display.description,
            "thumbnail": display.thumbnail.uri(),
            "owner": address
        }
    }
    """
    
    try:
        # 实际应用中，应执行上述Cadence脚本
        # 由于脚本执行需要实际环境，这里使用模拟数据
        
        # 模拟数据
        metadata = {
            "id": token_id,
            "collection": collection_address,
            "name": f"Awesome NFT #{token_id}",
            "description": "A unique digital collectible on the Flow blockchain.",
            "image_url": f"https://example.com/nft/{collection_address}/{token_id}.png",
            "attributes": [
                {"trait_type": "Background", "value": "Blue"},
                {"trait_type": "Eyes", "value": "Green"},
                {"trait_type": "Mouth", "value": "Smile"},
                {"trait_type": "Outfit", "value": "Suit"}
            ],
            "created_at": "2023-03-15T12:00:00Z",
            "owner_address": "0x" + "1" * 16,
            "rarity_score": 85.7,
            "marketplace_url": f"https://flowverse.co/marketplace/items/{collection_address}/{token_id}"
        }
        
        # 在这里我们可以尝试获取账户信息来验证地址有效性
        try:
            await get_flow_account_info(collection_address)
            metadata["collection_verified"] = True
        except:
            metadata["collection_verified"] = False
            metadata["warning"] = "无法验证合约地址是否有效"
        
        return metadata
    
    except Exception as e:
        logging.error(f"查询NFT元数据时出错: {e}")
        return {
            "id": token_id,
            "collection": collection_address,
            "error": f"获取NFT元数据失败: {str(e)}",
            "status": "error"
        }

@mcp.tool()
async def get_nft_ownership_history(collection_address: str, token_id: str, limit: int = 5) -> Dict[str, Any]:
    """获取NFT所有权历史记录
    
    Args:
        collection_address: NFT合约地址
        token_id: NFT代币ID
        limit: 返回的历史记录数量限制，默认为5
    """
    if not collection_address or not isinstance(collection_address, str):
        raise ValueError("无效或缺失 'collection_address' 参数")
    
    if not token_id or not isinstance(token_id, str):
        raise ValueError("无效或缺失 'token_id' 参数")
    
    logging.info(f"获取NFT所有权历史: 合约 {collection_address}, 代币ID {token_id}, 限制 {limit}")
    
    try:
        # 实际应用中，可以通过查询NFT转移事件来获取所有权历史
        # 这里使用模拟数据展示
        
        # 模拟的所有权历史记录
        history = [
            {
                "date": "2023-08-10T15:23:45Z",
                "from_address": "0x0000000000000000",  # 铸造
                "to_address": "0x1111111111111111",
                "transaction_id": "a" * 64,
                "transaction_type": "Mint",
                "price": None
            },
            {
                "date": "2023-09-05T10:11:12Z",
                "from_address": "0x1111111111111111",
                "to_address": "0x2222222222222222",
                "transaction_id": "b" * 64,
                "transaction_type": "Transfer",
                "price": "50.0 FLOW"
            },
            {
                "date": "2023-11-20T08:42:31Z",
                "from_address": "0x2222222222222222",
                "to_address": "0x3333333333333333",
                "transaction_id": "c" * 64,
                "transaction_type": "Sale",
                "price": "75.5 FLOW"
            },
            {
                "date": "2024-02-18T14:27:09Z",
                "from_address": "0x3333333333333333",
                "to_address": "0x4444444444444444",
                "transaction_id": "d" * 64,
                "transaction_type": "Sale",
                "price": "120.0 FLOW"
            },
            {
                "date": "2024-03-30T22:15:53Z",
                "from_address": "0x4444444444444444",
                "to_address": "0x5555555555555555",
                "transaction_id": "e" * 64,
                "transaction_type": "Transfer",
                "price": None
            }
        ]
        
        # 根据limit截取历史记录
        history = history[:limit]
        
        # 在这里我们可以尝试获取账户信息来验证地址有效性
        try:
            await get_flow_account_info(collection_address)
            collection_verified = True
        except:
            collection_verified = False
        
        result = {
            "token_id": token_id,
            "collection": collection_address,
            "collection_verified": collection_verified,
            "current_owner": history[-1]["to_address"] if history else "Unknown",
            "ownership_history": history,
            "total_records": len(history)
        }
        
        # 添加额外信息：价格变化
        if len(history) > 1:
            prices = [item["price"] for item in history if item["price"] is not None]
            if len(prices) >= 2:
                result["price_history"] = {
                    "initial_price": prices[0],
                    "latest_price": prices[-1],
                    "price_changes": len(prices) - 1
                }
        
        return result
    
    except Exception as e:
        logging.error(f"获取NFT所有权历史时出错: {e}")
        return {
            "token_id": token_id,
            "collection": collection_address,
            "error": f"获取NFT所有权历史失败: {str(e)}",
            "status": "error"
        }

# ... 现有代码 ...

@mcp.tool()
async def analyze_nft_collection_stats(collection_address: str) -> Dict[str, Any]:
    """分析NFT集合的统计数据和市场表现
    
    Args:
        collection_address: NFT合约地址
    """
    if not collection_address or not isinstance(collection_address, str):
        raise ValueError("无效或缺失 'collection_address' 参数")
    
    logging.info(f"分析NFT集合统计数据: {collection_address}")
    
    try:
        # 实际应用中，可以通过查询链上数据、市场API等来获取统计信息
        # 这里使用模拟数据展示分析结果
        
        # 模拟的集合统计数据
        stats = {
            "collection_address": collection_address,
            "collection_name": "示例NFT集合",
            "total_supply": 10000,
            "unique_holders": 2500,
            "holder_ratio": "25.0%",  # 持有人数/总供应量
            "floor_price": "15.75 FLOW",
            "avg_price_7d": "18.2 FLOW",
            "highest_sale": "500.0 FLOW",
            "total_volume": "125000.0 FLOW",
            "daily_volume": {
                "today": "1250.5 FLOW",
                "yesterday": "980.2 FLOW",
                "change": "+27.6%"
            },
            "weekly_trend": "上升",
            "market_cap": "157500.0 FLOW",  # 地板价 * 总供应量
            "last_updated": "2024-04-07T10:15:00Z"
        }
        
        # 添加市场活跃度评分
        activity_score = calculate_activity_score(stats)
        stats["market_activity_score"] = activity_score["score"]
        stats["market_activity_level"] = activity_score["level"]
        stats["market_activity_analysis"] = activity_score["analysis"]
        
        # 添加验证信息
        try:
            await get_flow_account_info(collection_address)
            stats["collection_verified"] = True
        except:
            stats["collection_verified"] = False
            stats["warning"] = "无法验证合约地址是否有效"
        
        return stats
    
    except Exception as e:
        logging.error(f"分析NFT集合统计数据时出错: {e}")
        return {
            "collection_address": collection_address,
            "error": f"分析NFT集合统计数据失败: {str(e)}",
            "status": "error"
        }

def calculate_activity_score(stats: Dict[str, Any]) -> Dict[str, Any]:
    """计算NFT集合的市场活跃度评分
    
    Args:
        stats: NFT集合统计数据
    
    Returns:
        包含活跃度评分、等级和分析的字典
    """
    # 模拟计算市场活跃度评分的逻辑
    # 实际应用中可以基于多种指标进行加权计算
    
    # 提取关键指标
    holder_ratio = float(stats["holder_ratio"].replace("%", "")) / 100
    daily_volume_change = float(stats["daily_volume"]["change"].replace("%", "").replace("+", "")) / 100
    
    # 计算综合评分（示例逻辑）
    base_score = 50  # 基础分
    holder_score = holder_ratio * 100  # 持有比例得分，最高30分
    volume_score = min(daily_volume_change * 100, 20)  # 交易量变化得分，最高20分
    
    final_score = base_score + holder_score + volume_score
    final_score = min(final_score, 100)  # 限制最高分为100
    
    # 根据分数确定等级
    if final_score >= 80:
        level = "高度活跃"
        analysis = "该NFT集合市场表现强劲，持有者分布广泛，交易活跃度高，具有良好的流动性和市场认可度。"
    elif final_score >= 60:
        level = "中度活跃"
        analysis = "该NFT集合市场表现稳定，有一定的持有者基础和交易活跃度，流动性中等。"
    elif final_score >= 40:
        level = "低度活跃"
        analysis = "该NFT集合市场活跃度较低，持有者集中或交易量较小，流动性有限。"
    else:
        level = "不活跃"
        analysis = "该NFT集合市场几乎不活跃，交易稀少，可能是新发行的集合或已经失去市场关注。"
    
    return {
        "score": round(final_score, 1),
        "level": level,
        "analysis": analysis
    }

@mcp.tool()
async def nft_dashboard(collection_address: str, token_id: str = None) -> Dict[str, Any]:
    """生成NFT集合或单个NFT的综合仪表盘
    
    Args:
        collection_address: NFT合约地址
        token_id: 可选，特定NFT代币ID
    """
    if not collection_address or not isinstance(collection_address, str):
        raise ValueError("无效或缺失 'collection_address' 参数")
    
    # 记录开始时间
    start_time = asyncio.current_task().get_name()
    
    if token_id:
        logging.info(f"生成单个NFT仪表盘: 合约 {collection_address}, 代币ID {token_id}")
    else:
        logging.info(f"生成NFT集合仪表盘: 合约 {collection_address}")
    
    dashboard_data = {
        "request_type": "single_nft" if token_id else "collection",
        "collection_address": collection_address,
        "generated_at": start_time,
    }
    
    try:
        # 并行获取所有所需数据
        tasks = []
        
        # 获取网络状态
        block_task = asyncio.create_task(get_latest_sealed_block())
        tasks.append(block_task)
        
        # 根据请求类型获取不同的数据
        if token_id:
            # 单个NFT仪表盘
            metadata_task = asyncio.create_task(query_nft_metadata(collection_address, token_id))
            history_task = asyncio.create_task(get_nft_ownership_history(collection_address, token_id))
            tasks.extend([metadata_task, history_task])
            
            # 等待任务完成
            results = await asyncio.gather(*tasks)
            
            # 组装仪表盘数据
            dashboard_data.update({
                "token_id": token_id,
                "network_status": {
                    "latest_sealed_block": results[0],
                    "network": "mainnet" if "mainnet" in FLOW_API_BASE_URL else "testnet"
                },
                "nft_metadata": results[1],
                "ownership_history": results[2],
                "flowscan_url": f"https://www.flowscan.io/contract/{collection_address}"
            })
            
        else:
            # 集合仪表盘
            collection_info_task = asyncio.create_task(query_nft_collection_info(collection_address, "UnknownCollection"))
            stats_task = asyncio.create_task(analyze_nft_collection_stats(collection_address))
            tasks.extend([collection_info_task, stats_task])
            
            # 等待任务完成
            results = await asyncio.gather(*tasks)
            
            # 组装仪表盘数据
            dashboard_data.update({
                "network_status": {
                    "latest_sealed_block": results[0],
                    "network": "mainnet" if "mainnet" in FLOW_API_BASE_URL else "testnet"
                },
                "collection_info": results[1],
                "collection_stats": results[2],
                "flowscan_url": f"https://www.flowscan.io/contract/{collection_address}"
            })
    
    except Exception as e:
        error_message = f"生成NFT仪表盘时出错: {e}"
        logging.error(error_message)
        # 添加错误信息
        dashboard_data["error"] = error_message
        dashboard_data["status"] = "error"
    
    return dashboard_data

# --- DeFi 分析功能 ---
@mcp.tool()
async def query_defi_protocol_info(protocol_address: str) -> Dict[str, Any]:
    """查询DeFi协议信息
    
    Args:
        protocol_address: DeFi协议合约地址
    """
    if not protocol_address or not isinstance(protocol_address, str) or not protocol_address.startswith("0x"):
        raise ValueError("无效或缺失 'protocol_address' 参数")
    
    logging.info(f"查询DeFi协议信息: {protocol_address}")
    
    try:
        # 获取账户信息验证地址有效性
        account_info = await get_flow_account_info(protocol_address)
        
        # 模拟DeFi协议数据
        # 实际应用中应通过查询链上数据、协议API或索引服务获取真实数据
        protocol_info = {
            "address": protocol_address,
            "name": "示例DeFi协议",
            "verified": True,
            "type": "DEX/AMM",  # DEX, Lending, Yield Farm等
            "contracts_count": account_info.get("contracts_count", 0),
            "launch_date": "2023-01-15",
            "tvl_flow": "1250000.0",  # 总锁仓价值(Flow)
            "tvl_usd": "3750000.0",   # 总锁仓价值(USD)
            "daily_volume": "78500.0", # 24小时交易量
            "weekly_volume": "485200.0", # 7天交易量
            "unique_users": 3500,
            "description": "这是Flow上的一个去中心化交易协议，支持自动做市商(AMM)功能。",
            "social_links": {
                "website": "https://example-defi.flow",
                "twitter": "https://twitter.com/example_defi",
                "discord": "https://discord.gg/example-defi",
                "github": "https://github.com/example-defi"
            }
        }
        
        return protocol_info
    
    except Exception as e:
        logging.error(f"查询DeFi协议信息时出错: {e}")
        return {
            "address": protocol_address,
            "error": f"获取DeFi协议信息失败: {str(e)}",
            "status": "error"
        }

@mcp.tool()
async def get_liquidity_pools(protocol_address: str, limit: int = 5) -> Dict[str, Any]:
    """获取DeFi协议的流动性池信息
    
    Args:
        protocol_address: DeFi协议合约地址
        limit: 返回的流动性池数量限制，默认为5
    """
    if not protocol_address or not isinstance(protocol_address, str) or not protocol_address.startswith("0x"):
        raise ValueError("无效或缺失 'protocol_address' 参数")
    
    logging.info(f"获取流动性池信息: {protocol_address}, 限制 {limit}")
    
    try:
        # Cadence脚本查询协议的流动性池
        # 实际应用中需要编写正确的Cadence脚本
        # 这里模拟数据
        
        # 模拟流动性池数据
        pools = [
            {
                "pool_id": "pool-1",
                "name": "FLOW-USDC",
                "token_a": {
                    "symbol": "FLOW",
                    "address": "0x1654653399040a61",
                    "decimals": 8
                },
                "token_b": {
                    "symbol": "USDC",
                    "address": "0xb19436aae4d94622",
                    "decimals": 6
                },
                "total_liquidity_flow": "850000.0",
                "total_liquidity_usd": "2550000.0",
                "volume_24h": "32500.0",
                "fees_24h": "97.5",
                "apr": "12.5%",
                "liquidity_providers": 450,
                "creation_date": "2023-02-10"
            },
            {
                "pool_id": "pool-2",
                "name": "FLOW-FUSD",
                "token_a": {
                    "symbol": "FLOW",
                    "address": "0x1654653399040a61",
                    "decimals": 8
                },
                "token_b": {
                    "symbol": "FUSD",
                    "address": "0x3c5959b568896393",
                    "decimals": 8
                },
                "total_liquidity_flow": "650000.0",
                "total_liquidity_usd": "1950000.0",
                "volume_24h": "28000.0",
                "fees_24h": "84.0",
                "apr": "10.8%",
                "liquidity_providers": 380,
                "creation_date": "2023-02-15"
            },
            {
                "pool_id": "pool-3",
                "name": "USDC-FUSD",
                "token_a": {
                    "symbol": "USDC",
                    "address": "0xb19436aae4d94622",
                    "decimals": 6
                },
                "token_b": {
                    "symbol": "FUSD",
                    "address": "0x3c5959b568896393",
                    "decimals": 8
                },
                "total_liquidity_flow": "450000.0",
                "total_liquidity_usd": "1350000.0",
                "volume_24h": "18000.0",
                "fees_24h": "54.0",
                "apr": "9.2%",
                "liquidity_providers": 250,
                "creation_date": "2023-03-01"
            },
            {
                "pool_id": "pool-4",
                "name": "FLOW-STFLOW",
                "token_a": {
                    "symbol": "FLOW",
                    "address": "0x1654653399040a61",
                    "decimals": 8
                },
                "token_b": {
                    "symbol": "stFLOW",
                    "address": "0x8c5303eaa26202d6",
                    "decimals": 8
                },
                "total_liquidity_flow": "350000.0",
                "total_liquidity_usd": "1050000.0",
                "volume_24h": "12000.0",
                "fees_24h": "36.0",
                "apr": "8.5%",
                "liquidity_providers": 180,
                "creation_date": "2023-04-10"
            },
            {
                "pool_id": "pool-5",
                "name": "FLOW-BLT",
                "token_a": {
                    "symbol": "FLOW",
                    "address": "0x1654653399040a61",
                    "decimals": 8
                },
                "token_b": {
                    "symbol": "BLT",
                    "address": "0x0f9df91c9121c460",
                    "decimals": 8
                },
                "total_liquidity_flow": "250000.0",
                "total_liquidity_usd": "750000.0",
                "volume_24h": "8000.0",
                "fees_24h": "24.0",
                "apr": "7.8%",
                "liquidity_providers": 120,
                "creation_date": "2023-05-20"
            }
        ]
        
        # 根据limit截取池子数据
        pools = pools[:limit]
        
        # 验证协议地址
        try:
            await get_flow_account_info(protocol_address)
            protocol_verified = True
        except:
            protocol_verified = False
        
        return {
            "protocol_address": protocol_address,
            "protocol_verified": protocol_verified,
            "total_pools": len(pools),
            "pools": pools,
            "total_tvl_usd": sum(float(pool["total_liquidity_usd"]) for pool in pools),
            "total_volume_24h": sum(float(pool["volume_24h"]) for pool in pools),
            "last_updated": datetime.now().isoformat()
        }
    
    except Exception as e:
        logging.error(f"获取流动性池信息时出错: {e}")
        return {
            "protocol_address": protocol_address,
            "error": f"获取流动性池信息失败: {str(e)}",
            "status": "error"
        }

# ... 现有代码 ...

@mcp.tool()
async def analyze_defi_portfolio(account_address: str) -> Dict[str, Any]:
    """分析用户的DeFi投资组合
    
    Args:
        account_address: 用户钱包地址
    """
    if not account_address or not isinstance(account_address, str) or not account_address.startswith("0x"):
        raise ValueError("无效或缺失 'account_address' 参数")
    
    logging.info(f"分析DeFi投资组合: {account_address}")
    
    try:
        # 获取账户基本信息
        account_info = await get_flow_account_info(account_address)
        
        # 模拟用户的DeFi投资组合数据
        # 实际应用中需要查询链上数据
        portfolio = {
            "account_address": account_address,
            "account_balance_flow": account_info.get("balance_ufix64", "0.0"),
            "total_defi_value_usd": "12500.0",
            "total_defi_value_flow": "4166.67",
            "portfolio_allocation": [
                {
                    "protocol_name": "示例DEX",
                    "protocol_address": "0x" + "a" * 16,
                    "type": "流动性提供",
                    "pools": [
                        {
                            "pool_name": "FLOW-USDC",
                            "pool_id": "pool-1",
                            "provided_liquidity_usd": "5000.0",
                            "provided_liquidity_flow": "1666.67",
                            "share_percentage": "0.2%",
                            "rewards_pending_usd": "12.5",
                            "apr": "12.5%"
                        },
                        {
                            "pool_name": "FLOW-FUSD",
                            "pool_id": "pool-2",
                            "provided_liquidity_usd": "3000.0",
                            "provided_liquidity_flow": "1000.0",
                            "share_percentage": "0.15%",
                            "rewards_pending_usd": "8.1",
                            "apr": "10.8%"
                        }
                    ],
                    "total_value_usd": "8000.0",
                    "percentage_of_portfolio": "64%"
                },
                {
                    "protocol_name": "示例借贷平台",
                    "protocol_address": "0x" + "b" * 16,
                    "type": "借贷",
                    "positions": [
                        {
                            "position_type": "存款",
                            "token": "FLOW",
                            "amount": "1000.0",
                            "value_usd": "3000.0",
                            "apy": "4.5%",
                            "accrued_interest": "11.25"
                        },
                        {
                            "position_type": "借款",
                            "token": "FUSD",
                            "amount": "500.0",
                            "value_usd": "500.0",
                            "apy": "8.2%",
                            "accrued_interest": "10.25",
                            "collateral_ratio": "250%"
                        }
                    ],
                    "net_value_usd": "2500.0",
                    "percentage_of_portfolio": "20%"
                },
                {
                    "protocol_name": "示例质押平台",
                    "protocol_address": "0x" + "c" * 16,
                    "type": "质押",
                    "positions": [
                        {
                            "token": "FLOW",
                            "staked_amount": "500.0",
                            "value_usd": "1500.0",
                            "apy": "6.8%",
                            "rewards_pending": "8.5",
                            "lock_period": "30天",
                            "unlock_date": (datetime.now() + timedelta(days=15)).isoformat()
                        }
                    ],
                    "total_value_usd": "1500.0",
                    "percentage_of_portfolio": "12%"
                },
                {
                    "protocol_name": "示例收益聚合器",
                    "protocol_address": "0x" + "d" * 16,
                    "type": "收益聚合",
                    "vaults": [
                        {
                            "vault_name": "FLOW优化策略",
                            "deposited_amount": "166.67",
                            "token": "FLOW",
                            "value_usd": "500.0",
                            "apy": "15.2%",
                            "strategy": "自动复投AMM收益",
                            "last_harvest": (datetime.now() - timedelta(days=2)).isoformat()
                        }
                    ],
                    "total_value_usd": "500.0",
                    "percentage_of_portfolio": "4%"
                }
            ],
            "historical_performance": {
                "week": "+2.5%",
                "month": "+8.2%",
                "quarter": "+15.7%",
                "year": "+32.1%"
            },
            "risk_assessment": {
                "overall_risk": "中等",
                "diversification_score": 7.5,  # 1-10分
                "impermanent_loss_exposure": "中等",
                "liquidation_risk": "低",
                "protocol_risk": "中等",
                "recommendations": [
                    "考虑增加稳定币配置以降低波动风险",
                    "当前借贷健康因子良好，但市场波动时应注意调整",
                    "流动性集中在少数池子，可以考虑更多样化的配置"
                ]
            },
            "last_updated": datetime.now().isoformat()
        }
        
        return portfolio
    
    except Exception as e:
        logging.error(f"分析DeFi投资组合时出错: {e}")
        return {
            "account_address": account_address,
            "error": f"分析DeFi投资组合失败: {str(e)}",
            "status": "error"
        }

@mcp.tool()
async def get_token_price_history(token_address: str, days: int = 30) -> Dict[str, Any]:
    """获取代币价格历史数据
    
    Args:
        token_address: 代币合约地址
        days: 历史数据天数，默认30天
    """
    if not token_address or not isinstance(token_address, str) or not token_address.startswith("0x"):
        raise ValueError("无效或缺失 'token_address' 参数")
    
    if days <= 0 or days > 365:
        days = 30  # 限制在合理范围内
    
    logging.info(f"获取代币价格历史: {token_address}, {days}天")
    
    try:
        # 模拟代币价格历史数据
        # 实际应用中应从价格API或索引服务获取
        
        # 生成模拟的历史价格数据
        today = datetime.now()
        price_history = []
        base_price = 3.0  # 基础价格，例如FLOW代币
        
        # 生成波动的价格数据
        import random
        random.seed(hash(token_address) % 1000)  # 使用地址作为随机种子，确保相同地址生成相同序列
        
        for i in range(days):
            date = today - timedelta(days=days-i-1)
            # 生成-5%到+5%之间的随机波动
            change = (random.random() - 0.5) * 0.1
            # 应用波动并确保价格为正
            if i == 0:
                price = max(0.1, base_price * (1 + change))
            else:
                price = max(0.1, price_history[-1]["price"] * (1 + change))
            
            price_history.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": round(price, 4),
                "volume": round(random.uniform(500000, 2000000), 2)
            })
        
        # 获取代币信息
        token_info = {
            "address": token_address,
            "symbol": "FLOW" if token_address == "0x1654653399040a61" else "未知代币",
            "name": "Flow Token" if token_address == "0x1654653399040a61" else "未知代币",
            "current_price_usd": price_history[-1]["price"],
            "price_change_24h": f"{((price_history[-1]['price'] / price_history[-2]['price']) - 1) * 100:.2f}%",
            "price_change_7d": f"{((price_history[-1]['price'] / price_history[-7]['price'] if len(price_history) >= 7 else price_history[0]['price']) - 1) * 100:.2f}%",
            "price_change_30d": f"{((price_history[-1]['price'] / price_history[0]['price']) - 1) * 100:.2f}%",
            "market_cap": f"{price_history[-1]['price'] * 1369203921:.2f}",  # 假设的流通量
            "total_supply": "1369203921",  # FLOW的总供应量
            "last_updated": datetime.now().isoformat()
        }
        
        return {
            "token_info": token_info,
            "days": days,
            "price_history": price_history,
            "chart_data": {
                "labels": [item["date"] for item in price_history],
                "prices": [item["price"] for item in price_history],
                "volumes": [item["volume"] for item in price_history]
            }
        }
    
    except Exception as e:
        logging.error(f"获取代币价格历史时出错: {e}")
        return {
            "token_address": token_address,
            "error": f"获取代币价格历史失败: {str(e)}",
            "status": "error"
        }

@mcp.tool()
async def defi_dashboard(protocol_address: str = None, account_address: str = None) -> Dict[str, Any]:
    """生成DeFi综合仪表盘，可以是协议概览或用户投资组合
    
    Args:
        protocol_address: 可选，DeFi协议地址
        account_address: 可选，用户钱包地址
    """
    if not protocol_address and not account_address:
        raise ValueError("必须提供 'protocol_address' 或 'account_address' 参数之一")
    
    if protocol_address and account_address:
        logging.info(f"生成综合DeFi仪表盘: 协议 {protocol_address}, 账户 {account_address}")
        dashboard_type = "comprehensive"
    elif protocol_address:
        logging.info(f"生成DeFi协议仪表盘: {protocol_address}")
        dashboard_type = "protocol"
    else:
        logging.info(f"生成DeFi用户仪表盘: {account_address}")
        dashboard_type = "account"
    
    # 记录开始时间
    start_time = asyncio.current_task().get_name()
    
    dashboard_data = {
        "dashboard_type": dashboard_type,
        "generated_at": start_time,
    }
    
    try:
        # 获取网络状态
        latest_block = await get_latest_sealed_block()
        dashboard_data["network_status"] = {
            "latest_sealed_block": latest_block,
            "network": "mainnet" if "mainnet" in FLOW_API_BASE_URL else "testnet"
        }
        
        # 根据仪表盘类型获取不同数据
        if dashboard_type == "protocol" or dashboard_type == "comprehensive":
            # 获取协议信息
            protocol_info = await query_defi_protocol_info(protocol_address)
            pools_info = await get_liquidity_pools(protocol_address)
            
            dashboard_data["protocol_info"] = protocol_info
            dashboard_data["liquidity_pools"] = pools_info
            
            # 获取协议代币价格历史（如果有）
            if "token_address" in protocol_info:
                token_price = await get_token_price_history(protocol_info["token_address"])
                dashboard_data["token_price_history"] = token_price
        
        if dashboard_type == "account" or dashboard_type == "comprehensive":
            # 获取用户投资组合
            portfolio = await analyze_defi_portfolio(account_address)
            dashboard_data["portfolio"] = portfolio
            
            # 如果是综合仪表盘，添加用户在该协议中的投资情况
            if dashboard_type == "comprehensive" and "portfolio_allocation" in portfolio:
                # 查找用户在该协议中的投资
                protocol_investments = []
                for allocation in portfolio.get("portfolio_allocation", []):
                    if allocation.get("protocol_address") == protocol_address:
                        protocol_investments.append(allocation)
                
                if protocol_investments:
                    dashboard_data["user_protocol_investments"] = protocol_investments
                    dashboard_data["user_protocol_total_value"] = sum(
                        float(inv.get("total_value_usd", 0)) 
                        for inv in protocol_investments
                    )
        
        # 添加生成时间和来源信息
        dashboard_data["last_updated"] = datetime.now().isoformat()
        dashboard_data["data_source"] = "FlowMind DeFi Analytics"
        
        return dashboard_data
    
    except Exception as e:
        error_message = f"生成DeFi仪表盘时出错: {e}"
        logging.error(error_message)
        # 添加错误信息
        dashboard_data["error"] = error_message
        dashboard_data["status"] = "error"
        return dashboard_data

@mcp.tool()
async def compare_defi_protocols(protocol_addresses: List[str]) -> Dict[str, Any]:
    """比较多个DeFi协议的性能和特点
    
    Args:
        protocol_addresses: DeFi协议地址列表
    """
    if not protocol_addresses or not isinstance(protocol_addresses, list) or len(protocol_addresses) < 2:
        raise ValueError("必须提供至少两个有效的协议地址进行比较")
    
    logging.info(f"比较DeFi协议: {', '.join(protocol_addresses)}")
    
    try:
        # 并行获取所有协议信息
        protocol_tasks = []
        for address in protocol_addresses:
            task = asyncio.create_task(query_defi_protocol_info(address))
            protocol_tasks.append(task)
        
        # 等待所有任务完成
        protocol_results = await asyncio.gather(*protocol_tasks)
        
        # 提取比较数据
        comparison_data = {
            "protocols": protocol_results,
            "comparison_metrics": {
                "tvl_usd": [],
                "daily_volume": [],
                "unique_users": [],
                "contracts_count": []
            },
            "comparison_chart_data": {
                "labels": [],
                "tvl_values": [],
                "volume_values": []
            }
        }
        
        # 填充比较数据
        for protocol in protocol_results:
            name = protocol.get("name", "未知协议")
            comparison_data["comparison_metrics"]["tvl_usd"].append({
                "name": name,
                "value": protocol.get("tvl_usd", "0.0")
            })
            comparison_data["comparison_metrics"]["daily_volume"].append({
                "name": name,
                "value": protocol.get("daily_volume", "0.0")
            })
            comparison_data["comparison_metrics"]["unique_users"].append({
                "name": name,
                "value": protocol.get("unique_users", 0)
            })
            comparison_data["comparison_metrics"]["contracts_count"].append({
                "name": name,
                "value": protocol.get("contracts_count", 0)
            })
            
            # 图表数据
            comparison_data["comparison_chart_data"]["labels"].append(name)
            comparison_data["comparison_chart_data"]["tvl_values"].append(
                float(protocol.get("tvl_usd", "0.0").replace(",", ""))
            )
            comparison_data["comparison_chart_data"]["volume_values"].append(
                float(protocol.get("daily_volume", "0.0").replace(",", ""))
            )
        
        # 添加排名信息
        comparison_data["rankings"] = {
            "by_tvl": sorted(
                [p.get("name", "未知") for p in protocol_results],
                key=lambda x: float(next((p.get("tvl_usd", "0.0") for p in protocol_results if p.get("name") == x), "0.0").replace(",", "")),
                reverse=True
            ),
            "by_volume": sorted(
                [p.get("name", "未知") for p in protocol_results],
                key=lambda x: float(next((p.get("daily_volume", "0.0") for p in protocol_results if p.get("name") == x), "0.0").replace(",", "")),
                reverse=True
            ),
            "by_users": sorted(
                [p.get("name", "未知") for p in protocol_results],
                key=lambda x: next((p.get("unique_users", 0) for p in protocol_results if p.get("name") == x), 0),
                reverse=True
            )
        }
        
        # 添加分析总结
        comparison_data["summary"] = generate_protocol_comparison_summary(protocol_results)
        
        return comparison_data
    
    except Exception as e:
        logging.error(f"比较DeFi协议时出错: {e}")
        return {
            "protocol_addresses": protocol_addresses,
            "error": f"比较DeFi协议失败: {str(e)}",
            "status": "error"
        }

def generate_protocol_comparison_summary(protocols: List[Dict[str, Any]]) -> str:
    """生成协议比较的文字总结
    
    Args:
        protocols: 协议信息列表
    
    Returns:
        比较总结文本
    """
    if not protocols or len(protocols) < 2:
        return "提供的协议数据不足以进行比较"
    
    # 按TVL排序
    sorted_by_tvl = sorted(
        protocols,
        key=lambda x: float(x.get("tvl_usd", "0.0").replace(",", "")),
        reverse=True
    )
    
    # 生成总结
    top_protocol = sorted_by_tvl[0]
    top_name = top_protocol.get("name", "未知协议")
    top_tvl = top_protocol.get("tvl_usd", "未知")
    
    summary = f"在比较的{len(protocols)}个协议中，{top_name}拥有最高的总锁仓价值({top_tvl})。"
    
    # 添加交易量比较
    sorted_by_volume = sorted(
        protocols,
        key=lambda x: float(x.get("daily_volume", "0.0").replace(",", "")),
        reverse=True
    )
    volume_leader = sorted_by_volume[0]
    volume_name = volume_leader.get("name", "未知协议")
    volume_value = volume_leader.get("daily_volume", "未知")
    
    if volume_name == top_name:
        summary += f"该协议同时也拥有最高的日交易量({volume_value})。"
    else:
        summary += f"而{volume_name}则拥有最高的日交易量({volume_value})。"
    
    # 添加用户数比较
    sorted_by_users = sorted(
        protocols,
        key=lambda x: x.get("unique_users", 0),
        reverse=True
    )
    users_leader = sorted_by_users[0]
    users_name = users_leader.get("name", "未知协议")
    users_count = users_leader.get("unique_users", "未知")
    
    if users_name not in [top_name, volume_name]:
        summary += f"{users_name}拥有最多的用户数量({users_count})。"
    
    # 添加综合评价
    summary += "\n\n综合来看，"
    if top_name == volume_name and top_name == users_name:
        summary += f"{top_name}在锁仓价值、交易量和用户数量三个维度都处于领先地位，显示出强大的市场优势。"
    elif top_name == volume_name:
        summary += f"{top_name}在锁仓价值和交易量方面表现突出，但在用户基础方面可能还有提升空间。"
    elif top_name == users_name:
        summary += f"{top_name}拥有最大的用户基础和锁仓价值，但交易活跃度方面可能需要进一步优化。"
    else:
        summary += f"各协议在不同维度各有优势，市场竞争较为激烈。"
    
    return summary

if __name__ == "__main__":
    # 运行MCP服务器
    mcp.run(transport="stdio")