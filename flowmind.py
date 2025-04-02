from mcp.server.fastmcp import FastMCP
import asyncio
import json
import logging
import os
from typing import Any, Dict
import httpx

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
async def say_hello(name: str = "World") -> str:
    """返回一个友好的问候语，这是一个简单的 hello 示例工具

    Args:
        name: 要问候的名字，默认是 "World"
    """
    return f"Hello, {name}! 这是我的第一个 Flow Mind MCP 工具。"


@mcp.tool()
async def get_current_time() -> str:
    """获取当前时间"""
    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"当前时间是: {current_time}"

@mcp.tool()
async def get_md5_hash(input_str: str) -> str:
    """返回输入字符串的MD5哈希值
    
    Args:
        input_str: 需要加密的字符串
    """
    import hashlib
    md5_hash = hashlib.md5(input_str.encode()).hexdigest()
    return f"'{input_str}' 的MD5哈希值是: {md5_hash}"

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
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP错误: {e.response.status_code} - {e.response.text}")
        # 尝试使用备用方法
        return await get_account_transactions_fallback(account_address, limit)
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
async def get_account_summary(account_address: str) -> Dict[str, Any]:
    """获取Flow账户摘要信息，包括账户基本信息和交易历史
    
    Args:
        account_address: Flow账户地址，以0x开头
    """
    if not account_address or not isinstance(account_address, str) or not account_address.startswith("0x"):
        raise ValueError("无效或缺失 'account_address' 参数")

    logging.info(f"开始获取账户 {account_address} 的摘要信息")
    summary_data = {}
    errors = []

    try:
        # 并行获取账户信息和交易历史
        account_info_task = asyncio.create_task(get_flow_account_info(account_address))
        transactions_task = asyncio.create_task(get_account_transactions(account_address, 5))
        
        # 等待所有任务完成
        account_info = await account_info_task
        transactions_result = await transactions_task
        
        summary_data["account_info"] = account_info
        
        # 添加交易历史信息
        if "error" in transactions_result:
            tx_info_message = (
                f"获取交易历史时出错: {transactions_result['error']}。"
                "备注: 要获取完整的交易历史，请确保设置了 QUICKNODE_API_URL 环境变量。"
            )
            summary_data["transactions_info"] = tx_info_message
        else:
            summary_data["transactions"] = transactions_result

    except Exception as e:
        error_message = f"获取账户摘要部分信息时出错: {e}"
        logging.warning(error_message)
        errors.append(error_message)
        # 即使出错，也尝试返回已获取的部分信息（如果有）和错误
        summary_data["warnings"] = errors # 将错误信息放入warnings字段

    if not summary_data: # 如果完全没获取到任何信息
         raise RuntimeError(f"无法获取账户摘要信息: {'; '.join(errors)}")

    return summary_data


if __name__ == "__main__":
    # 测试用例：获取账户交易历史
    async def test_get_account_transactions():
        test_address = "0x4332ef965cb743d2"  # 测试用的Flow账户地址
        print(f"开始测试获取账户 {test_address} 的交易历史...")
        
        try:
            # 调用函数获取交易历史
            result = await get_account_transactions(test_address, 5)
            print(result)
            # 格式化输出结果
            print("\n===== 账户交易历史测试结果 =====")
            if "error" in result:
                print(f"错误: {result['error']}")
            else:
                # 输出账户信息
                if "account_info" in result:
                    print("\n账户信息:")
                    for key, value in result["account_info"].items():
                        print(f"  {key}: {value}")
                
                # 输出交易总数
                print(f"\n交易总数: {result.get('total_transactions', 0)}")
                
                # 输出交易详情
                if "transactions" in result and result["transactions"]:
                    print("\n交易详情:")
                    for i, tx in enumerate(result["transactions"], 1):
                        print(f"\n交易 #{i}:")
                        for key, value in tx.items():
                            print(f"  {key}: {value}")
                else:
                    print("\n未找到交易记录")
                
                print(f"\n数据来源: {result.get('source', '未知')}")
            
            print("===== 测试完成 =====\n")
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
    
    # 运行测试用例
    import asyncio
    asyncio.run(test_get_account_transactions())
    
    # 初始化并运行服务器 (测试完成后可以取消注释此行)
    # mcp.run(transport="stdio")
