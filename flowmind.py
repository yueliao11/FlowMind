from mcp.server.fastmcp import FastMCP
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union
import httpx
import time
from datetime import datetime, timedelta
import base64

# 初始化 FastMCP 服务器
mcp = FastMCP("flowmind")

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Flow Native API
FLOW_API_BASE_URL = os.environ.get("FLOW_API_BASE_URL", "https://rest-mainnet.onflow.org/v1")

# Flowscan Simple API (via QuickNode - Requires specific endpoint URL and possibly Key)
# Replace with your actual QuickNode Flow Simple API endpoint
QUICKNODE_FLOW_API_URL = os.environ.get("QUICKNODE_FLOW_API_URL")
QUICKNODE_API_KEY =os.environ.get("QUICKNODE_API_KEY") # May be needed in headers

# Price Feed API (Example: CoinGecko Public API)
PRICE_API_BASE_URL = "https://api.coingecko.com/api/v3"

REQUEST_TIMEOUT = 20 # Increased timeout for potentially slower indexer APIs
# 使用异步 httpx 客户端
async_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True)

# --- Token Address to Price API ID Mapping (Example) ---
# Add more mappings as needed
TOKEN_ADDRESS_TO_PRICE_ID = {
    "0x1654653399040a61": "flow",        # FLOW Token Mainnet
    "0x7e60df042a9c0868": "flow",        # FLOW Token Testnet
    "0xa983fecbed621163": "usd-coin",    # USDC Mainnet (Axelar) - Example, verify correct ID
    "0x3c5959b568896393": "fusd",        # FUSD Mainnet - Example, verify correct ID
    # Add other common Flow tokens (stFlow, etc.)
}

# --- Error Handling ---
class FlowAPIError(Exception):
    pass

class QuickNodeAPIError(Exception):
    pass

class CadenceExecutionError(Exception):
    pass

class PriceAPIError(Exception):
    pass

# --- Helper Functions ---

async def _make_async_api_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_payload: Optional[Dict[str, Any]] = None,
    api_name: str = "API"
) -> Any:
    """Generic async HTTP request helper."""
    logging.info(f"向 {api_name} 发送 {method} 请求: {url} Params: {params} Payload: {json_payload is not None}")
    try:
        response = await async_client.request(method, url, headers=headers, params=params, json=json_payload)
        response.raise_for_status()
        # Handle potential empty responses or non-JSON responses
        if response.status_code == 204 or not response.content:
             logging.info(f"{api_name} 请求成功 (无内容): {url}")
             return None # Or return {} depending on expected behaviour
        data = response.json()
        logging.info(f"{api_name} 请求成功: {url}")
        return data
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        error_text = e.response.text
        logging.error(f"请求 {url} 时发生HTTP错误 ({api_name}): {status_code} - {error_text}", exc_info=False)
        raise FlowAPIError(f"{api_name} 请求失败 (HTTP {status_code}): {error_text}") from e
    except httpx.RequestError as e:
        logging.error(f"请求 {url} 时发生网络错误 ({api_name}): {e}", exc_info=False)
        raise ConnectionError(f"网络错误，无法连接到 {api_name}: {e}") from e
    except json.JSONDecodeError as e:
        logging.error(f"解析来自 {url} 的响应JSON失败 ({api_name}): {e.response.text}", exc_info=False)
        raise ValueError(f"无法解析 {api_name} 的响应: {e}") from e
    except Exception as e:
        logging.error(f"处理请求 {url} 时发生未知错误 ({api_name}): {e}", exc_info=True)
        raise RuntimeError(f"处理 {api_name} 请求时发生未知内部错误: {e}") from e

async def make_async_flow_api_request(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Sends GET request to Flow HTTP API."""
    api_url = f"{FLOW_API_BASE_URL}{endpoint}"
    # Ensure params keys are valid for Flow API (e.g., height, select, expand)
    valid_params = {k: v for k, v in params.items() if k in ['height', 'select', 'expand', 'start_block', 'end_block', 'event_type']} if params else None
    return await _make_async_api_request(api_url, params=valid_params, api_name="Flow HTTP API")

async def make_async_quicknode_request(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Sends GET request to QuickNode Flow Simple API."""
    if not QUICKNODE_FLOW_API_URL:
        raise ValueError("QUICKNODE_FLOW_API_URL 未配置")
    api_url = f"{QUICKNODE_FLOW_API_URL}{endpoint}" # Ensure endpoint starts with /
    headers = {"Accept": "application/json"}
    if QUICKNODE_API_KEY:
        headers["Authorization"] = f"Bearer {QUICKNODE_API_KEY}" # Or x-api-key, check QuickNode docs
        # Potentially add other headers like x-qn-api-version if needed
    return await _make_async_api_request(api_url, headers=headers, params=params, api_name="QuickNode Flow API")

async def make_async_price_api_request(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Sends GET request to Price Feed API (CoinGecko example)."""
    api_url = f"{PRICE_API_BASE_URL}{endpoint}"
    headers = {"Accept": "application/json"}
    return await _make_async_api_request(api_url, headers=headers, params=params, api_name="Price Feed API")

# --- Cadence Execution Helper ---

def _encode_cadence_value(value: Any, type_hint: Optional[str] = None) -> Dict[str, Any]:
    """Encodes a Python value into a Cadence JSON argument representation."""
    cadence_type = None
    encoded_value = value

    if isinstance(value, str):
        if value.startswith("0x") and len(value) == 18:
            cadence_type = "Address"
        elif type_hint == "String" or cadence_type is None: # Default to String if ambiguous
             cadence_type = "String"
        # Could add more string type detections (Path, etc.)
    elif isinstance(value, int):
        cadence_type = "Int64" if value < 0 else "UInt64" # Basic guess, might need UInt*, Int* based on context
        encoded_value = str(value) # Cadence numbers are strings in JSON
    elif isinstance(value, float):
         cadence_type = "UFix64" # Common case, might need Fix64
         encoded_value = f"{value:.8f}" # Format UFix64 with 8 decimal places
    elif isinstance(value, bool):
         cadence_type = "Bool"
    # Add more types: Array, Dictionary, Optional, Path, Type, etc. as needed
    else:
        raise TypeError(f"Unsupported type for Cadence encoding: {type(value)}")

    if type_hint and cadence_type != type_hint:
        logging.warning(f"Type hint '{type_hint}' differs from inferred type '{cadence_type}' for value '{value}'. Using inferred type.")

    return {"type": cadence_type, "value": encoded_value}


def _decode_cadence_value(cadence_result: Dict[str, Any]) -> Any:
    """Decodes a Cadence JSON result value into a Python object."""
    if not isinstance(cadence_result, dict) or "type" not in cadence_result or "value" not in cadence_result:
         # Handle cases where the result might not be the standard type/value dict
         # For example, sometimes the script execution endpoint itself returns decoded JSON directly.
         if isinstance(cadence_result, dict) and "value" in cadence_result and isinstance(cadence_result["value"], (dict, list)):
              # It might be already decoded JSON wrapped in a 'value' field
              return cadence_result["value"]
         # Otherwise, return the raw structure or handle as error
         logging.warning(f"Unexpected Cadence result format: {cadence_result}")
         return cadence_result # Return raw dict/value

    cadence_type = cadence_result["type"]
    value = cadence_result["value"]

    try:
        if cadence_type == "String":
            return value
        elif cadence_type == "Address":
            return value
        elif cadence_type in ["Int", "Int8", "Int16", "Int32", "Int64", "Int128", "Int256",
                            "UInt", "UInt8", "UInt16", "UInt32", "UInt64", "UInt128", "UInt256"]:
            return int(value)
        elif cadence_type in ["Fix64", "UFix64"]:
            return float(value)
        elif cadence_type == "Bool":
            return value
        elif cadence_type == "Optional":
            # Optional has a nested value or is null
            return _decode_cadence_value(value) if value is not None else None
        elif cadence_type == "Dictionary":
            # Dictionary items are key/value pairs in a list
            return {_decode_cadence_value(item["key"]): _decode_cadence_value(item["value"]) for item in value}
        elif cadence_type == "Array":
             # Array items are in the 'value' list
            return [_decode_cadence_value(item) for item in value]
        elif cadence_type == "Struct" or cadence_type == "Resource" or cadence_type == "Event" or cadence_type == "Contract":
             # These have an 'id' and 'fields' (list of name/value pairs)
             struct_data = {"_type": cadence_type, "_id": cadence_result.get("id")}
             struct_data.update({field["name"]: _decode_cadence_value(field["value"]) for field in value})
             return struct_data
        # Add more type decodings (Path, Capability, Type, Enum etc.)
        else:
            logging.warning(f"Unsupported Cadence type for decoding: {cadence_type}. Returning raw value.")
            return value
    except Exception as e:
        logging.error(f"Error decoding Cadence value (Type: {cadence_type}, Value: {value}): {e}")
        return value # Return raw value on error


async def execute_cadence_script(script: str, args: List[Any] = None) -> Any:
    """Executes a read-only Cadence script using the Flow HTTP API."""
    if args is None:
        args = []

    endpoint = "/scripts"
    url = f"{FLOW_API_BASE_URL}{endpoint}"

    # Encode script to Base64
    encoded_script = base64.b64encode(script.encode("utf-8")).decode("utf-8")

    # Encode arguments
    encoded_args = []
    try:
        for arg in args:
            # Argument values also need to be base64 encoded
            cadence_arg_json = _encode_cadence_value(arg)
            # JSON stringify the inner value dict, then base64 encode THAT string.
            # Note: The API docs are a bit ambiguous here. Let's try base64 encoding the whole arg JSON string.
            # If that fails, we might need to only encode the 'value' part, or experiment.
            arg_json_string = json.dumps(cadence_arg_json)
            encoded_args.append(base64.b64encode(arg_json_string.encode('utf-8')).decode('utf-8'))
            # --- Alternative (if above fails): Encode only the value ---
            # value_str = json.dumps(cadence_arg_json['value'])
            # encoded_value_b64 = base64.b64encode(value_str.encode('utf-8')).decode('utf-8')
            # encoded_args.append(base64.b64encode(json.dumps({"type": cadence_arg_json['type'], "value": encoded_value_b64}).encode('utf-8')).decode('utf-8'))

    except TypeError as e:
        raise ValueError(f"Error encoding Cadence arguments: {e}") from e

    payload = {
        "script": encoded_script,
        "arguments": encoded_args
    }

    logging.info(f"执行 Cadence 脚本 (前50字节): {script[:50]}...")

    try:
        # Use POST for script execution
        response_bytes = await _make_async_api_request(url, method="POST", json_payload=payload, api_name="Flow Script Execution")

        if not response_bytes:
             raise CadenceExecutionError("Cadence script execution returned empty response.")

        # Decode the base64 encoded JSON response value
        decoded_response_str = base64.b64decode(response_bytes).decode('utf-8')
        result_json = json.loads(decoded_response_str)

        # Decode the Cadence JSON into Python types
        decoded_result = _decode_cadence_value(result_json)
        logging.info(f"Cadence脚本执行成功")
        return decoded_result

    except (FlowAPIError, json.JSONDecodeError, base64.binascii.Error, ValueError) as e:
        logging.error(f"执行 Cadence 脚本时出错: {e}", exc_info=True)
        # Attempt to include decoded error if possible from FlowAPIError
        error_detail = str(e)
        if isinstance(e, FlowAPIError):
             # Try decoding potential Cadence error message within the HTTP error
             try:
                  # Format might be: "Flow API 请求失败 (HTTP 400): execution error code 1101: ..."
                  # Or the error text might contain JSON
                  raw_error = str(e).split("):", 1)[-1].strip()
                  error_json = json.loads(raw_error)
                  error_detail = f"Cadence Error: {error_json.get('message', raw_error)} (Code: {error_json.get('code')})"
             except Exception:
                  pass # Stick with original error string
        raise CadenceExecutionError(f"执行 Cadence 脚本失败: {error_detail}") from e
    except Exception as e:
        logging.error(f"执行 Cadence 脚本时发生未知错误: {e}", exc_info=True)
        raise CadenceExecutionError(f"执行 Cadence 脚本时发生未知内部错误: {e}") from e


# --- Core Tools ---

@mcp.tool()
async def get_flow_account_info(account_address: str) -> Dict[str, Any]:
    """查询Flow账户信息，包含余额 (使用 Flow HTTP API)

    Args:
        account_address: Flow账户地址，以0x开头
    """
    if not account_address or not isinstance(account_address, str) or not account_address.startswith("0x"):
        raise ValueError("无效或缺失 'account_address' 参数")
    endpoint = f"/accounts/{account_address}"
    try:
        result = await make_async_flow_api_request(endpoint, params={"expand": "contracts,keys"})
        # Format balance nicely
        balance_flow = 0.0
        try:
            balance_flow = float(result.get("balance", "0")) / 10**8 # FLOW has 8 decimals
        except ValueError:
             logging.warning(f"Could not parse balance: {result.get('balance')}")

        simplified_result = {
            "address": result.get("address"),
            "balance_flow": f"{balance_flow:.8f}", # Formatted FLOW balance
            "balance_ufix64": result.get("balance"), # Raw balance
            "keys_count": len(result.get("keys", [])),
            "contracts_count": len(result.get("contracts", {})),
            "contracts": list(result.get("contracts", {}).keys()), # List contract names
        }
        return simplified_result
    except FlowAPIError as e:
        # Handle 404 Not Found specifically
        if "404" in str(e):
             return {"error": f"账户 {account_address} 未找到或尚未初始化。", "status": "not_found"}
        raise # Re-raise other errors

@mcp.tool()
async def get_transaction_details(transaction_id: str) -> Dict[str, Any]:
    """查询Flow交易详情 (使用 Flow HTTP API)

    Args:
        transaction_id: 交易ID，64位十六进制字符串 (不含 0x)
    """
    # Remove 0x prefix if present
    transaction_id = transaction_id.replace("0x", "")
    if not transaction_id or not isinstance(transaction_id, str) or len(transaction_id) != 64:
        raise ValueError("无效或缺失 'transaction_id' 参数 (应为64位十六进制字符串)")

    endpoint = f"/transactions/{transaction_id}"
    try:
        result = await make_async_flow_api_request(endpoint)
        # Potentially simplify or enrich the result here if needed
        # Example: decode event payloads if possible (complex)
        return result
    except FlowAPIError as e:
         if "404" in str(e):
              return {"error": f"交易 {transaction_id} 未找到。", "status": "not_found"}
         raise


@mcp.tool()
async def get_latest_sealed_block() -> Dict[str, Any]:
    """查询Flow最新已密封区块信息 (使用 Flow HTTP API)"""
    endpoint = "/blocks"
    try:
        # API returns a list, usually the first item is the latest sealed
        data = await make_async_flow_api_request(endpoint, params={"height": "sealed", "expand": "payload"})
        if isinstance(data, list) and len(data) > 0:
            block_info = data[0]
            simplified_result = {
                "id": block_info.get("id"),
                "height": block_info.get("height"),
                "timestamp": block_info.get("timestamp"),
                "parent_id": block_info.get("parent_id"),
                "collection_guarantees_count": len(block_info.get("collection_guarantees", [])),
                # Payload info requires 'expand=payload'
                "transaction_count": len(block_info.get("payload", {}).get("transactions", [])),
                "block_time_seconds": 0 # Calculate if previous block info is available
            }
             # Try to calculate block time if possible (needs previous block)
            if len(data) > 1:
                 try:
                     prev_ts = datetime.fromisoformat(data[1]['timestamp'].replace('Z', '+00:00'))
                     curr_ts = datetime.fromisoformat(block_info['timestamp'].replace('Z', '+00:00'))
                     simplified_result["block_time_seconds"] = (curr_ts - prev_ts).total_seconds()
                 except Exception:
                     pass # Ignore if calculation fails

            return simplified_result
        else:
            raise ValueError("未能从API响应中获取有效的最新区块信息")
    except FlowAPIError as e:
        raise RuntimeError(f"获取最新区块失败: {e}") from e


@mcp.tool()
async def get_account_transactions(account_address: str, limit: int = 10, type: Optional[str] = None) -> Dict[str, Any]:
    """获取Flow账户的交易历史 (优先使用 QuickNode Flowscan API)

    Args:
        account_address: Flow账户地址，以0x开头
        limit: 返回的交易数量限制，默认为10
        type: 可选，过滤交易类型 (例如 'transfer', 'nft_mint', 'contract_interaction') - *依赖 QuickNode API 支持*
    """
    if not account_address or not isinstance(account_address, str) or not account_address.startswith("0x"):
        raise ValueError("无效或缺失 'account_address' 参数")

    if QUICKNODE_FLOW_API_URL:
        try:
            logging.info(f"尝试使用 QuickNode API 获取账户 {account_address} 的交易历史")
            # **ASSUMPTION**: Endpoint is like '/v1/accounts/{address}/transactions' - VERIFY THIS
            endpoint = f"/v1/accounts/{account_address}/transactions"
            params = {"limit": limit}
            if type:
                params["type"] = type # Assuming QuickNode supports type filtering

            transactions_data = await make_async_quicknode_request(endpoint, params=params)

            # **ASSUMPTION**: QuickNode response format, adapt as needed
            # Example: {"transactions": [{"hash": "...", "timestamp": "...", "type": "...", "status": "...", ...}]}
            transactions = []
            if isinstance(transactions_data, dict) and "transactions" in transactions_data:
                for tx in transactions_data["transactions"][:limit]:
                     # Adapt field names based on actual QuickNode response
                    tx_data = {
                        "id": tx.get("hash") or tx.get("transactionId") or tx.get("id"),
                        "date": tx.get("timestamp") or tx.get("time") or tx.get("blockTimestamp"),
                        "type": tx.get("transactionType") or tx.get("type"),
                        "status": tx.get("status"),
                        "block_height": tx.get("blockHeight"),
                        # Add other relevant fields like sender, receiver, value if available
                        "details": tx.get("details") # Placeholder for any extra info
                    }
                    transactions.append(tx_data)

            # Optionally get account summary from QuickNode too if available
            # endpoint_summary = f"/v1/accounts/{account_address}/summary"
            # summary_data = await make_async_quicknode_request(endpoint_summary)
            # account_info = {"balance": summary_data.get("balance"), ... }

            return {
                # "account_info": account_info, # Add if fetched
                "total_transactions_returned": len(transactions),
                "limit": limit,
                "transactions": transactions,
                "source": "QuickNode Flowscan API"
            }
        except Exception as e:
            logging.warning(f"QuickNode API 获取交易历史失败: {e}. 尝试备用方法。", exc_info=False)
            # Fall through to fallback if QuickNode fails
    else:
        logging.warning("QuickNode API URL 未配置，将使用备用方法获取交易历史。")

    # Fallback if QuickNode is unavailable or fails
    return await get_account_transactions_fallback(account_address)

async def get_account_transactions_fallback(account_address: str) -> Dict[str, Any]:
    """备用方法：仅返回账户信息，因为 Flow API 不直接提供历史交易列表。"""
    logging.info(f"使用备用方法获取账户 {account_address} 的信息 (无交易历史)")
    try:
        account_info = await get_flow_account_info(account_address)
        return {
            "account_info": account_info,
            "total_transactions": 0,
            "transactions": [],
            "message": "Flow HTTP API 不直接提供账户历史交易列表。请使用 Flowscan.org 或配置 QuickNode Flow Simple API 以获取交易历史。",
            "source": "Flow HTTP API (Fallback - No Transactions)"
        }
    except Exception as e:
        logging.error(f"备用方法获取账户信息时出错: {e}")
        return {"error": f"获取账户信息失败 (备用): {str(e)}"}


@mcp.tool()
async def get_wallet_dashboard(account_address: str) -> Dict[str, Any]:
    """生成Flow钱包仪表盘，包含账户信息、近期交易和网络状态"""
    if not account_address or not isinstance(account_address, str) or not account_address.startswith("0x"):
        raise ValueError("无效或缺失 'account_address' 参数")

    logging.info(f"开始获取钱包仪表盘信息，地址: {account_address}")
    dashboard_data = {}
    start_time = datetime.now().isoformat()

    try:
        # 并行获取所有所需数据
        # Use create_task for true parallelism
        account_info_task = asyncio.create_task(get_flow_account_info(account_address))
        # Get recent transactions (limit to 5 for dashboard)
        transactions_task = asyncio.create_task(get_account_transactions(account_address, 5))
        latest_block_task = asyncio.create_task(get_latest_sealed_block())

        # Wait for all tasks to complete, gathering results or exceptions
        results = await asyncio.gather(
            account_info_task,
            transactions_task,
            latest_block_task,
            return_exceptions=True # Important to handle individual task failures
        )

        # Process results, checking for errors
        account_info = results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])}
        transactions_result = results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])}
        latest_block = results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])}

        # Build dashboard data
        dashboard_data = {
            "account_address": account_address,
            "account_info": account_info,
            "recent_transactions": transactions_result, # Renamed for clarity
            "network_status": {
                "latest_sealed_block": latest_block,
                "network": "mainnet" if "mainnet" in FLOW_API_BASE_URL else "testnet"
            },
            "generated_at": start_time,
            "flowscan_url": f"https://flowscan.org/account/{account_address}" # Use flowscan.org
        }

    except Exception as e:
        # Catch any unexpected errors during task gathering/processing
        error_message = f"生成钱包仪表盘时发生意外错误: {e}"
        logging.error(error_message, exc_info=True)
        if not dashboard_data:
            dashboard_data = {"error": error_message}
        else:
            # Add error even if some data was fetched
             dashboard_data["error"] = dashboard_data.get("error", "") + "; " + error_message

    return dashboard_data

@mcp.tool()
async def monitor_account_activity(account_address: str, check_interval_seconds: int = 60, monitor_duration_seconds: int = 180) -> Dict[str, Any]:
    """(演示) 监控Flow账户活动，周期性检查最新交易。

    注意: MCP工具通常是请求/响应模式，长时间运行的监控不适合。
         此函数模拟一个短时监控过程用于演示。

    Args:
        account_address: Flow账户地址，以0x开头
        check_interval_seconds: 检查间隔秒数，默认为60秒 (最低10秒)
        monitor_duration_seconds: 监控总时长（演示用），默认180秒
    """
    if not account_address or not isinstance(account_address, str) or not account_address.startswith("0x"):
        raise ValueError("无效或缺失 'account_address' 参数")

    check_interval_seconds = max(10, check_interval_seconds)
    monitor_duration_seconds = max(check_interval_seconds, monitor_duration_seconds) # Ensure at least one check

    logging.info(f"开始模拟监控账户 {account_address} 活动，间隔 {check_interval_seconds} 秒，持续 {monitor_duration_seconds} 秒")

    monitoring_result = {
        "account_address": account_address,
        "monitor_start_time": datetime.now().isoformat(),
        "monitor_end_time": None,
        "check_interval_seconds": check_interval_seconds,
        "monitor_duration_seconds": monitor_duration_seconds,
        "checks_performed": 0,
        "new_transactions_found": [],
        "monitor_status": "starting",
        "error": None,
        "note": "这是一个模拟监控，实际监控需要不同的架构。"
    }

    known_tx_ids = set()

    try:
        # Initial fetch to establish baseline (get IDs of recent Txs)
        initial_txs_result = await get_account_transactions(account_address, 20) # Fetch more initially
        if "transactions" in initial_txs_result and isinstance(initial_txs_result["transactions"], list):
            known_tx_ids.update(tx.get("id") for tx in initial_txs_result["transactions"] if tx.get("id"))
        elif "error" in initial_txs_result:
             raise ConnectionError(f"无法获取初始交易: {initial_txs_result['error']}")

        monitoring_result["initial_known_tx_count"] = len(known_tx_ids)
        monitoring_result["monitor_status"] = "running"
        logging.info(f"监控已启动，已知 {len(known_tx_ids)} 个初始交易ID。")

        start_time = time.monotonic()
        while (time.monotonic() - start_time) < monitor_duration_seconds:
            monitoring_result["checks_performed"] += 1
            current_check_time = datetime.now().isoformat()
            logging.info(f"执行检查 #{monitoring_result['checks_performed']} at {current_check_time}")

            try:
                current_txs_result = await get_account_transactions(account_address, 10) # Check recent Txs
                if "error" in current_txs_result:
                     logging.warning(f"检查 #{monitoring_result['checks_performed']} 获取交易失败: {current_txs_result['error']}")
                     # Decide whether to continue or stop monitoring on error
                     # For demo, we'll log and continue
                     await asyncio.sleep(check_interval_seconds)
                     continue

                new_found_in_check = []
                if "transactions" in current_txs_result and isinstance(current_txs_result["transactions"], list):
                    for tx in current_txs_result["transactions"]:
                        tx_id = tx.get("id")
                        if tx_id and tx_id not in known_tx_ids:
                            logging.info(f"发现新交易: {tx_id}")
                            new_found_in_check.append(tx)
                            known_tx_ids.add(tx_id)

                if new_found_in_check:
                    monitoring_result["new_transactions_found"].extend(new_found_in_check)

            except Exception as check_err:
                 logging.error(f"检查 #{monitoring_result['checks_performed']} 期间发生错误: {check_err}", exc_info=True)
                 # Log error and continue for demo purposes

            # Wait for the next interval
            await asyncio.sleep(check_interval_seconds)

        monitoring_result["monitor_status"] = "completed"
        monitoring_result["total_new_transactions_found"] = len(monitoring_result["new_transactions_found"])
        logging.info(f"模拟监控完成。发现 {monitoring_result['total_new_transactions_found']} 个新交易。")

    except Exception as e:
        error_message = f"监控账户活动时出错: {e}"
        logging.error(error_message, exc_info=True)
        monitoring_result["monitor_status"] = "error"
        monitoring_result["error"] = str(e)
    finally:
        monitoring_result["monitor_end_time"] = datetime.now().isoformat()

    return monitoring_result

# --- NFT Browser Functionality ---

@mcp.tool()
async def query_nft_collection_info(collection_address: str, collection_name: Optional[str] = None) -> Dict[str, Any]:
    """查询NFT集合信息 (优先使用 QuickNode, 回退到 Cadence/Flow API)

    Args:
        collection_address: NFT合约部署的账户地址
        collection_name: 可选，NFT合约名称 (如果知道)
    """
    if not collection_address or not isinstance(collection_address, str):
        raise ValueError("无效或缺失 'collection_address' 参数")

    logging.info(f"查询NFT集合信息: Address {collection_address}, Name {collection_name}")

    # --- Strategy 1: QuickNode Flow Simple API ---
    if QUICKNODE_FLOW_API_URL:
        try:
            logging.info(f"尝试使用 QuickNode API 查询NFT集合 {collection_address}")
            # **ASSUMPTION**: Endpoint like '/v1/nfts/collections/{address}' or '/v1/nfts/contracts/{address}' - VERIFY
            # Needs adjustment based on whether QuickNode uses contract address or contract name
            endpoint = f"/v1/nfts/collections/{collection_address}"
            # Or maybe endpoint = f"/v1/nfts/contracts/{collection_address}" if it's by contract owner
            # params = {} if collection_name else {} # Pass name if API supports filtering by name?

            qn_data = await make_async_quicknode_request(endpoint) # Add params if needed

            # **ASSUMPTION**: Adapt parsing based on actual QuickNode response structure
            # Example: {"name": "...", "description": "...", "totalSupply": 10000, "ownerCount": 1500, "floorPrice": {"amount": "10.5", "currency": "FLOW"}, ...}
            if qn_data and isinstance(qn_data, dict):
                 floor_price = qn_data.get("floorPrice") or qn_data.get("floor_price")
                 volume_24h = qn_data.get("volume24h") or qn_data.get("volume_24h")
                 collection_info = {
                     "name": qn_data.get("name", collection_name),
                     "address": collection_address, # Collection address might differ from contract owner address
                     "contract_owner_address": qn_data.get("contractOwnerAddress", collection_address),
                     "description": qn_data.get("description"),
                     "external_url": qn_data.get("externalUrl"),
                     "banner_image_url": qn_data.get("bannerImageUrl"),
                     "total_supply": qn_data.get("totalSupply"),
                     "holders_count": qn_data.get("ownerCount") or qn_data.get("holderCount"),
                     "floor_price": f"{floor_price['amount']} {floor_price['currency']}" if isinstance(floor_price, dict) else floor_price,
                     "volume_24h": f"{volume_24h['amount']} {volume_24h['currency']}" if isinstance(volume_24h, dict) else volume_24h,
                     "verified": qn_data.get("verified", True), # Assume verified if from QuickNode
                     "source": "QuickNode Flowscan API"
                 }
                 return collection_info
            else:
                 logging.warning("QuickNode API 未返回有效的集合数据。")

        except Exception as e:
            logging.warning(f"QuickNode API 查询NFT集合失败: {e}. 尝试备用方法。", exc_info=False)
            # Fall through to fallback

    # --- Strategy 2: Flow HTTP API + Cadence (Basic Info) ---
    logging.info(f"使用 Flow API / Cadence 查询NFT集合 {collection_address}")
    try:
        # 1. Verify account and contract existence
        account_data = await get_flow_account_info(collection_address)
        if "error" in account_data:
             return {"error": f"无法获取合约账户信息: {account_data['error']}", "address": collection_address}

        contracts = account_data.get("contracts", [])
        if collection_name and collection_name not in contracts:
             warning = f"警告：在地址 {collection_address} 上未找到名为 '{collection_name}' 的合约。"
             logging.warning(warning)
             # Continue, maybe name was wrong, try finding common NFT contract names
        elif not collection_name and not any(c in ["NonFungibleToken", "MetadataViews"] for c in contracts):
             # If no name given and no standard contracts found, it's less likely an NFT contract address
             warning = f"警告：地址 {collection_address} 上未找到明确的NFT相关合约。"
             logging.warning(warning)

        # 2. Execute Cadence script for MetadataViews (if possible)
        #    This requires knowing the public path or capability receiver for the collection display.
        #    This is HIGHLY collection-specific. Using a generic placeholder script.
        cadence_script = f"""
        import MetadataViews from 0x1d7e57aa55817448 // Mainnet MetadataViews address

        // Trying common public paths for collection display. Replace with actual path if known.
        pub fun main(address: Address): &MetadataViews.NFTCollectionDisplay? {{
            let account = getAccount(address)
            // Common public paths - ADD MORE specific paths if known for popular collections
            let commonPaths: [PublicPath] = [
                /public/NFTCollectionDisplay{collection_name or ''},
                /public/CollectionDisplay,
                /public/metadata // Generic fallback path
            ]

            for path in commonPaths {{
                if let cap = account.getCapability<&{MetadataViews.NFTCollectionDisplay}>(path).borrow() {{
                    return cap
                }}
            }}

            // Try finding via standard Collection interface if Display is not available
            // This requires the collection to implement the standard Collection interface at a known path
             // let collectionCap = account.getCapability(/public/NFTCollection) // Example path
             // if let collection = collectionCap.borrow<&{{Type<AnyResource{{MetadataViews.ResolverCollection}}>}}>() {{
             //     let display = collection.resolveView(Type<MetadataViews.NFTCollectionDisplay>())
             //     // ... need to cast and return display details
             // }}

            return nil
        }}
        """
        # Replace standard addresses if on Testnet
        if "testnet" in FLOW_API_BASE_URL:
             cadence_script = cadence_script.replace("0x1d7e57aa55817448", "0x631e88ae7f1d7c20") # MetadataViews Testnet

        collection_display_info = None
        try:
             script_result = await execute_cadence_script(cadence_script, args=[collection_address])
             if script_result and isinstance(script_result, dict): # Assuming decode returns a dict for the struct
                 collection_display_info = script_result # Contains name, description, externalURL, etc.
                 collection_display_info.pop("_type", None) # Remove internal type info
                 collection_display_info.pop("_id", None)
        except CadenceExecutionError as e:
             logging.warning(f"获取集合元数据视图失败: {e}")
             collection_display_info = {"error": f"获取 MetadataViews 失败: {e}"}
        except Exception as e:
             logging.error(f"执行集合元数据脚本时发生意外错误: {e}", exc_info=True)
             collection_display_info = {"error": f"执行元数据脚本时出错: {e}"}


        # 3. Get Total Supply (Requires knowing the collection's public type/path)
        #    This is also collection-specific. Placeholder script.
        cadence_script_supply = f"""
        import NonFungibleToken from 0x1d7e57aa55817448 // Mainnet NFT address

        // Assuming the collection resource is stored at a standard private path
        // and conforms to NonFungibleToken.Collection. This is a BIG assumption.
        // Or trying to borrow via a public capability if available at a known path.
        pub fun main(address: Address): UInt64? {{
            let account = getAccount(address)
            // Try common public path for NFT Collection standard
            let collectionCap = account.getCapability<&{NonFungibleToken.Collection}>(/public/NFTCollection) // Example path

            if let collectionRef = collectionCap.borrow() {{
                return collectionRef.totalSupply
            }}

            // Fallback: Maybe the collection is stored directly? (Less common for public query)
            // if let collectionRef = account.borrow<&NonFungibleToken.Collection>(from: /storage/NFTCollection) {{ // Example path
            //     return collectionRef.getIDs().length // Alternative if totalSupply not implemented
            // }}

            return nil // Cannot determine supply
        }}
        """
        # Replace standard addresses if on Testnet
        if "testnet" in FLOW_API_BASE_URL:
             cadence_script_supply = cadence_script_supply.replace("0x1d7e57aa55817448", "0x631e88ae7f1d7c20") # NFT Testnet

        total_supply = None
        try:
             supply_result = await execute_cadence_script(cadence_script_supply, args=[collection_address])
             if isinstance(supply_result, int):
                 total_supply = supply_result
        except CadenceExecutionError as e:
            logging.warning(f"获取集合 total supply 失败: {e}")
        except Exception as e:
             logging.error(f"执行 total supply 脚本时发生意外错误: {e}", exc_info=True)


        # Combine results
        collection_info = {
            "name": collection_display_info.get("name", collection_name or "未知名称") if collection_display_info and "error" not in collection_display_info else (collection_name or "未知名称"),
            "address": collection_address,
            "contract_owner_address": collection_address, # Simplification
            "description": collection_display_info.get("description") if collection_display_info and "error" not in collection_display_info else None,
            "external_url": collection_display_info.get("externalURL") if collection_display_info and "error" not in collection_display_info else None,
            "banner_image_url": collection_display_info.get("squareImage", {}).get("url") if collection_display_info and "error" not in collection_display_info else None, # Example field name
            "total_supply": total_supply,
            "holders_count": None, # Hard to get reliably via script without iterating
            "floor_price": None, # Not available via standard Flow API/Cadence
            "volume_24h": None, # Not available via standard Flow API/Cadence
            "verified": collection_name in contracts if collection_name else False, # Basic verification
            "metadata_status": "Unavailable" if not collection_display_info or "error" in collection_display_info else "Available",
            "total_supply_status": "Unavailable" if total_supply is None else "Available",
            "warning": warning if 'warning' in locals() else None,
            "source": "Flow HTTP API / Cadence (Limited Data)"
        }
        return collection_info

    except Exception as e:
        logging.error(f"查询NFT集合信息时发生意外错误: {e}", exc_info=True)
        return {
            "address": collection_address,
            "name": collection_name,
            "error": f"获取NFT集合信息失败: {str(e)}",
            "status": "error"
        }


@mcp.tool()
async def query_nft_metadata(collection_address: str, token_id: str) -> Dict[str, Any]:
    """查询NFT元数据信息 (使用 Cadence)

    Args:
        collection_address: NFT合约部署账户地址
        token_id: NFT代币ID (通常是 UInt64)
    """
    if not collection_address or not isinstance(collection_address, str):
        raise ValueError("无效或缺失 'collection_address' 参数")
    if not token_id or not isinstance(token_id, (str, int)): # Allow int or string ID
        raise ValueError("无效或缺失 'token_id' 参数")

    try:
        # Ensure token_id is string for script, handle potential conversion errors
        token_id_uint64 = int(token_id)
    except ValueError:
        raise ValueError(f"无效的 token_id: '{token_id}'. 必须是数字.")

    logging.info(f"查询NFT元数据: 合约地址 {collection_address}, 代币ID {token_id_uint64}")

    # --- Cadence Script using MetadataViews ---
    # This script assumes the collection exposes MetadataViews.Display publicly.
    # The public path is the most common unknown. Trying a few common ones.
    cadence_script = f"""
    import NonFungibleToken from 0x1d7e57aa55817448
    import MetadataViews from 0x1d7e57aa55817448

    // Function to try borrowing from multiple common paths
    access(all) fun getDisplayCapability(account: AuthAccount, commonPaths: [PublicPath]): Capability<&AnyResource{{MetadataViews.Resolver}}>{{MetadataViews.Resolver}}>? {{
         for path in commonPaths {{
            if let cap = account.getCapability<&AnyResource{{MetadataViews.Resolver}}>(path).borrow() {{
                 // Check if it can resolve the Display view for the given NFT ID
                 let resolver = cap as! &{MetadataViews.Resolver} // Need to cast to check
                 if let view = resolver.resolveView(Type<MetadataViews.Display>()) {{
                     // We don't actually need the view here, just confirming the capability is likely correct
                     // The main function will borrow the specific NFT resolver later
                     return cap // Return the capability itself
                 }}
            }}
         }}
         return nil
    }}

    pub fun main(ownerAddress: Address, tokenId: UInt64): AnyStruct? {{
        let account = getAuthAccount(ownerAddress) // Use AuthAccount to access public capabilities

        // Try common public paths for the collection's resolver capability. Adjust/add paths as needed.
         let commonPaths: [PublicPath] = [
             /public/NFTCollectionMetadata, // Example path convention
             /public/Collection, // Generic
             /public/TopShotCollection // Specific example
             // Add more known public paths for collections here...
         ]

        var targetCap: Capability<&AnyResource{{MetadataViews.Resolver}}>{{MetadataViews.Resolver}}>? = nil

        // Find the right capability
         // This part is tricky because we need the *collection's* resolver, not the NFT's.
         // A better approach might be to borrow the collection ref and then borrow the NFT from it.

         // --- Revised Approach: Borrow Collection then NFT ---
         // Assume common path for the Collection itself
         let collectionPath = /public/NFTCollection // <<<< REPLACE with the ACTUAL public path if known!

         let collectionCap = account.getCapability<&{NonFungibleToken.CollectionPublic}>(collectionPath)
         if !collectionCap.check() {{
             // Try other common collection paths...
             // let collectionPath = /public/SomeOtherCollectionPath
             // collectionCap = account.getCapability<&{NonFungibleToken.CollectionPublic}>(collectionPath)
             // ...
             return {{"error": "无法在已知路径找到NFT集合公共能力 (例如 /public/NFTCollection). 需要具体路径。" }}
         }}

         let collectionRef = collectionCap.borrow()
             ?? panic("无法借用NFT集合引用")

        // Borrow the specific NFT using the Resolver interface
        let resolverRef = collectionRef.borrowViewResolver(id: tokenId)

        // Resolve the Display view
        let view = resolverRef.resolveView(Type<MetadataViews.Display>())

        if view == nil {{
            return {{"error": "无法解析此NFT的 MetadataViews.Display 视图。"}}
        }}

        // Cast the view to the expected struct type
        let display = view as! MetadataViews.Display

        // Resolve other views if needed (e.g., Royalties, Traits, ExternalURL)
        let externalURLView = resolverRef.resolveView(Type<MetadataViews.ExternalURL>())
        let externalURL = (externalURLView as? MetadataViews.ExternalURL)?.url

        let royaltiesView = resolverRef.resolveView(Type<MetadataViews.Royalties>())
        let royalties: [MetadataViews.Royalty] = []
        if royaltiesView != nil {{
             let royaltyInfo = royaltiesView as! MetadataViews.Royalties
             royalties = royaltyInfo.getRoyalties() // This returns an array of Royalty structs
        }}

        // Example: Traits View (assuming it exists and follows MetadataViews standard)
        // let traitsView = resolverRef.resolveView(Type<MetadataViews.Traits>())
        // let traits = traitsView as? MetadataViews.Traits
        // let traitList = traits?.traits ?? [] // This is hypothetical

        // Build the result dictionary
        let result: {{String: AnyStruct}} = {{
            "name": display.name,
            "description": display.description,
            "thumbnail_url": display.thumbnail.uri(), // Assumes thumbnail is HTTPFile or IPFSFile
            "owner": ownerAddress, // Owner of the collection contract, not the NFT owner
            "external_url": externalURL ?? "",
            "royalties": royalties, // Array of royalty structs {receiver: Address, cut: UFix64, description: String}
            // "traits": traitList // Add if traits view is resolved
        }}

        // We need the current owner. This usually requires a different script/approach,
        // often involving querying a central registry or the collection's internal owner mapping if public.
        // For now, we return the collection owner address.

        return result
    }}
    """
     # Replace standard addresses if on Testnet
    if "testnet" in FLOW_API_BASE_URL:
        cadence_script = cadence_script.replace("0x1d7e57aa55817448", "0x631e88ae7f1d7c20") # NFT & MetadataViews Testnet

    try:
        metadata_result = await execute_cadence_script(cadence_script, args=[collection_address, token_id_uint64])

        if metadata_result and isinstance(metadata_result, dict) and "error" not in metadata_result:
            # Add token ID and collection address for context
            metadata_result["token_id"] = str(token_id_uint64) # Return as string
            metadata_result["collection_address"] = collection_address
             # Attempt to format royalties nicely
            if "royalties" in metadata_result and isinstance(metadata_result["royalties"], list):
                 formatted_royalties = []
                 for r in metadata_result["royalties"]:
                      if isinstance(r, dict):
                           formatted_royalties.append(
                                f"{r.get('receiver', '?')}: {r.get('cut', 0.0)*100:.2f}% ({r.get('description', '')})"
                           )
                 metadata_result["royalties_formatted"] = formatted_royalties

            return metadata_result
        elif metadata_result and "error" in metadata_result:
             raise CadenceExecutionError(f"脚本执行返回错误: {metadata_result['error']}")
        else:
             raise CadenceExecutionError("脚本执行未返回有效数据。")

    except (CadenceExecutionError, ValueError) as e:
        logging.error(f"查询NFT元数据时出错: {e}", exc_info=False)
        return {
            "token_id": str(token_id),
            "collection_address": collection_address,
            "error": f"获取NFT元数据失败: {str(e)}",
            "status": "error",
            "note": "可能原因：合约地址错误、Token ID不存在、集合未实现MetadataViews或公共路径未知。"
        }
    except Exception as e:
        logging.error(f"查询NFT元数据时发生意外错误: {e}", exc_info=True)
        return {
            "token_id": str(token_id),
            "collection_address": collection_address,
            "error": f"获取NFT元数据失败: {str(e)}",
            "status": "error"
        }


@mcp.tool()
async def get_nft_ownership_history(collection_address: str, token_id: str, limit: int = 10) -> Dict[str, Any]:
    """获取NFT所有权历史记录 (优先使用 QuickNode, 回退到 Flow API Events)

    Args:
        collection_address: NFT合约地址
        token_id: NFT代币ID
        limit: 返回的历史记录数量限制，默认为10
    """
    if not collection_address or not isinstance(collection_address, str):
        raise ValueError("无效或缺失 'collection_address' 参数")
    if not token_id or not isinstance(token_id, (str, int)):
        raise ValueError("无效或缺失 'token_id' 参数")
    try:
        token_id_str = str(int(token_id)) # Normalize to string for comparison/filtering
    except ValueError:
        raise ValueError(f"无效的 token_id: '{token_id}'. 必须是数字.")

    logging.info(f"获取NFT所有权历史: 合约 {collection_address}, 代币ID {token_id_str}, 限制 {limit}")

    # --- Strategy 1: QuickNode Flow Simple API ---
    if QUICKNODE_FLOW_API_URL:
        try:
            logging.info(f"尝试使用 QuickNode API 获取NFT {collection_address}/{token_id_str} 的历史记录")
            # **ASSUMPTION**: Endpoint like '/v1/nfts/collections/{address}/tokens/{token_id}/transfers' - VERIFY
            endpoint = f"/v1/nfts/collections/{collection_address}/tokens/{token_id_str}/transfers"
            params = {"limit": limit, "sort": "desc"} # Assuming limit and sort params

            qn_data = await make_async_quicknode_request(endpoint, params=params)

            # **ASSUMPTION**: Adapt parsing based on actual QuickNode response structure
            # Example: {"transfers": [{"from": "...", "to": "...", "timestamp": "...", "transactionId": "...", "type": "Mint/Transfer/Sale", "price": {...}}, ...]}
            if qn_data and isinstance(qn_data, dict) and "transfers" in qn_data:
                history = []
                current_owner = None
                for i, transfer in enumerate(qn_data["transfers"][:limit]):
                    # Adapt field names
                    from_addr = transfer.get("from") or transfer.get("fromAddress")
                    to_addr = transfer.get("to") or transfer.get("toAddress")
                    price_info = transfer.get("price")

                    # Determine current owner from the latest transfer's 'to' address
                    if i == 0 and to_addr: # Assuming sorted desc
                        current_owner = to_addr

                    history.append({
                        "date": transfer.get("timestamp") or transfer.get("blockTimestamp"),
                        "from_address": from_addr if from_addr != "0x0000000000000000" else "0x0 (Mint)",
                        "to_address": to_addr,
                        "transaction_id": transfer.get("transactionId") or transfer.get("txHash"),
                        "transaction_type": transfer.get("type") or transfer.get("transferType"), # Mint, Transfer, Sale etc.
                        "price": f"{price_info['amount']} {price_info['currency']}" if isinstance(price_info, dict) else price_info,
                        "block_height": transfer.get("blockHeight")
                    })

                return {
                    "token_id": token_id_str,
                    "collection_address": collection_address,
                    "current_owner": current_owner,
                    "ownership_history": history,
                    "total_records_returned": len(history),
                    "limit": limit,
                    "source": "QuickNode Flowscan API"
                }
            else:
                logging.warning("QuickNode API 未返回有效的NFT传输数据。")

        except Exception as e:
            logging.warning(f"QuickNode API 获取NFT历史失败: {e}. 尝试备用方法。", exc_info=False)
            # Fall through to fallback

    # --- Strategy 2: Flow HTTP API Events ---
    logging.info(f"使用 Flow API Events 查询NFT {collection_address}/{token_id_str} 历史")
    try:
        # This requires knowing the exact event names emitted by the NFT contract.
        # Standard NonFungibleToken events are `Withdraw` and `Deposit`.
        # Format: A.{contract_address}.{ContractName}.{EventName}
        # We need the ContractName. Let's try to get it from account info.
        contract_name = None
        try:
            account_info = await get_flow_account_info(collection_address)
            # Find a contract likely to be the NFT contract (heuristic)
            potential_names = [k for k, v in account_info.get('contracts_details', {}).items() if 'NonFungibleToken' in v]
            if potential_names:
                 contract_name = potential_names[0] # Guessing the first one
            elif account_info.get("contracts"):
                 contract_name = account_info["contracts"][0] # Last resort guess
            else:
                 raise ValueError("无法确定NFT合约名称以查询事件。")
        except Exception as e:
            logging.error(f"无法确定NFT合约名称: {e}")
            return {"error": f"获取NFT历史失败：无法确定NFT合约名称以查询事件。", "status": "error"}

        # Standard Event Types
        # NOTE: Flow event query requires full type: A.CONTRACT_ADDRESS.CONTRACT_NAME.EVENT_NAME
        # Address needs to be padded to 16 hex digits without 0x for the type string
        padded_address = collection_address[2:].zfill(16)
        deposit_event = f"A.{padded_address}.{contract_name}.Deposit"
        withdraw_event = f"A.{padded_address}.{contract_name}.Withdraw"
        # Some contracts might emit a single Transfer event
        transfer_event = f"A.{padded_address}.{contract_name}.Transfer"

        all_events = []
        # Querying events can be slow and might require pagination or specific block ranges.
        # The `/events` endpoint needs a block range. Let's query the last N blocks (e.g., 1000).
        # This is NOT a reliable way to get FULL history. An indexer (like QuickNode) is needed.
        latest_block_info = await get_latest_sealed_block()
        end_height = latest_block_info.get("height")
        start_height = max(0, end_height - 10000) # Query last 10k blocks (adjust as needed, can be slow)

        logging.info(f"查询区块 {start_height} 到 {end_height} 的事件: {deposit_event}, {withdraw_event}, {transfer_event}")

        for event_type in [deposit_event, withdraw_event, transfer_event]:
             try:
                 # The API needs event type without the 'A.' prefix in the query parameter? Check docs.
                 # Let's assume the full type string is needed.
                 params = {
                     "type": event_type,
                     "start_block": str(start_height),
                     "end_block": str(end_height)
                 }
                 # Note: This endpoint structure might differ, check Flow HTTP API docs for /events query
                 # It might be /blocks/{start}-{end}/events?type=...
                 # Let's assume a simpler /events endpoint for now (may need correction)
                 endpoint = "/events" # This might need block range in path?
                 event_results = await make_async_flow_api_request(endpoint, params=params)

                 # Process results (API returns events grouped by block)
                 if isinstance(event_results, list):
                      for block_events in event_results:
                           for event in block_events.get("events", []):
                                # Decode event payload (JSON string within the event)
                                try:
                                     payload_data = json.loads(event.get("payload", "{}"))
                                     # Check if this event involves the target token ID
                                     event_token_id = payload_data.get("value", {}).get("fields", {}).get("id", {}).get("value")
                                     if event_token_id == token_id_str:
                                          all_events.append({
                                               "type": event.get("type"),
                                               "transaction_id": event.get("transaction_id"),
                                               "block_height": block_events.get("block_height"),
                                               "timestamp": block_events.get("block_timestamp"),
                                               "payload": payload_data.get("value", {}).get("fields", {}) # Extract fields like 'from', 'to'
                                          })
                                except (json.JSONDecodeError, AttributeError, KeyError) as e:
                                     logging.warning(f"无法解析事件 payload: {event.get('payload')}, Error: {e}")
                                     continue # Skip malformed events
             except FlowAPIError as e:
                  logging.warning(f"查询事件 {event_type} 失败: {e}")
                  # Continue to try other event types

        # Sort events by block height descending
        all_events.sort(key=lambda x: x.get("block_height", 0), reverse=True)

        # Process sorted events to build history (This logic is complex)
        history = []
        current_owner = None
        # Reconstruct transfers from Withdraw/Deposit pairs or single Transfer events
        # This requires careful state management and matching events by transaction ID.
        # For simplicity, we just list raw events involving the token ID.
        for event in all_events[:limit]:
             fields = event.get("payload", {})
             from_addr = fields.get("from", {}).get("value")
             to_addr = fields.get("to", {}).get("value")
             id_val = fields.get("id", {}).get("value") # Should match token_id_str

             # Determine type based on event name
             tx_type = "Unknown"
             if event['type'].endswith("Deposit"): tx_type = "Deposit"
             if event['type'].endswith("Withdraw"): tx_type = "Withdraw"
             if event['type'].endswith("Transfer"): tx_type = "Transfer"

             # Try to determine owner from latest deposit/transfer 'to'
             if not current_owner and (tx_type == "Deposit" or tx_type == "Transfer") and to_addr:
                  current_owner = to_addr.get('value') if isinstance(to_addr, dict) else to_addr


             history.append({
                 "date": event.get("timestamp"),
                 "from_address": from_addr.get('value') if isinstance(from_addr, dict) else from_addr,
                 "to_address": to_addr.get('value') if isinstance(to_addr, dict) else to_addr,
                 "transaction_id": event.get("transaction_id"),
                 "transaction_type": tx_type,
                 "price": None, # Not available from events
                 "block_height": event.get("block_height")
             })

        return {
            "token_id": token_id_str,
            "collection_address": collection_address,
            "current_owner": current_owner, # May be inaccurate with simple event list
            "ownership_history": history,
            "total_records_returned": len(history),
            "limit": limit,
            "source": "Flow HTTP API Events (Limited History / Accuracy)",
            "warning": "History derived from recent block events may be incomplete or lack context (e.g., sale price)."
        }

    except Exception as e:
        logging.error(f"获取NFT所有权历史时发生意外错误: {e}", exc_info=True)
        return {
            "token_id": token_id_str,
            "collection_address": collection_address,
            "error": f"获取NFT所有权历史失败: {str(e)}",
            "status": "error"
        }


@mcp.tool()
async def analyze_nft_collection_stats(collection_address: str) -> Dict[str, Any]:
    """分析NFT集合的统计数据和市场表现 (优先使用 QuickNode)

    Args:
        collection_address: NFT集合(合约)地址
    """
    if not collection_address or not isinstance(collection_address, str):
        raise ValueError("无效或缺失 'collection_address' 参数")

    logging.info(f"分析NFT集合统计数据: {collection_address}")

    # --- Strategy 1: QuickNode Flow Simple API ---
    if QUICKNODE_FLOW_API_URL:
        try:
            logging.info(f"尝试使用 QuickNode API 获取NFT集合 {collection_address} 的统计数据")
            # **ASSUMPTION**: Endpoint like '/v1/nfts/collections/{address}/stats' - VERIFY
            endpoint = f"/v1/nfts/collections/{collection_address}/stats"
            qn_data = await make_async_quicknode_request(endpoint)

            # **ASSUMPTION**: Adapt parsing based on actual QuickNode response structure
            # Example: {"name": "...", "totalSupply": ..., "ownerCount": ..., "floorPrice": {...}, "volume1d": {...}, "volume7d": {...}, "volumeAllTime": {...}, "marketCap": {...}}
            if qn_data and isinstance(qn_data, dict):
                fp = qn_data.get("floorPrice") or qn_data.get("floor_price")
                vol1d = qn_data.get("volume1d") or qn_data.get("volume_1d")
                vol7d = qn_data.get("volume7d") or qn_data.get("volume_7d")
                vol_all = qn_data.get("volumeAllTime") or qn_data.get("volume_all_time")
                market_cap = qn_data.get("marketCap") or qn_data.get("market_cap")
                holders = qn_data.get("ownerCount") or qn_data.get("holderCount")
                supply = qn_data.get("totalSupply") or qn_data.get("total_supply")
                holder_ratio = (holders / supply * 100) if holders and supply else 0

                stats = {
                    "collection_address": collection_address,
                    "collection_name": qn_data.get("name", "未知名称"),
                    "total_supply": supply,
                    "unique_holders": holders,
                    "holder_ratio": f"{holder_ratio:.2f}%",
                    "floor_price": f"{fp['amount']} {fp['currency']}" if isinstance(fp, dict) else fp,
                    "volume_1d": f"{vol1d['amount']} {vol1d['currency']}" if isinstance(vol1d, dict) else vol1d,
                    "volume_7d": f"{vol7d['amount']} {vol7d['currency']}" if isinstance(vol7d, dict) else vol7d,
                    "volume_all_time": f"{vol_all['amount']} {vol_all['currency']}" if isinstance(vol_all, dict) else vol_all,
                    "market_cap": f"{market_cap['amount']} {market_cap['currency']}" if isinstance(market_cap, dict) else market_cap,
                    # Add more stats if provided by QuickNode (avg price, highest sale, etc.)
                    "last_updated": qn_data.get("lastUpdated", datetime.now().isoformat()),
                    "source": "QuickNode Flowscan API"
                }
                # Add simple activity analysis based on available data
                # Requires more data (e.g., volume change %) for the original calculate_activity_score logic
                if vol1d or vol7d:
                     stats["market_activity_level"] = "活跃" if float(vol1d or 0) > 100 else "中等" # Very basic guess
                else:
                     stats["market_activity_level"] = "低/未知"

                return stats
            else:
                logging.warning("QuickNode API 未返回有效的集合统计数据。")

        except Exception as e:
            logging.warning(f"QuickNode API 获取NFT统计失败: {e}. 尝试备用方法。", exc_info=False)
            # Fall through to fallback

    # --- Strategy 2: Basic info from `query_nft_collection_info` ---
    logging.warning("无法从 QuickNode 获取统计数据，将返回基础集合信息。")
    try:
        basic_info = await query_nft_collection_info(collection_address)
        if "error" in basic_info:
            return basic_info # Return the error from the previous call

        # Return limited stats based on basic info
        stats = {
            "collection_address": collection_address,
            "collection_name": basic_info.get("name", "未知名称"),
            "total_supply": basic_info.get("total_supply"),
            "unique_holders": basic_info.get("holders_count"), # Likely None
            "holder_ratio": None,
            "floor_price": None,
            "volume_1d": None,
            "volume_7d": None,
            "volume_all_time": None,
            "market_cap": None,
            "market_activity_level": "未知 (需要 QuickNode API)",
            "last_updated": datetime.now().isoformat(),
            "source": basic_info.get("source", "Flow HTTP API / Cadence"),
            "warning": "统计数据有限，仅包含基本信息。需要 QuickNode API 获取市场数据。"
        }
        return stats

    except Exception as e:
        logging.error(f"分析NFT集合统计数据时发生意外错误: {e}", exc_info=True)
        return {
            "collection_address": collection_address,
            "error": f"分析NFT集合统计数据失败: {str(e)}",
            "status": "error"
        }

# --- NFT Dashboard ---
@mcp.tool()
async def nft_dashboard(collection_address: str, token_id: Optional[str] = None) -> Dict[str, Any]:
    """生成NFT集合或单个NFT的综合仪表盘"""
    if not collection_address or not isinstance(collection_address, str):
        raise ValueError("无效或缺失 'collection_address' 参数")

    start_time = datetime.now().isoformat()
    tasks = {}
    dashboard_data = {
        "request_type": "single_nft" if token_id else "collection",
        "collection_address": collection_address,
        "generated_at": start_time,
    }

    if token_id:
        logging.info(f"生成单个NFT仪表盘: 合约 {collection_address}, 代币ID {token_id}")
        tasks["metadata"] = asyncio.create_task(query_nft_metadata(collection_address, token_id))
        tasks["history"] = asyncio.create_task(get_nft_ownership_history(collection_address, token_id, limit=5)) # Limit history for dashboard
        tasks["collection_stats"] = asyncio.create_task(analyze_nft_collection_stats(collection_address)) # Add collection context
    else:
        logging.info(f"生成NFT集合仪表盘: 合约 {collection_address}")
        tasks["collection_info"] = asyncio.create_task(query_nft_collection_info(collection_address))
        tasks["collection_stats"] = asyncio.create_task(analyze_nft_collection_stats(collection_address))
        # Optional: Add recent activity/sales if available from QuickNode?
        # tasks["recent_sales"] = asyncio.create_task(get_collection_recent_sales(collection_address, limit=5))

    # Common task
    tasks["network_status"] = asyncio.create_task(get_latest_sealed_block())

    # Gather results
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    task_keys = list(tasks.keys())

    # Populate dashboard, handling errors for each task
    for i, key in enumerate(task_keys):
         if isinstance(results[i], Exception):
             dashboard_data[key] = {"error": str(results[i])}
             logging.error(f"生成NFT仪表盘时任务 '{key}' 失败: {results[i]}")
         else:
             dashboard_data[key] = results[i]

    # Add derived/combined info
    if token_id:
        dashboard_data["token_id"] = token_id
        # Try to get current owner from history if available
        if isinstance(dashboard_data.get("history"), dict) and "current_owner" in dashboard_data["history"]:
            dashboard_data["current_owner"] = dashboard_data["history"]["current_owner"]
        # Link to explorer for the specific NFT
        dashboard_data["explorer_url"] = f"https://flowscan.org/nft/{collection_address}/{token_id}" # Adjust URL format if needed
    else:
         dashboard_data["explorer_url"] = f"https://flowscan.org/contract/{collection_address}" # Link to collection/contract

    # Rename network status block for clarity
    if "network_status" in dashboard_data and isinstance(dashboard_data["network_status"], dict) and "error" not in dashboard_data["network_status"]:
         dashboard_data["network_status"] = {
             "latest_sealed_block": dashboard_data["network_status"],
             "network": "mainnet" if "mainnet" in FLOW_API_BASE_URL else "testnet"
         }

    # Check for overall errors
    if any(isinstance(res, Exception) for res in results):
        dashboard_data["status"] = "partial_error"
        dashboard_data["error_summary"] = "一个或多个数据获取任务失败。"

    return dashboard_data


# --- DeFi Analysis Functionality (Highly Simplified) ---

@mcp.tool()
async def query_defi_protocol_info(protocol_address: str) -> Dict[str, Any]:
    """查询DeFi协议信息 (优先 QuickNode, 回退 基础账户信息)

    Args:
        protocol_address: DeFi协议主要合约账户地址
    """
    if not protocol_address or not isinstance(protocol_address, str) or not protocol_address.startswith("0x"):
        raise ValueError("无效或缺失 'protocol_address' 参数")

    logging.info(f"查询DeFi协议信息: {protocol_address}")

    # --- Strategy 1: QuickNode Flow Simple API ---
    if QUICKNODE_FLOW_API_URL:
        try:
            logging.info(f"尝试使用 QuickNode API 查询DeFi协议 {protocol_address}")
            # **ASSUMPTION**: Endpoint like '/v1/defi/protocols/{address}/summary' - VERIFY
            endpoint = f"/v1/defi/protocols/{protocol_address}/summary"
            qn_data = await make_async_quicknode_request(endpoint)

            # **ASSUMPTION**: Adapt parsing based on actual QuickNode response structure
            # Example: {"name": "...", "category": "DEX", "tvlUsd": "1234567.89", "volume24hUsd": "12345.67", "userCount": 1234, "chains": ["flow"], ...}
            if qn_data and isinstance(qn_data, dict):
                 tvl = qn_data.get("tvlUsd") or qn_data.get("tvl_usd")
                 vol24h = qn_data.get("volume24hUsd") or qn_data.get("volume_24h_usd")
                 users = qn_data.get("userCount") or qn_data.get("users_24h") # Check field name

                 protocol_info = {
                    "address": protocol_address,
                    "name": qn_data.get("name", "未知协议"),
                    "category": qn_data.get("category") or qn_data.get("type"), # DEX, Lending, etc.
                    "description": qn_data.get("description"),
                    "tvl_usd": tvl,
                    "volume_24h_usd": vol24h,
                    "unique_users": users,
                    # Add more if available: launch date, website, social links...
                    "website": qn_data.get("websiteUrl") or qn_data.get("website"),
                    "twitter": qn_data.get("twitterUrl"),
                    "audit_info": qn_data.get("audits"), # List of audits?
                    "verified": True,
                    "source": "QuickNode Flowscan API"
                 }
                 return protocol_info
            else:
                logging.warning("QuickNode API 未返回有效的DeFi协议数据。")

        except Exception as e:
            logging.warning(f"QuickNode API 查询DeFi协议失败: {e}. 尝试备用方法。", exc_info=False)
            # Fall through to fallback

    # --- Strategy 2: Basic Account Info ---
    logging.warning(f"无法从 QuickNode 获取DeFi统计数据，将返回基础账户信息。")
    try:
        account_info = await get_flow_account_info(protocol_address)
        if "error" in account_info:
            return {"error": f"无法获取协议账户信息: {account_info['error']}", "address": protocol_address}

        protocol_info = {
            "address": protocol_address,
            "name": f"未知协议 @ {protocol_address}", # Placeholder name
            "category": "未知",
            "description": "需要 QuickNode API 或协议特定查询以获取详细信息。",
            "tvl_usd": None,
            "volume_24h_usd": None,
            "unique_users": None,
            "contracts_count": account_info.get("contracts_count"),
            "contracts_list": account_info.get("contracts"),
            "verified": True, # Account exists
            "source": "Flow HTTP API (Basic Account Info Only)",
            "warning": "仅返回基础账户信息。需要 QuickNode API 或协议特定查询获取 TVL、交易量等 DeFi 指标。"
        }
        return protocol_info

    except Exception as e:
        logging.error(f"查询DeFi协议信息时发生意外错误: {e}", exc_info=True)
        return {
            "address": protocol_address,
            "error": f"查询DeFi协议信息失败: {str(e)}",
            "status": "error"
        }


@mcp.tool()
async def get_liquidity_pools(protocol_address: str, limit: int = 5) -> Dict[str, Any]:
    """获取DeFi协议的流动性池信息 (优先 QuickNode, 回退 Cadence - 需特定脚本)

    Args:
        protocol_address: DeFi协议合约地址
        limit: 返回的流动性池数量限制，默认为5
    """
    if not protocol_address or not isinstance(protocol_address, str) or not protocol_address.startswith("0x"):
        raise ValueError("无效或缺失 'protocol_address' 参数")

    logging.info(f"获取流动性池信息: {protocol_address}, 限制 {limit}")

    # --- Strategy 1: QuickNode Flow Simple API ---
    if QUICKNODE_FLOW_API_URL:
        try:
            logging.info(f"尝试使用 QuickNode API 获取协议 {protocol_address} 的流动性池")
            # **ASSUMPTION**: Endpoint like '/v1/defi/protocols/{address}/pools' - VERIFY
            endpoint = f"/v1/defi/protocols/{protocol_address}/pools"
            params = {"limit": limit, "sortBy": "tvlUsd", "order": "desc"} # Example params

            qn_data = await make_async_quicknode_request(endpoint, params=params)

            # **ASSUMPTION**: Adapt parsing based on actual QuickNode response structure
            # Example: {"pools": [{"id": "...", "name": "FLOW/USDC", "token0": {...}, "token1": {...}, "tvlUsd": "...", "volume24hUsd": "...", "apr": "12.5", ...}, ...]}
            if qn_data and isinstance(qn_data, dict) and "pools" in qn_data:
                pools = []
                total_tvl = 0.0
                total_volume = 0.0
                for pool in qn_data["pools"][:limit]:
                     pool_tvl = float(pool.get("tvlUsd", 0.0))
                     pool_vol = float(pool.get("volume24hUsd", 0.0))
                     total_tvl += pool_tvl
                     total_volume += pool_vol
                     pools.append({
                        "pool_id": pool.get("id") or pool.get("poolId"),
                        "name": pool.get("name") or f"{pool.get('token0', {}).get('symbol', '?')}/{pool.get('token1', {}).get('symbol', '?')}",
                        "token_a": pool.get("token0"), # {"symbol": "...", "address": "...", "reserve": "..."}
                        "token_b": pool.get("token1"),
                        "total_liquidity_usd": pool.get("tvlUsd"),
                        "volume_24h_usd": pool.get("volume24hUsd"),
                        "fees_24h_usd": pool.get("fees24hUsd"), # Assuming available
                        "apr_percent": pool.get("apr") or pool.get("apy"), # Check field name
                        "creation_date": pool.get("creationTimestamp") # Assuming available
                     })

                return {
                    "protocol_address": protocol_address,
                    "total_pools_returned": len(pools),
                    "limit": limit,
                    "pools": pools,
                    "aggregate_tvl_usd": f"{total_tvl:.2f}",
                    "aggregate_volume_24h_usd": f"{total_volume:.2f}",
                    "source": "QuickNode Flowscan API"
                }
            else:
                logging.warning("QuickNode API 未返回有效的流动性池数据。")

        except Exception as e:
            logging.warning(f"QuickNode API 获取流动性池失败: {e}. 尝试备用方法。", exc_info=False)
            # Fall through to fallback

    # --- Strategy 2: Cadence (Requires Protocol-Specific Script) ---
    logging.warning("无法从 QuickNode 获取流动性池。需要协议特定的 Cadence 脚本。")
    # Placeholder: Execute a known script if available, otherwise return error/message.
    # Example: If we know this is 'FlowSwap', we could run a FlowSwap-specific script.
    # Since we don't know the protocol, we cannot proceed with Cadence generically.
    protocol_name = "未知协议" # Could try getting name from query_defi_protocol_info
    try:
        proto_info = await query_defi_protocol_info(protocol_address)
        protocol_name = proto_info.get("name", protocol_name)
    except Exception:
        pass # Ignore errors here

    return {
        "protocol_address": protocol_address,
        "protocol_name": protocol_name,
        "error": "获取流动性池失败：需要 QuickNode API 或协议特定的 Cadence 脚本。",
        "status": "requires_specific_script",
        "message": f"无法为通用协议地址 {protocol_address} 获取流动性池。如果知道协议类型（例如 FlowSwap, IncrementFi），可能需要使用针对该协议的特定工具或脚本。",
        "source": "Cadence (Not Attempted - Requires Specific Script)"
    }


@mcp.tool()
async def analyze_defi_portfolio(account_address: str) -> Dict[str, Any]:
    """分析用户的DeFi投资组合 (高度简化 - 仅显示余额)

    注意: 这是一个非常复杂的功能。真实实现需要索引服务或针对已知协议的
          大量 Cadence 脚本来查询 LP 头寸、质押、借贷等。
          此版本仅为占位符，返回基本账户信息。

    Args:
        account_address: 用户钱包地址
    """
    if not account_address or not isinstance(account_address, str) or not account_address.startswith("0x"):
        raise ValueError("无效或缺失 'account_address' 参数")

    logging.warning(f"分析 DeFi 投资组合功能高度简化，仅返回账户余额。完整实现需要索引器或特定脚本。")

    try:
        # 获取账户基本信息
        account_info = await get_flow_account_info(account_address)
        if "error" in account_info:
            return {"error": f"无法获取账户信息: {account_info['error']}", "account_address": account_address}

        # Placeholder Portfolio Structure
        portfolio = {
            "account_address": account_address,
            "account_balance_flow": account_info.get("balance_flow", "0.0"),
            "total_defi_value_usd": None, # Needs price feeds and position data
            "portfolio_allocation": [], # Requires querying specific protocols
            "tracked_protocols": [], # List protocols checked (none in this version)
            "risk_assessment": {
                "overall_risk": "未知",
                "message": "无法评估风险，缺少DeFi头寸数据。"
            },
            "last_updated": datetime.now().isoformat(),
            "status": "simplified",
            "message": "DeFi 投资组合分析功能高度简化。仅显示账户余额。完整分析需要集成第三方索引服务或为特定DeFi协议编写查询脚本。"
        }
        return portfolio

    except Exception as e:
        logging.error(f"分析DeFi投资组合时发生意外错误: {e}", exc_info=True)
        return {
            "account_address": account_address,
            "error": f"分析DeFi投资组合失败: {str(e)}",
            "status": "error"
        }


@mcp.tool()
async def get_token_price_history(token_address: str, days: int = 30) -> Dict[str, Any]:
    """获取代币价格历史数据 (使用 CoinGecko API 示例)

    Args:
        token_address: 代币合约地址 (需要映射到价格 API ID)
        days: 历史数据天数，默认30天 (最大值取决于API)
    """
    if not token_address or not isinstance(token_address, str) or not token_address.startswith("0x"):
        raise ValueError("无效或缺失 'token_address' 参数")

    # Map Flow address to Price API ID (CoinGecko example)
    price_api_id = TOKEN_ADDRESS_TO_PRICE_ID.get(token_address.lower()) # Use lower case for matching
    if not price_api_id:
        return {
            "token_address": token_address,
            "error": f"无法找到地址 {token_address} 对应的价格API ID。请更新 TOKEN_ADDRESS_TO_PRICE_ID 映射。",
            "status": "missing_mapping"
        }

    # Validate days (CoinGecko limits based on granularity)
    if days <= 0: days = 1
    if days > 90: # CoinGecko provides daily data > 90 days
        interval = "daily"
    else: # Hourly data for <= 90 days
         interval = None # Let API decide default (often hourly)
         # If you specifically want daily for < 90 days, set interval='daily'

    logging.info(f"获取代币价格历史: {price_api_id} ({token_address}), {days}天")

    try:
        # --- Fetch Current Price & Market Data ---
        # Endpoint: /simple/price?ids={id}&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true&include_24hr_change=true&include_last_updated_at=true
        current_data_endpoint = "/simple/price"
        current_params = {
            "ids": price_api_id,
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true"
        }
        current_price_data = await make_async_price_api_request(current_data_endpoint, params=current_params)

        if not current_price_data or price_api_id not in current_price_data:
            raise PriceAPIError(f"未能从价格API获取当前数据 for {price_api_id}")

        token_data = current_price_data[price_api_id]
        current_price_usd = token_data.get("usd")
        market_cap_usd = token_data.get("usd_market_cap")
        volume_24h_usd = token_data.get("usd_24h_vol")
        change_24h_percent = token_data.get("usd_24h_change")
        last_updated_at = token_data.get("last_updated_at") # Unix timestamp

        # --- Fetch Historical Chart Data ---
        # Endpoint: /coins/{id}/market_chart?vs_currency=usd&days={days}&interval={interval}
        history_endpoint = f"/coins/{price_api_id}/market_chart"
        history_params = {
            "vs_currency": "usd",
            "days": str(days)
        }
        if interval:
             history_params["interval"] = interval

        historical_data = await make_async_price_api_request(history_endpoint, params=history_params)

        if not historical_data or "prices" not in historical_data:
             raise PriceAPIError(f"未能从价格API获取历史图表数据 for {price_api_id}")

        # Process historical data [[timestamp_ms, price], ...]
        price_history = []
        chart_labels = []
        chart_prices = []
        chart_volumes = [] # Volume might be in a separate key 'total_volumes'

        for point in historical_data.get("prices", []):
             dt_obj = datetime.fromtimestamp(point[0] / 1000) # Convert ms timestamp
             date_str = dt_obj.strftime("%Y-%m-%d" + (" %H:%M" if interval != "daily" else ""))
             price = round(point[1], 6) # Adjust rounding as needed
             price_history.append({"date": date_str, "price_usd": price})
             chart_labels.append(date_str)
             chart_prices.append(price)

        # Process volume data if available [[timestamp_ms, volume], ...]
        for point in historical_data.get("total_volumes", []):
             # Align volume with the closest price timestamp if needed, or just append
             chart_volumes.append(round(point[1], 2))
             # Could add volume to price_history dict if timestamps match perfectly

        # Try getting token metadata (name, symbol)
        token_meta_endpoint = f"/coins/{price_api_id}"
        token_meta_params = {"localization": "false", "tickers": "false", "market_data": "false", "community_data": "false", "developer_data": "false", "sparkline": "false"}
        token_meta = await make_async_price_api_request(token_meta_endpoint, params=token_meta_params)
        token_symbol = token_meta.get("symbol", "?").upper()
        token_name = token_meta.get("name", "未知代币")


        # Assemble final result
        token_info = {
            "address": token_address,
            "price_api_id": price_api_id,
            "symbol": token_symbol,
            "name": token_name,
            "current_price_usd": current_price_usd,
            "price_change_24h_percent": round(change_24h_percent, 2) if change_24h_percent is not None else None,
            "market_cap_usd": round(market_cap_usd, 2) if market_cap_usd is not None else None,
            "volume_24h_usd": round(volume_24h_usd, 2) if volume_24h_usd is not None else None,
            "last_updated": datetime.fromtimestamp(last_updated_at).isoformat() if last_updated_at else None,
            "source": "CoinGecko API (Example)" # Specify source
        }

        return {
            "token_info": token_info,
            "days": days,
            "price_history_usd": price_history,
            "chart_data": {
                "labels": chart_labels,
                "prices_usd": chart_prices,
                "volumes_usd": chart_volumes if len(chart_volumes) == len(chart_labels) else [] # Ensure length matches
            }
        }

    except PriceAPIError as e:
         logging.error(f"获取代币价格时发生价格API错误: {e}", exc_info=False)
         return {
             "token_address": token_address,
             "error": f"获取代币价格信息失败 (API Error): {str(e)}",
             "status": "api_error"
         }
    except Exception as e:
        logging.error(f"获取代币价格历史时发生意外错误: {e}", exc_info=True)
        return {
            "token_address": token_address,
            "error": f"获取代币价格历史失败: {str(e)}",
            "status": "error"
        }


# --- DeFi Dashboard & Comparison ---
@mcp.tool()
async def defi_dashboard(protocol_address: Optional[str] = None, account_address: Optional[str] = None) -> Dict[str, Any]:
    """生成DeFi综合仪表盘 (协议概览或用户投资组合 - 用户部分简化)"""

    if not protocol_address and not account_address:
        raise ValueError("必须提供 'protocol_address' 或 'account_address' 参数之一")

    start_time = datetime.now().isoformat()
    dashboard_type = "unknown"
    if protocol_address and account_address: dashboard_type = "comprehensive"
    elif protocol_address: dashboard_type = "protocol"
    elif account_address: dashboard_type = "account"

    logging.info(f"生成 DeFi 仪表盘 ({dashboard_type}): Protocol={protocol_address}, Account={account_address}")

    dashboard_data = {
        "dashboard_type": dashboard_type,
        "protocol_address": protocol_address,
        "account_address": account_address,
        "generated_at": start_time,
    }
    tasks = {}

    # Common Task: Network Status
    tasks["network_status"] = asyncio.create_task(get_latest_sealed_block())

    # Protocol specific tasks
    if protocol_address:
        tasks["protocol_info"] = asyncio.create_task(query_defi_protocol_info(protocol_address))
        tasks["liquidity_pools"] = asyncio.create_task(get_liquidity_pools(protocol_address, limit=5)) # Limit for dashboard
        # Optionally add token price if protocol has one
        # Need to get token address from protocol_info first, then call get_token_price_history

    # Account specific tasks (Simplified)
    if account_address:
        tasks["portfolio"] = asyncio.create_task(analyze_defi_portfolio(account_address))

    # Gather results
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    task_keys = list(tasks.keys())

    # Populate dashboard
    for i, key in enumerate(task_keys):
         if isinstance(results[i], Exception):
             dashboard_data[key] = {"error": str(results[i])}
             logging.error(f"生成DeFi仪表盘时任务 '{key}' 失败: {results[i]}")
         else:
             dashboard_data[key] = results[i]

    # Rename network status block
    if "network_status" in dashboard_data and isinstance(dashboard_data["network_status"], dict) and "error" not in dashboard_data["network_status"]:
         dashboard_data["network_status"] = {
             "latest_sealed_block": dashboard_data["network_status"],
             "network": "mainnet" if "mainnet" in FLOW_API_BASE_URL else "testnet"
         }

    # Add derived data for comprehensive dashboard (very basic)
    if dashboard_type == "comprehensive":
         # Check if user portfolio data is available and if protocol info is available
         portfolio_data = dashboard_data.get("portfolio", {})
         protocol_info_data = dashboard_data.get("protocol_info", {})
         if isinstance(portfolio_data, dict) and "error" not in portfolio_data and \
            isinstance(protocol_info_data, dict) and "error" not in protocol_info_data:
             # This section needs logic to see if the user's simplified portfolio
             # interacts with the specified protocol. Extremely hard without real portfolio data.
             dashboard_data["user_interaction_summary"] = "用户与此协议的交互信息需要更详细的投资组合分析功能。"


    # Check for overall errors
    if any(isinstance(res, Exception) for res in results):
        dashboard_data["status"] = "partial_error"
        dashboard_data["error_summary"] = "一个或多个数据获取任务失败。"
    elif dashboard_type == "account" and dashboard_data.get("portfolio", {}).get("status") == "simplified":
         dashboard_data["status"] = "simplified_account_view"
    else:
         dashboard_data["status"] = "success"


    dashboard_data["data_source_summary"] = "Data sourced from Flow HTTP API, QuickNode Flowscan API (if configured), CoinGecko API, and Cadence scripts (where applicable)."
    return dashboard_data

@mcp.tool()
async def compare_defi_protocols(protocol_addresses: List[str]) -> Dict[str, Any]:
    """比较多个DeFi协议的性能和特点 (依赖 query_defi_protocol_info)"""
    if not protocol_addresses or not isinstance(protocol_addresses, list) or len(protocol_addresses) < 2:
        raise ValueError("必须提供至少两个有效的协议地址进行比较")
    if len(protocol_addresses) > 5:
        logging.warning("比较超过5个协议可能会很慢或达到速率限制。")
        protocol_addresses = protocol_addresses[:5] # Limit comparison

    logging.info(f"比较DeFi协议: {', '.join(protocol_addresses)}")

    tasks = [asyncio.create_task(query_defi_protocol_info(addr)) for addr in protocol_addresses]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    comparison_data = {
        "protocols_compared": protocol_addresses,
        "comparison_results": [],
        "comparison_chart_data": {
            "labels": [],
            "tvl_usd_values": [],
            "volume_24h_usd_values": [],
            "users_values": [],
        },
        "rankings": {},
        "summary": "",
        "errors": []
    }
    valid_protocols = []

    # Process results
    for i, res in enumerate(results):
        addr = protocol_addresses[i]
        if isinstance(res, Exception):
            comparison_data["errors"].append({"address": addr, "error": str(res)})
            comparison_data["comparison_results"].append({"address": addr, "error": str(res)})
        elif isinstance(res, dict) and "error" in res:
             comparison_data["errors"].append({"address": addr, "error": res["error"]})
             comparison_data["comparison_results"].append(res) # Include error details
        elif isinstance(res, dict):
             comparison_data["comparison_results"].append(res)
             valid_protocols.append(res) # Add to list for ranking/charting
        else:
             # Unexpected result format
             err_msg = f"Unexpected result format for {addr}: {type(res)}"
             comparison_data["errors"].append({"address": addr, "error": err_msg})
             comparison_data["comparison_results"].append({"address": addr, "error": err_msg})


    # Generate comparisons only if there are valid results
    if valid_protocols:
        try:
            chart_data = comparison_data["comparison_chart_data"]
            for proto in valid_protocols:
                name = proto.get("name", proto.get("address", "未知"))
                chart_data["labels"].append(name)
                # Safely convert numeric strings to float, defaulting to 0 on error
                try: tvl = float(proto.get("tvl_usd", "0").replace(",", ""))
                except (ValueError, TypeError): tvl = 0.0
                try: vol = float(proto.get("volume_24h_usd", "0").replace(",", ""))
                except (ValueError, TypeError): vol = 0.0
                try: users = int(proto.get("unique_users", 0))
                except (ValueError, TypeError): users = 0

                chart_data["tvl_usd_values"].append(tvl)
                chart_data["volume_24h_usd_values"].append(vol)
                chart_data["users_values"].append(users)

            # Generate Rankings (Handle potential missing data gracefully)
            def safe_get_float(data, key, default=0.0):
                try: return float(str(data.get(key, default) or default).replace(",", ""))
                except (ValueError, TypeError): return default
            def safe_get_int(data, key, default=0):
                 try: return int(data.get(key, default) or default)
                 except (ValueError, TypeError): return default

            comparison_data["rankings"] = {
                "by_tvl_usd": sorted(
                    valid_protocols,
                    key=lambda p: safe_get_float(p, "tvl_usd"), reverse=True),
                "by_volume_24h_usd": sorted(
                    valid_protocols,
                    key=lambda p: safe_get_float(p, "volume_24h_usd"), reverse=True),
                "by_users": sorted(
                    valid_protocols,
                    key=lambda p: safe_get_int(p, "unique_users"), reverse=True)
            }
            # Generate summary using the helper function (needs the helper definition from original code)
            comparison_data["summary"] = generate_protocol_comparison_summary(valid_protocols) # Ensure this helper is defined

        except Exception as analysis_err:
             logging.error(f"比较协议时分析数据出错: {analysis_err}", exc_info=True)
             comparison_data["analysis_error"] = f"比较分析时出错: {analysis_err}"
    else:
        comparison_data["summary"] = "没有足够的数据进行比较。"


    return comparison_data


# --- Helper Definition for Comparison Summary (Copied from original) ---
def generate_protocol_comparison_summary(protocols: List[Dict[str, Any]]) -> str:
    """生成协议比较的文字总结"""
    if not protocols or len(protocols) < 1: # Changed to < 1, as even 1 protocol can be described
        return "提供的协议数据不足以生成总结。"

    # Helper to safely get float value
    def safe_get_float(data, key, default=0.0):
        try: return float(str(data.get(key, default) or default).replace(",", ""))
        except (ValueError, TypeError): return default
    # Helper to safely get int value
    def safe_get_int(data, key, default=0):
         try: return int(data.get(key, default) or default)
         except (ValueError, TypeError): return default

    # Sort by TVL
    sorted_by_tvl = sorted(protocols, key=lambda x: safe_get_float(x, "tvl_usd"), reverse=True)
    top_protocol_tvl = sorted_by_tvl[0]
    top_name_tvl = top_protocol_tvl.get("name", "未知协议")
    top_tvl_value = top_protocol_tvl.get("tvl_usd", "未知")

    # Sort by Volume
    sorted_by_volume = sorted(protocols, key=lambda x: safe_get_float(x, "volume_24h_usd"), reverse=True)
    top_protocol_vol = sorted_by_volume[0]
    top_name_vol = top_protocol_vol.get("name", "未知协议")
    top_vol_value = top_protocol_vol.get("volume_24h_usd", "未知")

    # Sort by Users
    sorted_by_users = sorted(protocols, key=lambda x: safe_get_int(x, "unique_users"), reverse=True)
    top_protocol_users = sorted_by_users[0]
    top_name_users = top_protocol_users.get("name", "未知协议")
    top_users_value = top_protocol_users.get("unique_users", "未知")

    # Generate Summary
    num_protocols = len(protocols)
    summary = f"在比较的 {num_protocols} 个协议中:\n"
    summary += f"- **锁仓价值 (TVL):** {top_name_tvl} 领先，TVL 约为 ${top_tvl_value}。\n"

    if num_protocols > 1:
        if top_name_vol == top_name_tvl:
            summary += f"- **日交易量:** {top_name_vol} 同时领先，约为 ${top_vol_value}。\n"
        else:
            summary += f"- **日交易量:** {top_name_vol} 领先，约为 ${top_vol_value}。\n"

        if top_name_users == top_name_tvl and top_name_users == top_name_vol:
            summary += f"- **用户数:** {top_name_users} 在 TVL、交易量和用户数上均领先，用户数约为 {top_users_value}。\n"
        elif top_name_users == top_name_tvl:
             summary += f"- **用户数:** {top_name_users} 同时在用户数上领先，约为 {top_users_value}。\n"
        elif top_name_users == top_name_vol:
             summary += f"- **用户数:** {top_name_users} 同时在用户数上领先，约为 {top_users_value}。\n"
        else:
            summary += f"- **用户数:** {top_name_users} 领先，约为 {top_users_value}。\n"
    else: # Only one protocol
         summary += f"- **日交易量:** 约为 ${top_vol_value}。\n"
         summary += f"- **用户数:** 约为 {top_users_value}。\n"


    # Add a concluding remark
    if num_protocols > 1:
        leaders = {top_name_tvl, top_name_vol, top_name_users}
        if len(leaders) == 1:
            summary += f"\n**总结:** {top_name_tvl} 在所有关键指标上均表现突出。"
        elif len(leaders) == 2:
             summary += f"\n**总结:** 市场由 {top_name_tvl} 和 {top_name_vol if top_name_vol != top_name_tvl else top_name_users} 等少数协议主导。"
        else:
             summary += f"\n**总结:** 各协议在不同指标上各有优势，市场呈现多元化竞争格局。"
    else:
         summary += f"\n**总结:** {top_name_tvl} 是唯一被分析的协议。"


    # Append source info if available
    sources = list(set(p.get("source", "未知来源") for p in protocols))
    summary += f"\n(数据来源: {', '.join(sources)})"

    return summary.strip()


# --- Main Execution ---
if __name__ == "__main__":
    # Check for necessary environment variables
    if not QUICKNODE_FLOW_API_URL:
        logging.warning("环境变量 QUICKNODE_FLOW_API_URL 未设置。依赖 QuickNode API 的功能将受限。")
    if not os.environ.get("COINGECKO_API_KEY"): # Example if CoinGecko Pro is needed
         pass # Using public API for now
        # logging.warning("环境变量 COINGECKO_API_KEY 未设置。价格查询可能受速率限制。")

    # 运行MCP服务器
    print("FlowMind MCP Server is starting...")
    mcp.run(transport="stdio")
    print("FlowMind MCP Server stopped.")