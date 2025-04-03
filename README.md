# FlowMind

## 场景描述
FlowMind 是一个用于管理 Flow 区块链资产的工具集。它可以帮助用户查询 Flow 区块链上的账户信息、交易历史和最新区块信息。该工具集适用于金融资产管理平台，用户可以通过平台查看他们的账户余额、交易历史，并获取最新的区块信息。

## 功能说明

### 主要功能
1. **账户信息查询**
   - 使用 `get_flow_account_info` 函数获取用户账户的基本信息，包括余额、密钥数量和合约数量。

2. **交易历史查询**
   - 使用 `get_account_transactions` 函数获取用户账户的交易历史，显示交易ID、日期、类型和状态。

3. **最新区块信息**
   - 使用 `get_latest_sealed_block` 函数获取 Flow 区块链上最新已密封区块的信息，如区块ID、高度和时间戳。

### 程序说明
- **异步请求处理**：通过 `make_async_flow_api_request` 函数处理所有与 Flow API 的异步请求，确保请求的可靠性和错误处理。
- **工具函数**：程序中定义了一系列工具函数，使用 `@mcp.tool()` 装饰器注册，方便在 MCP 环境中调用。

## 使用方法
1. 克隆此仓库到本地：
   ```bash
   git clone <repository-url>
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行程序：
   ```bash
   python flowmind.py
   ```

## 贡献
欢迎贡献代码！请提交 Pull Request 或报告 Issue。

## 许可证
本项目采用 MIT 许可证。

## MCP 配置

以下是 FlowMind 项目的 MCP 配置示例：

```json
{
  "mcpServers": {
    "hello": {
      "command": "bash",
      "args": [
        "-c",
        " uv --directory /Volumes/extdisk/project/ run flowmind.py"
      ]
    }
  },
  "globalShortcut": ""
}
```
