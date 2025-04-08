# FlowMind

[![FlowMind Demo](https://img.shields.io/badge/Demo-YouTube-red)](https://youtu.be/EHca03gn3Z0?si=IhtJUZpAiebwvvr9)
<iframe width="560" height="315" src="https://www.youtube.com/embed/EHca03gn3Z0?si=RqgOfgbILbSz8bys" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## 场景描述 | Scenario Description

FlowMind 是一个用于管理 Flow 区块链资产的工具集。它可以帮助用户查询 Flow 区块链上的账户信息、交易历史和最新区块信息。该工具集适用于金融资产管理平台，用户可以通过平台查看他们的账户余额、交易历史，并获取最新的区块信息。

FlowMind is a toolkit for managing assets on the Flow blockchain. It helps users query account information, transaction history, and the latest block information on the Flow blockchain. This toolkit is suitable for financial asset management platforms, allowing users to view their account balances, transaction histories, and obtain the latest block information.

## 功能说明 | Features

### 主要功能 | Main Features

1. **账户信息查询 | Account Information Query**
   - 使用 `get_flow_account_info` 函数获取用户账户的基本信息，包括余额、密钥数量和合约数量。
   - Use the `get_flow_account_info` function to get basic information about user accounts, including balance, number of keys, and number of contracts.

2. **交易历史查询 | Transaction History Query**
   - 使用 `get_account_transactions` 函数获取用户账户的交易历史，显示交易ID、日期、类型和状态。
   - Use the `get_account_transactions` function to get the transaction history of user accounts, displaying transaction ID, date, type, and status.

3. **最新区块信息 | Latest Block Information**
   - 使用 `get_latest_sealed_block` 函数获取 Flow 区块链上最新已密封区块的信息，如区块ID、高度和时间戳。
   - Use the `get_latest_sealed_block` function to get information about the latest sealed block on the Flow blockchain, such as block ID, height, and timestamp.

### 程序说明 | Program Description

- **异步请求处理 | Asynchronous Request Processing**：通过 `make_async_flow_api_request` 函数处理所有与 Flow API 的异步请求，确保请求的可靠性和错误处理。
  - Process all asynchronous requests to the Flow API through the `make_async_flow_api_request` function, ensuring request reliability and error handling.
  
- **工具函数 | Tool Functions**：程序中定义了一系列工具函数，使用 `@mcp.tool()` 装饰器注册，方便在 MCP 环境中调用。
  - The program defines a series of tool functions, registered using the `@mcp.tool()` decorator, making them easy to call in the MCP environment.

## 使用方法 | Usage

1. 克隆此仓库到本地 | Clone this repository to your local machine:
   ```bash
   git clone <repository-url>

   安装依赖 | Install dependencies:bash运行3. 运行程序 | Run the program:bash运行贡献 | Contribution欢迎贡献代码！请提交 Pull Request 或报告 Issue。Contributions are welcome! Please submit Pull Requests or report Issues.许可证 | License本项目采用 MIT 许可证。This project is licensed under the MIT License.MCP 配置 | MCP Configuration以下是 FlowMind 项目的 MCP 配置示例：Below is an example MCP configuration for the FlowMind project:json,收起代码