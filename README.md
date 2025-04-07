# FlowMind

FlowMind 是一个功能强大的 Flow 区块链浏览器和分析工具，提供钱包浏览、NFT 分析和 DeFi 投资组合管理功能。它帮助用户全面了解他们在 Flow 区块链上的资产状况、NFT 收藏品和 DeFi 投资。

## 功能概述

FlowMind 提供三大核心功能模块：

### 1. 钱包浏览功能

- **账户信息查询**：获取账户余额、密钥数量和合约数量
- **交易历史查询**：查看账户的交易记录，包括交易ID、日期、类型和状态
- **最新区块信息**：获取 Flow 区块链上最新已密封区块的信息
- **钱包仪表盘**：生成综合性钱包仪表盘，包含账户信息、交易历史和网络状态
- **账户活动监控**：实时监控账户活动，检测新交易

### 2. NFT 浏览功能

- **NFT 集合信息**：查询 NFT 合约信息，包括名称、总供应量、持有者数量等
- **NFT 元数据查询**：获取单个 NFT 的详细元数据
- **所有权历史追踪**：追踪 NFT 的所有权变更历史
- **NFT 集合统计分析**：分析 NFT 集合的市场表现，包括地板价、交易量和持有者分布
- **NFT 仪表盘**：生成 NFT 集合或单个 NFT 的综合仪表盘

### 3. DeFi 分析功能

- **DeFi 协议信息**：查询 DeFi 协议的基本信息和性能指标
- **流动性池分析**：获取 DeFi 协议的流动性池信息
- **投资组合分析**：分析用户在各 DeFi 协议中的资产分配和收益情况
- **代币价格历史**：查询代币价格历史数据和趋势分析
- **DeFi 仪表盘**：生成 DeFi 协议或用户投资组合的综合仪表盘
- **协议比较**：比较多个 DeFi 协议的性能和特点

## 技术实现

FlowMind 基于以下技术构建：

- **FastMCP 框架**：提供工具函数注册和调用能力
- **Flow HTTP API**：获取区块链基础数据
- **Flowscan API**：获取交易历史和扩展数据
- **异步请求处理**：使用 `httpx` 和 `asyncio` 实现高效的异步请求
- **数据可视化**：提供结构化数据用于仪表盘展示

## 使用方法

### 安装

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
