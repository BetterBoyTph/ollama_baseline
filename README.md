# 甄嬛传角色对话系统 (Ollama Baseline)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue" alt="Python 3.13">
  <img src="https://img.shields.io/badge/PyTorch-2.x-red" alt="PyTorch 2.x">
  <img src="https://img.shields.io/badge/Transformers-4.30%2B-brightgreen" alt="Transformers 4.30+">
  <img src="https://img.shields.io/badge/Streamlit-1.20%2B-orange" alt="Streamlit 1.20+">
</p>

## 🎭 项目简介

这是一个基于《甄嬛传》角色数据的智能对话系统，使用 LoRA 微调技术训练甄嬛角色模型，支持多种交互方式。用户可以与甄嬛进行对话，体验宫廷生活、诗词歌赋等传统文化内容。

## 📁 项目结构

```
ollama_baseline/
├── application/          # Web应用界面
│   └── huanhuan_web.py  # Streamlit对话界面
├── dataScripts/         # 数据处理脚本
│   ├── huanhuan_data_prepare.py  # 训练数据预处理
│   └── download_data.py          # 数据集下载
├── deployment/          # 模型部署
│   ├── FAST_DEPLOYMENT_GUIDE.md  # 快速部署指南
│   ├── Modelfile.huanhuan        # Ollama模型文件
│   └── huanhuan_fast_lora.gguf   # LoRA权重文件
├── evaluate/             # 模型评估
│   ├── evaluator.py             # 核心评估模块
│   ├── example_usage.py         # 使用示例
│   ├── requirements.txt         # 评估模块依赖
│   └── README.md               # 评估模块说明
├── mcp_server/          # MCP服务器
│   ├── __init__.py      # 服务器入口
│   └── server.py        # MCP服务器核心逻辑
├── training/            # 模型训练
│   ├── huanhuan_train.py        # 训练脚本
│   ├── huanhuan_config.yaml     # 训练配置
│   ├── huanhuan_config_fast.yaml # 快速训练配置
│   └── logs/                    # 训练日志
├── data/               # 数据目录
├── requirements.txt    # 项目依赖
└── README.md          # 项目说明
```

## 🌟 核心特性

- 📚 **专业知识**: 基于《甄嬛传》电视剧内容训练，具备丰富的宫廷文化知识
- ⚡ **高效微调**: 采用 LoRA 高效微调技术，显著降低训练成本
- 🖥️ **Web界面**: 基于 Streamlit 构建的友好交互界面
- 🔄 **多模态交互**: 支持实时对话、参数调节、聊天历史管理等功能
- 📊 **反馈机制**: 内置用户反馈收集系统，持续优化模型效果
- 🚀 **灵活部署**: 支持 Ollama 和 vLLM 两种部署方式

## 🏗️ 系统架构

```
甄嬛传角色对话系统
├── Web应用层 (Streamlit)
├── 服务层 (MCP Server)
├── 模型层 (Qwen2.5-0.5B + LoRA)
├── 数据层 (训练数据、用户反馈)
└── 部署层 (Ollama/vLLM)
```

### 🎯 模型评估 (evaluate)
- **evaluator.py**: 大模型自动化评估模块
- 支持BLEU、ROUGE、语义相似度等多种评估指标
- 可对比多个模型并生成详细评估报告
- 参考 [评估模块说明](evaluate/README.md) 了解详细使用方法

### 🔌 MCP服务器 (mcp_server)
- **server.py**: MCP (Model Context Protocol) 服务器实现
- 提供与甄嬛模型交互的API接口
- 支持对话、模型信息查询、状态检查等功能

### 🎯 模型训练 (training)
- **huanhuan_train.py**: 甄嬛角色模型训练脚本
- **huanhuan_config.yaml**: 完整训练配置
- **huanhuan_config_fast.yaml**: 快速训练配置
- 基于LoRA技术进行高效微调
- 支持GPU/MPS/CPU多种设备

## 📦 安装依赖

### 方式一：使用 pip 安装

```bash
pip install -r requirements.txt
```

### 方式二：使用 conda 环境（推荐）

```bash
# 1. 创建conda环境（指定Python版本）
conda create -n huanhuan python=3.13

# 2. 激活环境
conda activate huanhuan

# 3. 安装依赖
pip install -r requirements.txt
# 或者优先使用conda安装
conda install pytorch transformers -c pytorch -c huggingface
pip install -r requirements.txt

# 4. 退出环境
conda deactivate
```

### 方式三：使用 uv

```bash
# 1. 安装uv（如果未安装）
pip install uv

# 2. 创建虚拟环境
uv venv huanhuan_env

# 3. 激活环境
source huanhuan_env/bin/activate  # macOS/Linux

# 4. 使用uv安装依赖（比pip快10-100倍）
uv pip install -r requirements.txt
```