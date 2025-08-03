# 使用vLLM部署甄嬛角色模型指南

基于 Qwen2.5-0.5B + LoRA 微调的甄嬛角色模型vLLM部署指南

## 📋 前提条件

### 1. 确认训练完成
确保以下文件存在：
```
training/training/models/huanhuan_fast/
├── adapter_config.json
├── adapter_model.safetensors
└── train_results.json
```

### 2. 安装 vLLM
```bash
# 创建虚拟环境
conda create -n vllm-huanhuan python=3.10 -y
conda activate vllm-huanhuan

# 安装 vLLM (推荐0.5.4版本以确保兼容性)
pip install vllm==0.5.4

# 或者安装最新版本
pip install vllm

# 验证安装
python -c "import vllm; print(vllm.__version__)"
```

## 🐳 Docker部署方式

### 1. 拉取vLLM Docker镜像
```bash
docker pull vllm/vllm-openai:latest
```

### 2. 准备模型文件
将训练好的LoRA模型文件复制到项目目录中：
```bash
# 确保模型文件在指定位置
ls training/training/models/huanhuan_fast/
```

### 3. 启动Docker容器
```bash
# 运行vLLM容器
docker run --gpus all \
    --name huanhuan-vllm \
    -v $(pwd)/training/training/models/huanhuan_fast:/models/huanhuan_fast \
    -v $(pwd)/deployment:/deployment \
    -p 8000:8000 \
    --env NCCL_IGNORE_DISABLED_P2P=1 \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter /models/huanhuan_fast \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-lora \
    --max-lora-rank 64 \
    --tensor-parallel-size 1
```

### 4. 验证部署
```bash
# 测试API
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": "你是谁？",
    "max_tokens": 200
  }'
```

## ⚡ 快速部署方式

### 1. 直接命令行部署
```bash
# 确保在项目根目录
cd /path/to/ollama_baseline

# 启动vLLM服务
python -m vllm.entrypoints.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter ./training/training/models/huanhuan_fast \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-lora \
    --max-lora-rank 64
```

### 2. 使用Python代码部署
创建 [deployment/vllm_server.py](file:///e:/program/python/ollama_baseline/deployment/vllm_server.py) 文件：

```python
from vllm import LLM
from vllm.sampling_params import SamplingParams

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.05,
    max_tokens=512
)

# 初始化模型
llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    enable_lora=True,
    max_lora_rank=64,
    dtype="auto"
)

# 添加LoRA适配器
llm.add_lora("./training/training/models/huanhuan_fast")

# 系统提示词
system_prompt = """你是甄嬛，《甄嬛传》中的女主角。你是大理寺少卿甄远道之女，
因选秀入宫，后成为熹贵妃。你聪慧机智，温婉贤淑，知书达理，
擅长诗词歌赋。请用甄嬛的语气和风格来回答问题，
语言要古典雅致，谦逊有礼，体现出宫廷女子的教养和智慧。

回答时请注意：
1. 使用"臣妾"自称
2. 语言要典雅，多用"便是"、"倒是"、"只是"等古典用词
3. 体现出温婉贤淑的性格特点
4. 可以适当提及宫廷生活和诗词文化
5. 保持角色的一致性和真实性"""

def generate_response(prompt):
    # 构造完整的对话
    full_prompt = f"{system_prompt}\n\n{prompt}"
    
    # 生成响应
    outputs = llm.generate(full_prompt, sampling_params)
    return outputs[0].outputs[0].text

# 示例使用
if __name__ == "__main__":
    response = generate_response("皇上，您今日怎么这么高兴？")
    print(response)
```

运行服务器：
```bash
python deployment/vllm_server.py
```

### 3. OpenAI兼容API部署
```bash
# 启动OpenAI兼容的API服务器
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter ./training/training/models/huanhuan_fast \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-lora \
    --max-lora-rank 64
```

使用API调用：
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": "请介绍一下你自己",
    "max_tokens": 500,
    "temperature": 0.8
  }'
```

## 🧪 模型推理示例

### Python API推理
```python
from vllm import LLM, SamplingParams

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.9,
    max_tokens=200
)

# 加载模型
llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    enable_lora=True,
    max_lora_rank=64
)

# 添加LoRA适配器
llm.add_lora("./training/training/models/huanhuan_fast")

# 生成文本
prompts = [
    "你好，甄嬛，今天天气如何？",
    "请问你是谁？"
]
outputs = llm.generate(prompts, sampling_params)

# 打印输出
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### 批量推理
```python
from vllm import LLM, SamplingParams

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.9,
    max_tokens=300
)

# 加载模型
llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    enable_lora=True,
    max_lora_rank=64,
    tensor_parallel_size=1  # 根据GPU数量调整
)

# 添加LoRA适配器
llm.add_lora("./training/training/models/huanhuan_fast")

# 批量生成
prompts = [
    "皇上，您今日怎么这么高兴？",
    "华妃待我如何？",
    "请为我作一首诗",
    "你最喜欢什么花？"
]

outputs = llm.generate(prompts, sampling_params)
for i, output in enumerate(outputs):
    print(f"问题 {i+1}: {prompts[i]}")
    print(f"回答: {output.outputs[0].text}\n")
```

## 🛠️ 性能优化建议

### 1. GPU内存优化
```bash
# 设置GPU内存使用比例
--gpu-memory-utilization 0.8

# 设置最大模型长度
--max-model-len 4096

# 设置张量并行大小（多GPU）
--tensor-parallel-size 2
```

### 2. 推理参数优化
```python
sampling_params = SamplingParams(
    temperature=0.8,        # 控制输出随机性
    top_p=0.9,              # 核采样
    top_k=40,               # top-k采样
    repetition_penalty=1.05, # 重复惩罚
    max_tokens=512          # 最大生成token数
)
```

### 3. LoRA配置优化
```bash
# 设置最大LoRA rank
--max-lora-rank 64

# 设置最大LoRA数量
--max-loras 4
```

## 🧾 注意事项

1. 确保基础模型 Qwen/Qwen2.5-0.5B-Instruct 已下载或可访问
2. LoRA适配器路径必须正确
3. GPU内存至少需要8GB以上
4. 根据实际硬件配置调整参数
5. 如果使用Docker，请确保已安装NVIDIA Container Toolkit
6. 如遇到CUDA相关错误，可能需要指定正确的CUDA版本

## 📞 API调用示例

### 单次对话
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": "你好，甄嬛",
    "max_tokens": 300,
    "temperature": 0.8
  }'
```

### 聊天接口（如果使用OpenAI兼容API）
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {
        "role": "system",
        "content": "你是甄嬛，《甄嬛传》中的女主角..."
      },
      {
        "role": "user",
        "content": "皇上，您今日怎么这么高兴？"
      }
    ],
    "max_tokens": 300,
    "temperature": 0.8
  }'
```