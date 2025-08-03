# 甄嬛模型 Docker vLLM 部署指南

基于 Qwen2.5-0.5B + LoRA 微调的甄嬛角色模型 Docker vLLM 部署指南

## 📋 前提条件

### 1. 确认训练完成
确保以下文件存在：
```
training/training/models/huanhuan_fast/
├── adapter_config.json
├── adapter_model.safetensors
└── train_results.json
```

### 2. 安装 Docker 和 NVIDIA Container Toolkit
```bash
# Ubuntu安装Docker
sudo apt update
sudo apt install docker.io

# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

# 重启Docker服务
sudo systemctl restart docker
```

### 3. 验证环境
```bash
# 验证Docker安装
docker --version

# 验证NVIDIA Docker支持
docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi
```

## 🐳 Docker vLLM 部署方式

### 1. 准备模型文件
将训练好的LoRA模型文件转换为vLLM兼容格式：
```bash
# 确保模型文件在指定位置
ls training/training/models/huanhuan_fast/

# 如果还没有转换模型，执行转换
python convert_lora_to_gguf.py
```

### 2. 拉取vLLM Docker镜像
```bash
# 拉取vLLM官方镜像
docker pull vllm/vllm-openai:latest
```

### 3. 启动Docker容器
```bash
# 运行vLLM容器 (适用于RTX 3090 24G环境)
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
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --enforce-eager
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

## ⚡ 快速部署方式 (适用于Autodl实例)

### 1. 安装依赖
```bash
# 激活虚拟环境
source /root/autodl-tmp/huanhuan_env/bin/activate

# 安装vLLM
pip install vllm==0.5.4

# 验证安装
python -c "import vllm; print(vllm.__version__)"
```

### 2. 启动vLLM服务
```bash
# 确保在项目根目录
cd /path/to/ollama_baseline

# 启动vLLM服务 (针对RTX 3090 24G配置优化)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter ./training/training/models/huanhuan_fast \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-lora \
    --max-lora-rank 64 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --enforce-eager
```

## ⚙️ 系统配置和并发参数说明

### 并发参数设置说明

根据系统配置 (RTX 3090 24G, 16核心CPU, 120G内存) 动态设置了以下参数：

1. `MAX_CONCURRENT_REQUESTS` 动态计算:
   - 根据CPU物理核心数的一半计算
   - 最小值为1，最大值为8
   - 对于16核心CPU，设置为8

2. `--tensor-parallel-size 1`:
   - 单GPU设置，无需张量并行

3. `--gpu-memory-utilization 0.8`:
   - 限制GPU内存使用率，保留空间给系统和其他进程

4. `--max-model-len 4096`:
   - 设置最大序列长度，平衡性能和内存使用

5. `--enforce-eager`:
   - 强制使用eager模式，避免CUDA图形编译开销

这些参数确保了在RTX 3090 24G环境下能够稳定运行，并提供良好的并发处理能力。

## 🧪 模型推理示例

### Python API推理
```python
import requests
import json

# 配置
VLLM_HOST = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def generate_response(prompt, max_tokens=200, temperature=0.8):
    """生成回复"""
    request_data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40
    }
    
    response = requests.post(
        f"{VLLM_HOST}/v1/completions",
        json=request_data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['text'] if result.get('choices') else ""
    else:
        raise Exception(f"请求失败: {response.status_code} - {response.text}")

# 示例使用
if __name__ == "__main__":
    response = generate_response("你好，甄嬛，今天天气如何？")
    print(response)
```

### 批量推理
```python
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

VLLM_HOST = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 根据CPU核心数动态设置并发参数
cpu_count = psutil.cpu_count(logical=False) or 4
MAX_CONCURRENT_REQUESTS = min(8, max(1, cpu_count // 2))

def generate_response(prompt_data):
    """生成单个回复"""
    prompt, max_tokens, temperature = prompt_data
    request_data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40
    }
    
    try:
        response = requests.post(
            f"{VLLM_HOST}/v1/completions",
            json=request_data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['text'] if result.get('choices') else ""
    except Exception as e:
        return f"错误: {str(e)}"

def batch_generate(prompts_data):
    """批量生成回复"""
    results = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        # 提交所有任务
        future_to_prompt = {
            executor.submit(generate_response, prompt_data): prompt_data 
            for prompt_data in prompts_data
        }
        
        # 收集结果
        for future in as_completed(future_to_prompt):
            prompt_data = future_to_prompt[future]
            try:
                result = future.result()
                results.append((prompt_data[0], result))  # (prompt, response)
            except Exception as e:
                results.append((prompt_data[0], f"错误: {str(e)}"))
    
    return results

# 示例使用
if __name__ == "__main__":
    prompts = [
        ("皇上，您今日怎么这么高兴？", 200, 0.8),
        ("华妃待我如何？", 150, 0.7),
        ("请为我作一首诗", 300, 0.9),
        ("你最喜欢什么花？", 100, 0.7)
    ]
    
    results = batch_generate(prompts)
    for prompt, response in results:
        print(f"问题: {prompt}")
        print(f"回答: {response}\n")
```

## 🛠️ 性能优化建议

### 1. GPU内存优化
```bash
# 设置GPU内存使用比例
--gpu-memory-utilization 0.8

# 设置最大模型长度
--max-model-len 4096

# 设置张量并行大小（多GPU）
--tensor-parallel-size 1
```

### 2. 推理参数优化
```python
request_data = {
    "temperature": 0.8,        # 控制输出随机性
    "top_p": 0.9,              # 核采样
    "top_k": 40,               # top-k采样
    "max_tokens": 512          # 最大生成token数
}
```

### 3. LoRA配置优化
```bash
# 设置最大LoRA rank
--max-lora-rank 64

# 启用LoRA
--enable-lora
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