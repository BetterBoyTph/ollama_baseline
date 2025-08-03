# 在vLLM部署中实现与Ollama Modelfile.huanhuan相同的参数和系统提示设置

本文档说明如何在vLLM部署中实现与Ollama的Modelfile.huanhuan文件中相同的参数配置和系统提示词设置。

## Ollama Modelfile.huanhuan配置分析

根据Modelfile.huanhuan文件，Ollama配置包含以下关键参数：

1. **基础模型**: `qwen2.5:0.5b`
2. **LoRA适配器**: `./huanhuan_fast_lora.gguf`
3. **采样参数**:
   - temperature: 0.8
   - top_p: 0.9
   - top_k: 40
   - repeat_penalty: 1.05
   - num_ctx: 2048
   - num_predict: 512
4. **系统提示词**: 详细的甄嬛角色设定和回答规范

## vLLM参数映射

要使vLLM部署与Ollama配置保持一致，需要进行以下参数映射：

### 命令行参数映射

| Ollama参数 | vLLM参数 | 说明 |
|------------|----------|------|
| temperature | --temperature | 控制输出随机性 |
| top_p | --top-p | 核采样参数 |
| top_k | --top-k | top-k采样参数 |
| repeat_penalty | --repetition-penalty | 重复惩罚 |
| num_ctx | --max-model-len | 最大上下文长度 |
| num_predict | --max-tokens | 最大生成token数 |

### vLLM启动命令示例

```bash
python -m vllm.entrypoints.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter ./training/models/huanhuan_fast \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-lora \
    --max-lora-rank 32 \
    --temperature 0.8 \
    --top-p 0.9 \
    --top-k 40 \
    --repetition-penalty 1.05 \
    --max-model-len 2048 \
    --max-tokens 512 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8
```

## 系统提示词设置

Ollama中的系统提示词需要在vLLM中通过以下方式实现：

### 方法1: 在API调用中包含系统提示

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {
        "role": "system",
        "content": "你是甄嬛，《甄嬛传》中的女主角。你是大理寺少卿甄远道之女，因选秀入宫，后成为熹贵妃。你聪慧机智，温婉贤淑，知书达理，擅长诗词歌赋。请用甄嬛的语气和风格来回答问题，语言要古典雅致，谦逊有礼，体现出宫廷女子的教养和智慧。\n\n回答时请注意：\n1. 使用\"臣妾\"自称\n2. 语言要典雅，多用\"便是\"、\"倒是\"、\"只是\"等古典用词\n3. 体现出温婉贤淑的性格特点\n4. 可以适当提及宫廷生活和诗词文化\n5. 保持角色的一致性和真实性"
      },
      {
        "role": "user",
        "content": "皇上，您今日怎么这么高兴？"
      }
    ],
    "max_tokens": 512,
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.05
  }'
```

### 方法2: 在Python代码中设置系统提示

```python
from vllm import LLM
from vllm.sampling_params import SamplingParams

# 配置采样参数，与Ollama保持一致
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
    max_lora_rank=32,
    max_model_len=2048,
    gpu_memory_utilization=0.8,
    tensor_parallel_size=1
)

# 添加LoRA适配器
llm.add_lora("./training/training/models/huanhuan_fast")

# Ollama Modelfile.huanhuan中的系统提示词
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

def generate_response(user_prompt):
    # 构造完整的对话消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # 生成响应
    outputs = llm.chat(messages, sampling_params)
    return outputs[0].outputs[0].text

# 示例使用
if __name__ == "__main__":
    response = generate_response("皇上，您今日怎么这么高兴？")
    print(response)
```

## 配置文件方式

还可以创建一个配置文件来管理这些参数，类似于Ollama的Modelfile：

### 创建vLLM配置文件 (vllm_config.yaml)

```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
adapter: ./training/training/models/huanhuan_fast
host: 0.0.0.0
port: 8000
enable_lora: true
max_lora_rank: 32
temperature: 0.8
top_p: 0.9
top_k: 40
repetition_penalty: 1.05
max_tokens: 512
max_model_len: 2048
tensor_parallel_size: 1
gpu_memory_utilization: 0.8
system_prompt: |
  你是甄嬛，《甄嬛传》中的女主角。你是大理寺少卿甄远道之女，
  因选秀入宫，后成为熹贵妃。你聪慧机智，温婉贤淑，知书达理，
  擅长诗词歌赋。请用甄嬛的语气和风格来回答问题，
  语言要古典雅致，谦逊有礼，体现出宫廷女子的教养和智慧。

  回答时请注意：
  1. 使用"臣妾"自称
  2. 语言要典雅，多用"便是"、"倒是"、"只是"等古典用词
  3. 体现出温婉贤淑的性格特点
  4. 可以适当提及宫廷生活和诗词文化
  5. 保持角色的一致性和真实性
```

## 总结

为了在vLLM部署中实现与Ollama Modelfile.huanhuan相同的效果，需要注意以下几点：

1. 确保采样参数完全一致：temperature、top_p、top_k、repeat_penalty等
2. 设置合适的max_model_len和max_tokens参数以匹配num_ctx和num_predict
3. 在每次请求中包含系统提示词，以确保模型行为与Ollama版本一致
4. 正确配置LoRA适配器路径和相关参数
5. 根据硬件环境调整GPU内存使用率和并行设置

通过以上配置，可以在vLLM中实现与Ollama相同的模型行为和输出效果。