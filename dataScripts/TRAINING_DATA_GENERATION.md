# 从用户反馈生成训练数据使用说明

## 概述

本工具用于从用户对模型回复的反馈中提取高质量的对话样本，用于后续的模型微调和优化。通过筛选高评分的对话样本，我们可以获得更符合用户期望的训练数据。

## 工具文件

- `generate_training_data_from_feedback.py` - 主要的生成工具脚本

## 使用方法

### 基本用法

```bash
# 使用默认配置生成训练数据
python dataScripts/generate_training_data_from_feedback.py
```

### 指定参数

```bash
# 指定最低评分阈值
python dataScripts/generate_training_data_from_feedback.py --min-rating 5

# 指定输出目录
python dataScripts/generate_training_data_from_feedback.py --output-dir ./my_training_data

# 指定输出格式（只生成JSONL格式）
python dataScripts/generate_training_data_from_feedback.py --format jsonl

# 组合多个参数
python dataScripts/generate_training_data_from_feedback.py --min-rating 4 --format json alpaca --output-dir ./training_data_v2
```

### 按模型筛选

```bash
# 只提取特定模型的反馈数据
python dataScripts/generate_training_data_from_feedback.py --model-name huanhuan-qwen
```

## 参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| --feedback-file | string | None | 反馈数据文件路径（可选） |
| --output-dir | string | data/training_samples | 输出目录路径 |
| --min-rating | int | 4 | 最低评分阈值（1-5） |
| --format | string array | json jsonl alpaca | 输出格式（可选：json, jsonl, alpaca） |
| --model-name | string | None | 筛选特定模型的样本 |

## 输出格式

### JSON格式
包含所有元数据的完整格式：

```json
[
  {
    "instruction": "用户的问题",
    "input": "",
    "output": "模型的回答",
    "rating": 5,
    "model_name": "huanhuan-qwen",
    "feedback_id": "唯一反馈ID",
    "timestamp": "反馈时间"
  }
]
```

### JSONL格式
每行一个JSON对象，适合大数据处理：

```
{"instruction": "用户的问题1", "input": "", "output": "模型的回答1", "rating": 5, "model_name": "huanhuan-qwen", "feedback_id": "ID1", "timestamp": "时间1"}
{"instruction": "用户的问题2", "input": "", "output": "模型的回答2", "rating": 4, "model_name": "huanhuan-qwen", "feedback_id": "ID2", "timestamp": "时间2"}
```

### Alpaca格式
标准的微调数据格式：

```json
[
  {
    "instruction": "用户的问题",
    "input": "",
    "output": "模型的回答"
  }
]
```

## 使用建议

### 1. 评分阈值选择
- **5分**：只选择最优秀的样本，数据质量最高但数量较少
- **4分及以上**：较好的平衡点，既保证质量又有足够的样本数量
- **3分及以上**：包含更多样本，但可能包含一些质量一般的样本

### 2. 数据清洗
生成的数据建议进行人工审核，确保：
- 回答内容符合角色设定
- 没有敏感或不当内容
- 语言表达自然流畅

### 3. 格式选择
- **JSON**：便于查看和编辑，适合小规模数据
- **JSONL**：适合大规模数据处理和流式处理
- **Alpaca**：标准微调格式，可直接用于大多数微调框架

### 4. 模型特定数据
如果系统中有多个模型，建议按模型分别生成训练数据，这样可以：
- 针对不同模型的特点进行优化
- 避免模型间的风格混淆
- 更精确地评估各模型的表现

## 集成到训练流程

生成的训练数据可以很容易地集成到现有的训练流程中：

1. 定期运行此工具生成新的训练样本
2. 将新样本与原始训练数据合并
3. 使用合并后的数据集进行增量训练
4. 部署更新后的模型并继续收集反馈

## 注意事项

1. 确保反馈数据文件存在且格式正确
2. 输出目录会自动创建，无需手动创建
3. 生成的数据仅包含4分及以上的高评分样本，以保证质量
4. 建议定期运行此工具以获取最新的高质量样本
5. 生成的数据应经过人工审核后再用于模型训练