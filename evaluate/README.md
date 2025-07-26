# 大模型自动化评估系统

本系统用于对基于LoRA微调的甄嬛角色大模型输出结果进行自动化评估，提供多样化的评估指标，可直接用于验证模型效果和指导模型迭代。

## 📁 项目结构

```
evaluate/
├── evaluator.py          # 核心评估模块
├── example_usage.py      # 使用示例
├── requirements.txt      # 评估模块依赖
└── README.md            # 本说明文档
```

## 🚀 功能特性

1. **多样化评估指标**：提供BLEU、ROUGE、语义相似度、连贯性、多样性等多种评估指标
2. **多模型对比**：支持同时评估多个模型并生成对比报告
3. **详细指标解释**：每个评估指标都有详细解释，帮助理解模型表现
4. **易于使用**：提供命令行接口和Python API两种使用方式
5. **报告生成**：自动生成详细的评估报告，便于分析和分享

## 📦 安装依赖

在使用评估系统前，请安装所需依赖：

```bash
# 进入评估目录
cd evaluate

# 安装评估模块依赖
pip install -r requirements.txt
```

注意：评估模块依赖项目根目录下的requirements.txt中的核心依赖，确保已安装项目基础依赖。

## 📊 评估指标详解

### 1. BLEU (Bilingual Evaluation Understudy)
- **用途**：主要用于机器翻译评估，通过比较生成文本和参考文本中的n-gram重叠来评估质量
- **范围**：0-1，值越高表示质量越好
- **特点**：侧重于词汇级别的匹配准确度

### 2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **包含指标**：
  - ROUGE-1：基于单词级别的重叠
  - ROUGE-2：基于二元组级别的重叠
  - ROUGE-L：基于最长公共子序列
- **范围**：0-1，值越高表示质量越好
- **特点**：主要用于文本摘要评估，侧重于召回率

### 3. 语义相似度 (Semantic Similarity)
- **原理**：基于TF-IDF向量化文本，然后计算向量间的余弦相似度
- **范围**：0-1，值越高表示语义越相似
- **特点**：衡量生成文本与参考文本的语义接近程度

### 4. 连贯性 (Coherence)
- **原理**：将文本分割为句子，计算相邻句子间的相似度平均值
- **范围**：0-1，值越高表示文本连贯性越好
- **特点**：评估文本内部的逻辑连贯程度

### 5. 多样性 (Diversity)
- **原理**：通过计算多个生成文本之间的平均相似度来衡量，多样性 = 1 - 平均相似度
- **范围**：0-1，值越高表示多样性越好
- **特点**：衡量模型生成不同响应的能力，避免重复和单调

## 🛠️ 使用方法

### 方法一：命令行使用

```bash
# 评估单个模型
python evaluator.py --models huanhuan_fast --data ../data/test_data.json

# 评估多个模型并生成报告
python evaluator.py --models huanhuan_fast huanhuan_full --data ../data/test_data.json --output report.md

# 限制评估样本数量（适用于快速测试）
python evaluator.py --models huanhuan_fast --data ../data/test_data.json --samples 10
```

### 方法二：Python API使用

```python
from evaluate.evaluator import ModelEvaluator

# 初始化评估器
evaluator = ModelEvaluator()

# 评估模型
results = evaluator.compare_models(
    model_names=["huanhuan_fast", "huanhuan_full"],
    test_data_path="../data/test_data.json"
)

# 生成报告
report = evaluator.generate_report(results, "evaluation_report.md")
print(report)
```

### 测试数据格式

测试数据应为JSON或JSONL格式，包含以下字段：

```json
[
  {
    "prompt": "用户输入的提示词",
    "references": ["参考答案1", "参考答案2"]
  }
]
```

示例：
```json
[
  {
    "prompt": "你是谁？",
    "references": ["我是甄嬛。", "臣妾是甄嬛。"]
  },
  {
    "prompt": "你在哪？",
    "references": ["我在宫中。", "臣妾在紫禁城。"]
  }
]
```

## 📈 输出报告示例

评估完成后会生成详细报告，包含：

1. **模型性能对比表**：直观展示各模型在各项指标上的得分
2. **评估指标说明**：详细解释每个指标的含义和用途
3. **各模型详细结果**：展示每个模型的具体得分和分析

## ⚙️ 配置选项

### 环境变量

- `OLLAMA_HOST`: Ollama服务地址，默认为 `http://localhost:11434`

### 参数说明

- `--models`: 要评估的模型名称列表
- `--data`: 测试数据路径
- `--output`: 评估报告输出路径
- `--samples`: 采样数量，用于限制评估的样本数

## 🧪 使用示例

运行示例脚本了解如何使用评估系统：

```bash
python example_usage.py
```

该脚本会：
1. 创建示例测试数据
2. 评估指定模型
3. 生成评估报告

## 📝 注意事项

1. 确保Ollama服务正在运行且模型已正确部署
2. 测试数据格式需符合要求
3. 评估过程可能需要一定时间，取决于数据量和模型响应速度
4. 部分指标（如BLEU）在生成文本较短时可能不够准确
5. 评估结果仅供参考，实际效果还需结合人工评估

## 🤝 贡献

欢迎提交Issue和Pull Request来改进评估系统。