# 甄嬛传角色对话系统优化方案

本文档详细说明了对基于《甄嬛传》角色数据的智能对话系统的优化方案，包括数据集优化、训练参数优化、模型部署优化等方面。

## 1. 数据集优化方案

### 1.1 当前数据集问题分析

1. **数据量不足**：原始数据仅约40条对话，对于大语言模型微调来说严重不足
2. **数据分布单一**：大部分对话集中在特定场景，缺乏多样性
3. **缺乏上下文**：数据均为单轮对话，没有多轮对话历史
4. **语言风格不够丰富**：表达方式较为单一

### 1.2 数据扩充方案

#### 目标数据量
我们将数据集扩充至2000条，其中包括：
- 原始数据增强：1500条
- 多轮对话数据：500条

#### 数据增强技术

1. **同义词替换**
   - 使用同义词词典替换文本中的关键词汇
   - 替换比例：10-15%
   - 保持语义一致性的同时增加表达多样性

2. **语气词插入**
   - 在适当位置插入古典语气词（呢、啊、呀、吧等）
   - 插入比例：10%
   - 增强语言的生动性和古典韵味

3. **模式转换**
   - 使用多种古典表达模式重构句子
   - 例如："臣妾觉得..."、"皇上，...，臣妾斗胆进言"等
   - 增强角色语言风格的一致性

4. **多轮对话生成**
   - 基于单轮对话生成3-5轮的对话历史
   - 增强模型对上下文的理解和回应能力

#### 实现代码
详见 [dataScripts/data_augmentation.py](file:///e:/program/python/ollama_baseline/dataScripts/data_augmentation.py)

### 1.3 数据质量控制

1. **格式验证**：确保所有数据符合[instruction, input, output]格式
2. **内容过滤**：去除不符合角色设定的对话
3. **语言风格检查**：确保语言风格符合古典宫廷特色
4. **重复数据去重**：避免完全相同的数据重复出现

## 2. 训练参数优化方案

### 2.1 模型配置优化

| 参数 | 原值 | 优化值 | 说明 |
|------|------|--------|------|
| [base_model](file://e:\program\python\ollama_baseline\training\huanhuan_train.py#L119-L119) | Qwen/Qwen2.5-0.5B | Qwen/Qwen2.5-0.5B | 保持不变，该模型适合中文任务 |
| [max_length](file://e:\program\python\ollama_baseline\training\huanhuan_train.py#L0-L0) | 128 | 2048 | 大幅增加以支持更长对话 |

### 2.2 训练超参数优化

| 参数 | 原值 | 优化值 | 说明 |
|------|------|--------|------|
| [num_train_epochs](file://e:\program\python\ollama_baseline\training\huanhuan_train.py#L0-L0) | 1 | 5 | 增加训练轮数以充分学习角色特征 |
| [per_device_train_batch_size](file://e:\program\python\ollama_baseline\training\huanhuan_train.py#L0-L0) | 1 | 4 | 增加批次大小提高训练效率 |
| [learning_rate](file://e:\program\python\ollama_baseline\training\huanhuan_train.py#L0-L0) | 1e-3 | 3e-4 | 降低学习率以稳定训练过程 |
| [gradient_accumulation_steps](file://e:\program\python\ollama_baseline\training\huanhuan_train.py#L0-L0) | 2 | 4 | 调整梯度累积以匹配新的批次大小 |
| [warmup_ratio](file://e:\program\python\ollama_baseline\training\huanhuan_train.py#L0-L0) | 0.05 | 0.1 | 增加预热比例以稳定初期训练 |

### 2.3 LoRA参数优化

| 参数 | 原值 | 优化值 | 说明 |
|------|------|--------|------|
| r (rank) | 2 | 8 | 增加LoRA秩以提升表达能力 |
| [lora_alpha](file://e:\program\python\ollama_baseline\training\huanhuan_train.py#L232-L232) | 4 | 16 | 调整缩放因子以匹配新的秩 |
| [target_modules](file://e:\program\python\ollama_baseline\training\huanhuan_train.py#L234-L234) | ["q_proj"] | 多模块 | 增加目标模块以全面适配模型 |
| [lora_dropout](file://e:\program\python\ollama_baseline\training\huanhuan_train.py#L233-L233) | 0.1 | 0.05 | 减少dropout以降低正则化强度 |

### 2.4 数据配置优化

| 参数 | 原值 | 优化值 | 说明 |
|------|------|--------|------|
| [train_file](file://e:\program\python\ollama_baseline\dataScripts\huanhuan_data_prepare.py#L0-L0) | train.jsonl | augmented_train.jsonl | 使用增强后的数据集 |
| [max_source_length](file://e:\program\python\ollama_baseline\training\huanhuan_train.py#L0-L0) | 128 | 512 | 增加输入长度以支持更长上下文 |
| [max_target_length](file://e:\program\python\ollama_baseline\training\huanhuan_train.py#L0-L0) | 128 | 512 | 增加输出长度以支持更长回复 |

### 2.5 生成参数优化

| 参数 | 原值 | 优化值 | 说明 |
|------|------|--------|------|
| [max_new_tokens](file://e:\program\python\ollama_baseline\training\huanhuan_train.py#L0-L0) | 256 | 512 | 增加生成长度以支持更详细回复 |
| [temperature](file://e:\program\python\ollama_baseline\llama.cpp\common\common.h#L136-L136) | 0.8 | 0.7 | 降低温度以提高一致性 |
| [top_p](file://e:\program\python\ollama_baseline\llama.cpp\common\common.h#L136-L136) | 0.9 | 0.92 | 调整top-p以平衡多样性 |
| [top_k](file://e:\program\python\ollama_baseline\llama.cpp\common\common.h#L135-L135) | 50 | 50 | 保持top-k不变 |
| repetition_penalty | 1.05 | 1.1 | 增加重复惩罚以减少重复内容 |

## 3. 模型部署优化 (Modelfile.huanhuan_v1)

### 3.1 参数优化

```dockerfile
# 优化点1: 调整temperature为0.7，平衡创造性和一致性
PARAMETER temperature 0.7

# 优化点2: 调整top_p为0.92，适度增加词汇多样性
PARAMETER top_p 0.92

# 优化点3: 增加top_k到50，提供更多候选词汇
PARAMETER top_k 50

# 优化点4: 增加repeat_penalty到1.1，更好地防止重复
PARAMETER repeat_penalty 1.1

# 优化点5: 增加num_ctx到2048，支持更长上下文
PARAMETER num_ctx 2048

# 优化点6: 增加num_predict到768，支持更长回复
PARAMETER num_predict 768

# 优化点7: 添加mirostat参数，提高生成质量
PARAMETER mirostat 0
```

### 3.2 系统提示优化

优化后的系统提示增加了以下要点：
1. "回复应适度详细，展现你的博学和才情"
2. "遇到敏感话题时，回答要委婉得体，体现你的智慧"

这些补充更好地指导模型生成符合甄嬛角色特征的回复。

### 3.3 优化理由

1. **temperature调整**：从0.8降到0.7，降低随机性，提高回复一致性
2. **top_p调整**：从0.9增加到0.92，适度增加词汇多样性
3. **repeat_penalty调整**：从1.05增加到1.1，更好地防止重复生成
4. **num_predict增加**：从512增加到768，支持更长的回复内容
5. **系统提示增强**：提供更多细节指导，使角色表现更加丰富

## 4. 评估指标体系优化

### 4.1 新增评估指标

为了建立更完善的自动评估指标体系，我们在原有的BLEU、ROUGE、语义相似度、连贯性和多样性等指标基础上，新增了以下两个关键指标：

1. **角色一致性 (Character Consistency)**
   - 评估生成文本是否符合特定角色的语言风格和特征
   - 通过检测文本中是否包含特定角色关键词来评估
   - 对于甄嬛角色，检测如"臣妾"、"皇上"、"娘娘"等关键词的出现频率

2. **流畅度 (Fluency)**
   - 评估生成文本的语言流畅程度
   - 通过分析文本的长度、句子结构复杂度等来评估
   - 理想的句子长度通常在10-25个词之间

### 4.2 ROUGE指标优化

我们对ROUGE指标的计算进行了优化：
- 引入专业的ROUGE库进行计算，提高准确性
- 保留原有的计算方法作为备用方案，确保兼容性

### 4.3 评估报告增强

评估报告现在包含更详细的指标说明和更全面的性能对比：
- 增加角色一致性和流畅度的评估结果
- 提供更详细的指标解释，帮助理解各项指标的含义
- 优化报告格式，使其更易于阅读和比较

### 4.4 多模型对比功能

增强的评估系统支持更方便的多模型对比功能：
- 可以同时评估多个模型并生成对比报告
- 支持指定采样数量，便于快速测试
- 提供详细的个体结果和平均分数

## 5. 用户反馈机制

### 5.1 反馈机制设计

为了实现模型的持续优化，我们引入了用户反馈机制，让用户能够对模型的回复进行评价。该机制包括以下几个核心组件：

1. **反馈收集界面**：在Web应用中为每个模型回复提供评分和评论功能
2. **反馈存储系统**：将用户反馈保存到JSON文件中，便于后续分析
3. **反馈分析面板**：提供可视化界面展示反馈统计数据
4. **训练数据导出**：将高质量反馈导出为训练数据，用于模型微调

### 5.2 反馈收集实现

在[application/huanhuan_web.py](file:///e:/program/python/ollama_baseline/application/huanhuan_web.py)中，我们为每个模型回复添加了反馈收集功能：

- 用户可以对每个回复进行1-5星评分
- 用户可以添加文字评论，提供详细反馈
- 反馈数据包含会话ID、模型名称、用户输入、模型回复、评分和评论

### 5.3 反馈处理模块

我们创建了[application/feedback_handler.py](file:///e:/program/python/ollama_baseline/application/feedback_handler.py)模块来处理反馈数据：

1. **保存反馈**：将用户反馈保存到[data/user_feedback.json](file:///e:/program/python/ollama_baseline/data/user_feedback.json)文件
2. **加载反馈**：从文件中读取所有反馈数据
3. **统计分析**：计算总体反馈统计、按模型分组统计等
4. **模式分析**：分析反馈分布和趋势
5. **训练数据导出**：将高质量反馈（4-5星）导出为训练数据

### 5.4 反馈分析界面

在Web应用中添加了专门的反馈分析页面：

1. **总体统计**：显示总反馈数、正面反馈率、平均评分等关键指标
2. **模型对比**：按模型展示反馈统计数据，便于比较不同模型的表现
3. **最近反馈**：展示最近的用户反馈，便于及时了解用户意见

### 5.5 训练数据导出

提供了训练数据导出功能：

1. **筛选高质量反馈**：只导出4星及以上的正面反馈
2. **格式化输出**：将反馈数据转换为训练数据格式（instruction, input, output）
3. **文件导出**：保存为JSONL格式文件，便于后续训练使用

### 5.6 持续优化流程

基于用户反馈的持续优化流程如下：

1. 用户在对话过程中对模型回复进行评分和评论
2. 系统收集并存储反馈数据
3. 定期分析反馈数据，识别模型的优缺点
4. 将高质量反馈导出为训练数据
5. 使用导出的训练数据对模型进行微调
6. 部署优化后的模型，形成闭环优化流程

## 6. 使用方法

### 6.1 数据增强

```bash
python dataScripts/data_augmentation.py
```

### 6.2 模型训练

```bash
python training/huanhuan_train.py --config training/huanhuan_config_optimized.yaml
```

### 6.3 模型部署

```bash
ollama create huanhuan-qwen-optimized -f deployment/Modelfile.huanhuan_v1
ollama run huanhuan-qwen-optimized
```

### 6.4 模型评估

```bash
# 评估单个模型
python evaluate/evaluator.py --models huanhuan-qwen-optimized --data data/test_data.json

# 比较多个模型
python evaluate/evaluator.py --models huanhuan_fast huanhuan-qwen-optimized --data data/test_data.json --output evaluation_report.md
```

### 6.5 用户反馈收集与分析

```bash
# 运行Web应用（包含反馈功能）
streamlit run application/huanhuan_web.py

# 导出训练数据
# 在Web应用的"训练数据导出"页面点击"导出训练数据"按钮
```

## 7. 预期效果

通过以上优化措施，预期可以实现以下改进：

1. **数据质量提升**：从40条扩充到2000条，大幅提升训练数据量
2. **模型性能提升**：通过优化的LoRA参数和训练策略，提高角色一致性
3. **生成质量提升**：通过优化的生成参数和系统提示，提高回复质量和多样性
4. **部署性能提升**：通过优化的Modelfile参数，提高推理效率和输出质量
5. **评估体系完善**：通过新增的评估指标，更全面地评估模型性能
6. **持续优化能力**：通过用户反馈机制，实现模型的持续优化和迭代

## 8. 后续优化建议

1. **持续数据扩充**：收集更多《甄嬛传》相关对话数据
2. **多角色训练**：扩展到其他角色，构建完整的人物关系网络
3. **评估体系完善**：建立更完善的自动评估指标体系
4. **用户反馈机制**：引入用户反馈以持续优化模型表现