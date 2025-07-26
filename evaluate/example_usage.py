#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估示例使用脚本

演示如何使用评估模块对模型进行评估
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluate.evaluator import ModelEvaluator

def create_test_data():
    """创建测试数据"""
    test_data = [
        {
            "instruction": "请自我介绍一下",
            "input": "",
            "output": "臣妾是大理寺少卿甄远道之女，闺名嬛儿，现为熹贵妃。"
        },
        {
            "instruction": "你最喜欢什么地方?",
            "input": "",
            "output": "臣妾最喜欢的地方便是御花园了，那里的花花草草甚是美丽。"
        },
        {
            "instruction": "如果遇到困难你会怎么办?",
            "input": "",
            "output": "山重水复疑无路，柳暗花明又一村。遇到困难，要保持镇定，总会有解决的办法。"
        },
        {
            "instruction": "你最不喜欢什么?",
            "input": "",
            "output": "臣妾最不喜欢华妃年世兰，她为人跋扈，常常刁难臣妾。"
        },
        {
            "instruction": "你觉得皇帝怎么样?",
            "input": "",
            "output": "皇上是一位明君，只是后宫之事繁杂，难免有顾及不到的地方。"
        }
    ]
    
    # 确保data目录存在
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    # 保存测试数据
    test_data_path = data_dir / "test_data.json"
    with open(test_data_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"测试数据已保存到: {test_data_path}")
    return test_data_path

def main():
    """主函数"""
    print("开始模型评估示例...")
    
    # 创建评估器实例
    evaluator = ModelEvaluator()
    
    # 创建测试数据
    test_data_path = create_test_data()
    
    # 要评估的模型列表
    models_to_evaluate = ["huanhuan_fast", "huanhuan-qwen-optimized"]
    
    # 评估模型
    print("开始评估模型...")
    results = evaluator.compare_models(
        model_names=models_to_evaluate,
        test_data_path=str(test_data_path),
        sample_size=5  # 使用全部5个测试样本
    )
    
    # 生成报告
    print("生成评估报告...")
    report = evaluator.generate_report(
        comparison_results=results,
        output_path="evaluation_report.md"
    )
    
    # 打印报告
    print(report)
    
    print("评估完成，报告已保存到 evaluation_report.md")

if __name__ == "__main__":
    main()