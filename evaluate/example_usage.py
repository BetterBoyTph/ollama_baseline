#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估系统使用示例
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluate.evaluator import ModelEvaluator


def create_sample_test_data():
    """
    创建示例测试数据
    """
    test_data = [
        {
            "prompt": "你是谁？",
            "references": ["我是甄嬛。", "臣妾是甄嬛。"]
        },
        {
            "prompt": "你在哪？",
            "references": ["我在宫中。", "臣妾在紫禁城。"]
        },
        {
            "prompt": "今天天气如何？",
            "references": ["今日天气甚好。", "天气晴朗明媚。"]
        },
        {
            "prompt": "你对皇上的感情如何？",
            "references": ["皇上对臣妾恩重如山，臣妾自当尽心侍奉。", "臣妾对皇上自是感激不尽。"]
        },
        {
            "prompt": "宫中生活如何？",
            "references": ["宫中生活虽有诸多规矩，但臣妾已习以为常。", "宫中岁月悠长，需得静心度日。"]
        }
    ]
    
    import json
    with open('../data/test_data.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)


def main():
    """
    主函数 - 演示如何使用评估系统
    """
    # 创建示例测试数据
    create_sample_test_data()
    
    # 初始化评估器
    evaluator = ModelEvaluator()
    
    # 定义要评估的模型（根据实际部署的模型名称调整）
    models_to_evaluate = ["huanhuan_fast", "huanhuan-qwen-optimized"]
    
    # 评估模型
    print("开始评估模型...")
    results = evaluator.compare_models(
        model_names=models_to_evaluate,
        test_data_path="../data/test_data.json",
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