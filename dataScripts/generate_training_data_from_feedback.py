#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从用户反馈生成高质量训练数据

该工具用于从用户反馈中提取高质量的对话样本，用于模型微调
"""

import json
import argparse
from pathlib import Path
from application.feedback_handler import FeedbackHandler
from loguru import logger


class FeedbackTrainingDataGenerator:
    """
    从反馈数据生成训练数据的工具类
    """
    
    def __init__(self, feedback_file=None, output_dir=None):
        """
        初始化生成器
        
        Args:
            feedback_file: 反馈数据文件路径
            output_dir: 输出目录路径
        """
        self.feedback_handler = FeedbackHandler(feedback_file)
        self.output_dir = Path(output_dir) if output_dir else Path("data/training_samples")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_high_quality_samples(self, min_rating=4):
        """
        提取高质量反馈样本
        
        Args:
            min_rating: 最低评分阈值
            
        Returns:
            list: 高质量样本列表
        """
        logger.info(f"开始提取评分 >= {min_rating} 的高质量样本")
        
        # 获取所有反馈数据
        feedback_list = self.feedback_handler.load_feedback()
        logger.info(f"总共 {len(feedback_list)} 条反馈数据")
        
        # 筛选高质量样本
        high_quality_samples = []
        for feedback in feedback_list:
            rating = feedback.get('rating', 0)
            user_input = feedback.get('user_input', '').strip()
            model_response = feedback.get('model_response', '').strip()
            
            # 检查必要字段
            if rating >= min_rating and user_input and model_response:
                sample = {
                    'instruction': user_input,
                    'input': '',
                    'output': model_response,
                    'rating': rating,
                    'model_name': feedback.get('model_name', 'unknown'),
                    'feedback_id': feedback.get('id', ''),
                    'timestamp': feedback.get('timestamp', '')
                }
                high_quality_samples.append(sample)
        
        logger.info(f"提取到 {len(high_quality_samples)} 条高质量样本")
        return high_quality_samples
    
    def save_as_json(self, samples, filename="high_quality_samples.json"):
        """
        保存为JSON格式
        
        Args:
            samples: 样本列表
            filename: 文件名
        """
        filepath = self.output_dir / filename
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
            logger.info(f"高质量样本已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存JSON文件失败: {e}")
    
    def save_as_jsonl(self, samples, filename="high_quality_samples.jsonl"):
        """
        保存为JSONL格式（每行一个JSON对象）
        
        Args:
            samples: 样本列表
            filename: 文件名
        """
        filepath = self.output_dir / filename
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"高质量样本已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存JSONL文件失败: {e}")
    
    def save_as_alpaca_format(self, samples, filename="alpaca_format_samples.json"):
        """
        保存为Alpaca格式
        
        Args:
            samples: 样本列表
            filename: 文件名
        """
        alpaca_format_samples = []
        for sample in samples:
            alpaca_sample = {
                "instruction": sample['instruction'],
                "input": sample['input'],
                "output": sample['output']
            }
            alpaca_format_samples.append(alpaca_sample)
        
        filepath = self.output_dir / filename
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(alpaca_format_samples, f, ensure_ascii=False, indent=2)
            logger.info(f"Alpaca格式样本已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存Alpaca格式文件失败: {e}")
    
    def generate_training_data(self, min_rating=4, formats=['json', 'jsonl', 'alpaca']):
        """
        生成训练数据
        
        Args:
            min_rating: 最低评分阈值
            formats: 输出格式列表 ('json', 'jsonl', 'alpaca')
        """
        logger.info("开始生成训练数据...")
        
        # 提取高质量样本
        high_quality_samples = self.extract_high_quality_samples(min_rating)
        
        if not high_quality_samples:
            logger.warning("未找到高质量样本")
            return
        
        # 根据指定格式保存
        if 'json' in formats:
            self.save_as_json(high_quality_samples)
        
        if 'jsonl' in formats:
            self.save_as_jsonl(high_quality_samples)
        
        if 'alpaca' in formats:
            self.save_as_alpaca_format(high_quality_samples)
        
        logger.info("训练数据生成完成!")
        
        # 显示统计信息
        self.show_statistics(high_quality_samples)
    
    def show_statistics(self, samples):
        """
        显示样本统计信息
        
        Args:
            samples: 样本列表
        """
        if not samples:
            return
        
        total_samples = len(samples)
        avg_rating = sum(s['rating'] for s in samples) / total_samples
        model_distribution = {}
        
        for sample in samples:
            model_name = sample.get('model_name', 'unknown')
            model_distribution[model_name] = model_distribution.get(model_name, 0) + 1
        
        logger.info("=== 样本统计信息 ===")
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"平均评分: {avg_rating:.2f}")
        logger.info("模型分布:")
        for model_name, count in model_distribution.items():
            logger.info(f"  {model_name}: {count} 条")
    
    def filter_by_model(self, model_name):
        """
        根据模型名称筛选样本
        
        Args:
            model_name: 模型名称
            
        Returns:
            list: 筛选后的样本列表
        """
        feedback_list = self.feedback_handler.load_feedback()
        filtered_samples = []
        
        for feedback in feedback_list:
            if feedback.get('model_name') == model_name:
                user_input = feedback.get('user_input', '').strip()
                model_response = feedback.get('model_response', '').strip()
                
                if user_input and model_response:
                    sample = {
                        'instruction': user_input,
                        'input': '',
                        'output': model_response,
                        'rating': feedback.get('rating', 0),
                        'model_name': feedback.get('model_name', 'unknown'),
                        'feedback_id': feedback.get('id', ''),
                        'timestamp': feedback.get('timestamp', '')
                    }
                    filtered_samples.append(sample)
        
        return filtered_samples


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="从用户反馈生成高质量训练数据")
    parser.add_argument(
        "--feedback-file",
        type=str,
        help="反馈数据文件路径（可选，默认使用系统默认路径）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training_samples",
        help="输出目录路径（默认: data/training_samples）"
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=4,
        help="最低评分阈值（默认: 4）"
    )
    parser.add_argument(
        "--format",
        type=str,
        nargs='+',
        choices=['json', 'jsonl', 'alpaca'],
        default=['json', 'jsonl', 'alpaca'],
        help="输出格式（json, jsonl, alpaca）"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="筛选特定模型的样本（可选）"
    )
    
    args = parser.parse_args()
    
    # 创建生成器实例
    generator = FeedbackTrainingDataGenerator(
        feedback_file=args.feedback_file,
        output_dir=args.output_dir
    )
    
    # 生成训练数据
    if args.model_name:
        logger.info(f"筛选模型 '{args.model_name}' 的样本...")
        samples = generator.filter_by_model(args.model_name)
        if samples:
            logger.info(f"找到 {len(samples)} 条样本")
            
            # 保存筛选后的样本
            if 'json' in args.format:
                generator.save_as_json(samples, f"{args.model_name}_samples.json")
            
            if 'jsonl' in args.format:
                generator.save_as_jsonl(samples, f"{args.model_name}_samples.jsonl")
            
            if 'alpaca' in args.format:
                generator.save_as_alpaca_format(samples, f"{args.model_name}_alpaca_format.json")
        else:
            logger.warning(f"未找到模型 '{args.model_name}' 的样本")
    else:
        generator.generate_training_data(
            min_rating=args.min_rating,
            formats=args.format
        )


if __name__ == "__main__":
    main()