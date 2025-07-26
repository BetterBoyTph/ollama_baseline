#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户反馈处理模块

用于收集、存储和分析用户对模型输出的反馈
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
from loguru import logger


class FeedbackHandler:
    """
    用户反馈处理器
    """
    
    def __init__(self, feedback_file: str = "../data/user_feedback.json"):
        """
        初始化反馈处理器
        
        Args:
            feedback_file: 反馈数据存储文件路径
        """
        self.feedback_file = Path(__file__).parent / feedback_file
        self.ensure_feedback_file_exists()
    
    def ensure_feedback_file_exists(self):
        """
        确保反馈文件存在
        """
        if not self.feedback_file.exists():
            # 创建空的反馈文件
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            logger.info(f"创建反馈文件: {self.feedback_file}")
    
    def save_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """
        保存用户反馈
        
        Args:
            feedback_data: 反馈数据
            
        Returns:
            保存是否成功
        """
        try:
            # 读取现有反馈数据
            feedback_list = self.load_feedback()
            
            # 添加时间戳
            feedback_data['timestamp'] = datetime.now().isoformat()
            
            # 添加到反馈列表
            feedback_list.append(feedback_data)
            
            # 保存到文件
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_list, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存用户反馈成功: {feedback_data.get('session_id', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"保存用户反馈失败: {e}")
            return False
    
    def load_feedback(self) -> List[Dict[str, Any]]:
        """
        加载所有反馈数据
        
        Returns:
            反馈数据列表
        """
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载反馈数据失败: {e}")
            return []
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        获取反馈统计信息
        
        Returns:
            反馈统计信息
        """
        feedback_list = self.load_feedback()
        
        if not feedback_list:
            return {
                'total_feedback': 0,
                'positive_feedback': 0,
                'negative_feedback': 0,
                'positive_rate': 0.0,
                'avg_rating': 0.0
            }
        
        # 统计各类反馈
        positive_count = 0
        negative_count = 0
        total_rating = 0
        
        for feedback in feedback_list:
            rating = feedback.get('rating', 0)
            if rating > 3:  # 4-5星为正面反馈
                positive_count += 1
            elif rating <= 3:  # 1-3星为负面反馈
                negative_count += 1
            
            total_rating += rating
        
        total_count = len(feedback_list)
        positive_rate = positive_count / total_count if total_count > 0 else 0
        avg_rating = total_rating / total_count if total_count > 0 else 0
        
        return {
            'total_feedback': total_count,
            'positive_feedback': positive_count,
            'negative_feedback': negative_count,
            'positive_rate': round(positive_rate, 4),
            'avg_rating': round(avg_rating, 2)
        }
    
    def get_feedback_by_model(self) -> Dict[str, Dict[str, Any]]:
        """
        按模型获取反馈统计
        
        Returns:
            按模型分组的反馈统计
        """
        feedback_list = self.load_feedback()
        
        if not feedback_list:
            return {}
        
        # 按模型分组
        model_feedback = {}
        for feedback in feedback_list:
            model_name = feedback.get('model_name', 'unknown')
            if model_name not in model_feedback:
                model_feedback[model_name] = {
                    'feedbacks': [],
                    'stats': {
                        'total': 0,
                        'positive': 0,
                        'negative': 0,
                        'avg_rating': 0.0
                    }
                }
            model_feedback[model_name]['feedbacks'].append(feedback)
        
        # 计算各模型统计信息
        for model_name, model_data in model_feedback.items():
            feedbacks = model_data['feedbacks']
            total = len(feedbacks)
            positive = sum(1 for f in feedbacks if f.get('rating', 0) > 3)
            negative = sum(1 for f in feedbacks if f.get('rating', 0) <= 3)
            avg_rating = sum(f.get('rating', 0) for f in feedbacks) / total if total > 0 else 0
            
            model_data['stats'] = {
                'total': total,
                'positive': positive,
                'negative': negative,
                'avg_rating': round(avg_rating, 2)
            }
        
        return model_feedback
    
    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的反馈
        
        Args:
            limit: 限制返回的数量
            
        Returns:
            最近的反馈列表
        """
        feedback_list = self.load_feedback()
        
        # 按时间排序
        feedback_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return feedback_list[:limit]
    
    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """
        分析反馈模式
        
        Returns:
            反馈模式分析结果
        """
        feedback_list = self.load_feedback()
        
        if not feedback_list:
            return {}
        
        # 按评分统计
        rating_counts = {}
        for feedback in feedback_list:
            rating = feedback.get('rating', 0)
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
        
        # 按模型统计
        model_stats = self.get_feedback_by_model()
        
        # 按时间统计（按天）
        daily_stats = {}
        for feedback in feedback_list:
            timestamp = feedback.get('timestamp', '')
            if timestamp:
                try:
                    date = timestamp.split('T')[0]  # 提取日期部分
                    daily_stats[date] = daily_stats.get(date, 0) + 1
                except:
                    pass
        
        return {
            'rating_distribution': rating_counts,
            'model_stats': model_stats,
            'daily_feedback': daily_stats
        }
    
    def export_feedback_for_training(self, output_file: str = None) -> List[Dict[str, Any]]:
        """
        导出可用于模型训练的反馈数据
        
        Args:
            output_file: 输出文件路径，如果为None则不保存到文件
            
        Returns:
            可用于训练的数据列表
        """
        feedback_list = self.load_feedback()
        training_data = []
        
        for feedback in feedback_list:
            # 只导出正面反馈（4-5星）用于训练
            if feedback.get('rating', 0) >= 4:
                training_item = {
                    'instruction': feedback.get('user_input', ''),
                    'input': '',
                    'output': feedback.get('model_response', ''),
                    'rating': feedback.get('rating', 0)
                }
                training_data.append(training_item)
        
        # 保存到文件
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in training_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                logger.info(f"导出训练数据到: {output_file}")
            except Exception as e:
                logger.error(f"导出训练数据失败: {e}")
        
        return training_data


def main():
    """
    主函数 - 用于测试反馈处理器
    """
    handler = FeedbackHandler()
    
    # 示例反馈数据
    sample_feedback = {
        'session_id': 'test_session_001',
        'model_name': 'huanhuan-qwen-optimized',
        'user_input': '你好，你是谁？',
        'model_response': '臣妾是甄嬛，大理寺少卿甄远道之女。',
        'rating': 5,
        'comment': '回答很符合角色设定'
    }
    
    # 保存反馈
    handler.save_feedback(sample_feedback)
    
    # 获取统计信息
    stats = handler.get_feedback_stats()
    print("反馈统计:", stats)
    
    # 获取按模型分组的统计
    model_stats = handler.get_feedback_by_model()
    print("按模型分组统计:", model_stats)
    
    # 分析反馈模式
    patterns = handler.analyze_feedback_patterns()
    print("反馈模式分析:", patterns)


if __name__ == "__main__":
    main()