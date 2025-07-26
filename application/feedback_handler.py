#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户反馈处理模块

处理用户对模型回复的反馈，包括满意度评分和建议
"""

import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger


class FeedbackHandler:
    """
    用户反馈处理器
    """
    
    def __init__(self, feedback_file: str = None):
        """
        初始化反馈处理器
        
        Args:
            feedback_file: 反馈数据存储文件路径
        """
        # 如果未指定反馈文件，则使用项目根目录下的data目录
        if feedback_file is None:
            # 获取当前脚本所在目录的父级目录作为项目根目录
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            self.feedback_file = project_root / "data" / "user_feedback.json"
        else:
            self.feedback_file = Path(__file__).parent / feedback_file
            
        self.ensure_feedback_file_exists()
    
    def ensure_feedback_file_exists(self):
        """
        确保反馈文件存在
        """
        # 确保目录存在
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果文件不存在，创建一个空的JSON数组
        if not self.feedback_file.exists():
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            logger.info(f"创建反馈文件: {self.feedback_file}")
    
    def load_feedback(self) -> List[Dict[str, Any]]:
        """
        加载所有反馈数据
        
        Returns:
            List[Dict[str, Any]]: 反馈数据列表
        """
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                feedback_list = json.load(f)
            return feedback_list
        except Exception as e:
            logger.error(f"加载反馈数据失败: {e}")
            return []
    
    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的反馈数据
        
        Args:
            limit: 限制返回的反馈数量
            
        Returns:
            List[Dict[str, Any]]: 最近的反馈数据列表
        """
        try:
            feedback_list = self.load_feedback()
            
            # 按时间戳排序并取最近的几条
            sorted_feedback = sorted(
                feedback_list, 
                key=lambda x: x.get('timestamp', ''), 
                reverse=True
            )
            
            return sorted_feedback[:limit]
        except Exception as e:
            logger.error(f"获取最近反馈失败: {e}")
            return []
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        获取反馈统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            feedback_list = self.load_feedback()
            
            if not feedback_list:
                return {
                    "total_feedback": 0,
                    "positive_feedback": 0,
                    "negative_feedback": 0,
                    "positive_rate": 0.0,
                    "avg_rating": 0,
                    "rating_distribution": {}
                }
            
            # 计算统计数据
            ratings = [f.get('rating', 0) for f in feedback_list if 'rating' in f]
            total_feedback = len(feedback_list)
            positive_feedback = sum(1 for f in feedback_list if f.get('rating', 0) >= 4)
            negative_feedback = total_feedback - positive_feedback
            positive_rate = positive_feedback / total_feedback if total_feedback > 0 else 0
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            
            # 评分分布
            rating_distribution = {}
            for rating in range(1, 6):
                rating_distribution[str(rating)] = ratings.count(rating)
            
            return {
                "total_feedback": total_feedback,
                "positive_feedback": positive_feedback,
                "negative_feedback": negative_feedback,
                "positive_rate": positive_rate,
                "avg_rating": round(avg_rating, 2),
                "rating_distribution": rating_distribution
            }
            
        except Exception as e:
            logger.error(f"获取反馈统计失败: {e}")
            return {
                "total_feedback": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "positive_rate": 0.0,
                "avg_rating": 0,
                "rating_distribution": {}
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
    """主函数 - 用于测试"""
    handler = FeedbackHandler()
    
    # 测试保存反馈
    test_feedback = {
        "session_id": "test_session_123",
        "message": "这是一条测试消息",
        "response": "这是模型的回复",
        "rating": 5,
        "comment": "测试反馈"
    }
    
    if handler.save_feedback(test_feedback):
        print("✅ 反馈保存成功")
    else:
        print("❌ 反馈保存失败")
    
    # 测试获取统计信息
    stats = handler.get_feedback_stats()
    print(f"📊 反馈统计: {stats}")


if __name__ == "__main__":
    main()