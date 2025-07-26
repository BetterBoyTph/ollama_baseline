#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估模块

用于对大模型输出结果进行自动化评估，包含多种评估指标
"""

import os
import json
import re
import jieba
from collections import Counter
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from rouge import Rouge


class ModelEvaluator:
    """
    模型评估器类，用于评估大模型输出结果
    """

    def __init__(self):
        """
        初始化模型评估器
        """
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def load_test_data(self, data_path: str) -> List[Dict]:
        """
        加载测试数据
        
        Args:
            data_path: 测试数据文件路径
            
        Returns:
            测试数据列表
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.jsonl'):
                return [json.loads(line) for line in f]
            else:
                return json.load(f)

    def get_model_response(self, model_name: str, prompt: str, **kwargs) -> str:
        """
        获取模型响应
        
        Args:
            model_name: 模型名称
            prompt: 提示词
            **kwargs: 其他参数
            
        Returns:
            模型响应文本
        """
        request_data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": kwargs
        }

        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=request_data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except Exception as e:
            print(f"获取模型响应失败: {e}")
            return ""

    def calculate_bleu(self, candidate: str, references: List[str], n: int = 4) -> float:
        """
        计算BLEU分数
        
        BLEU (Bilingual Evaluation Understudy) 是机器翻译评估的经典指标。
        它通过比较生成文本和参考文本中的n-gram重叠来评估质量。
        分数范围为0-1，越高越好。
        
        Args:
            candidate: 候选文本（模型生成）
            references: 参考文本列表（标准答案）
            n: n-gram的最大长度
            
        Returns:
            BLEU分数
        """
        def get_ngrams(text, n):
            words = list(jieba.cut(text))
            ngrams = []
            for i in range(len(words) - n + 1):
                ngrams.append(' '.join(words[i:i + n]))
            return ngrams

        def calculate_precision(candidate_ngrams, reference_ngrams_list):
            candidate_counts = Counter(candidate_ngrams)
            max_counts = Counter()
            
            for ref_ngrams in reference_ngrams_list:
                ref_counts = Counter(ref_ngrams)
                for ngram in candidate_counts:
                    max_counts[ngram] = max(max_counts[ngram], ref_counts[ngram])
            
            clip_count = sum(min(candidate_counts[ngram], max_counts[ngram]) 
                           for ngram in candidate_counts)
            total_count = sum(candidate_counts.values())
            
            return clip_count, total_count

        if not candidate.strip():
            return 0.0

        bleu_scores = []
        for i in range(1, n + 1):
            candidate_ngrams = get_ngrams(candidate, i)
            reference_ngrams_list = [get_ngrams(ref, i) for ref in references]
            
            clip_count, total_count = calculate_precision(candidate_ngrams, reference_ngrams_list)
            
            if total_count == 0:
                bleu_scores.append(0.0)
            else:
                bleu_scores.append(clip_count / total_count)

        # 计算 brevity penalty (简洁性惩罚)
        candidate_len = len(list(jieba.cut(candidate)))
        ref_lens = [len(list(jieba.cut(ref))) for ref in references]
        closest_ref_len = min(ref_lens, key=lambda x: abs(x - candidate_len))
        
        if candidate_len > closest_ref_len:
            bp = 1.0
        else:
            bp = np.exp(1 - closest_ref_len / candidate_len) if candidate_len > 0 else 0

        # 几何平均
        if 0 in bleu_scores or len(bleu_scores) == 0:
            bleu = 0.0
        else:
            log_avg = sum(np.log(score) for score in bleu_scores) / len(bleu_scores)
            bleu = bp * np.exp(log_avg)
        
        return bleu

    def calculate_rouge(self, candidate: str, reference: str) -> Dict[str, float]:
        """
        计算ROUGE分数
        
        ROUGE (Recall-Oriented Understudy for Gisting Evaluation) 主要用于摘要评估。
        包括ROUGE-N（基于n-gram）、ROUGE-L（基于最长公共子序列）等变体。
        分数范围为0-1，越高越好。
        
        Args:
            candidate: 候选文本（模型生成）
            reference: 参考文本（标准答案）
            
        Returns:
            ROUGE分数字典 {'rouge-1', 'rouge-2', 'rouge-l'}
        """
        # 使用专门的ROUGE库进行计算，提高准确性
        try:
            rouge = Rouge()
            scores = rouge.get_scores(candidate, reference)
            return {
                'rouge-1': scores[0]['rouge-1']['f'],
                'rouge-2': scores[0]['rouge-2']['f'],
                'rouge-l': scores[0]['rouge-l']['f']
            }
        except:
            # 如果ROUGE库计算失败，使用原有方法
            def get_ngrams(text, n):
                words = list(jieba.cut(text))
                ngrams = []
                for i in range(len(words) - n + 1):
                    ngrams.append(' '.join(words[i:i + n]))
                return set(ngrams)

            def lcs_length(x, y):
                """计算最长公共子序列长度"""
                m, n = len(x), len(y)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if x[i-1] == y[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
                return dp[m][n]

            def rouge_n(candidate, reference, n):
                """计算ROUGE-N"""
                candidate_ngrams = get_ngrams(candidate, n)
                reference_ngrams = get_ngrams(reference, n)
                
                if len(candidate_ngrams) == 0 or len(reference_ngrams) == 0:
                    return 0.0
                    
                overlap = len(candidate_ngrams & reference_ngrams)
                return overlap / len(reference_ngrams)

            def rouge_l(candidate, reference):
                """计算ROUGE-L"""
                candidate_words = list(jieba.cut(candidate))
                reference_words = list(jieba.cut(reference))
                
                if len(candidate_words) == 0 or len(reference_words) == 0:
                    return 0.0
                    
                lcs = lcs_length(candidate_words, reference_words)
                return lcs / len(reference_words)

            if not candidate.strip() or not reference.strip():
                return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}

            rouge1 = rouge_n(candidate, reference, 1)
            rouge2 = rouge_n(candidate, reference, 2)
            rougel = rouge_l(candidate, reference)
            
            return {
                'rouge-1': rouge1,
                'rouge-2': rouge2,
                'rouge-l': rougel
            }

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        计算语义相似度（基于TF-IDF的余弦相似度）
        
        通过TF-IDF向量化文本，然后计算向量间的余弦相似度来衡量语义相似程度。
        分数范围为0-1，越高表示语义越相似。
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            语义相似度分数
        """
        if not text1.strip() or not text2.strip():
            return 0.0

        # 使用TF-IDF向量化
        vectorizer = TfidfVectorizer(tokenizer=jieba.cut, lowercase=False)
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0

    def calculate_diversity(self, texts: List[str]) -> float:
        """
        计算文本多样性
        
        通过计算多个生成文本之间的平均相似度来衡量多样性，
        多样性 = 1 - 平均相似度。分数范围为0-1，越高表示多样性越好。
        
        Args:
            texts: 文本列表
            
        Returns:
            多样性分数
        """
        if len(texts) < 2:
            return 0.0

        # 计算所有文本对之间的相似度
        similarities = []
        vectorizer = TfidfVectorizer(tokenizer=jieba.cut, lowercase=False)
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    similarity = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[j:j+1])[0][0]
                    similarities.append(similarity)
            
            avg_similarity = np.mean(similarities)
            diversity = 1.0 - avg_similarity
            return float(diversity)
        except:
            return 0.0

    def calculate_coherence(self, text: str) -> float:
        """
        计算文本连贯性（基于句子间相似度）
        
        将文本分割为句子，计算相邻句子间的相似度平均值。
        分数范围为0-1，越高表示文本连贯性越好。
        
        Args:
            text: 输入文本
            
        Returns:
            连贯性分数
        """
        # 简单的句子分割
        sentences = re.split(r'[。！？.;!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0

        # 计算相邻句子间的相似度
        similarities = []
        vectorizer = TfidfVectorizer(tokenizer=jieba.cut, lowercase=False)
        
        try:
            for i in range(len(sentences) - 1):
                tfidf_matrix = vectorizer.fit_transform([sentences[i], sentences[i+1]])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                similarities.append(similarity)
            
            coherence = np.mean(similarities)
            return float(coherence)
        except:
            return 0.0

    def calculate_character_consistency(self, text: str, keywords: List[str] = None) -> float:
        """
        计算角色一致性
        
        通过检测文本中是否包含特定角色关键词来评估角色一致性。
        分数范围为0-1，越高表示角色一致性越好。
        
        Args:
            text: 输入文本
            keywords: 角色关键词列表
            
        Returns:
            角色一致性分数
        """
        if keywords is None:
            # 默认的甄嬛角色关键词
            keywords = ["臣妾", "皇上", "娘娘", "本宫", "便是", "倒是"]
        
        if not text.strip():
            return 0.0
            
        # 计算关键词出现频率
        keyword_count = sum(1 for keyword in keywords if keyword in text)
        consistency_score = keyword_count / len(keywords)
        
        return min(consistency_score, 1.0)

    def calculate_fluency(self, text: str) -> float:
        """
        计算文本流畅度
        
        通过分析文本的长度、句子结构复杂度等来评估文本流畅度。
        分数范围为0-1，越高表示文本越流畅。
        
        Args:
            text: 输入文本
            
        Returns:
            流畅度分数
        """
        if not text.strip():
            return 0.0
            
        # 基本的流畅度评估
        sentences = re.split(r'[。！？.;!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
            
        # 平均句子长度（以词为单位）
        words_per_sentence = [len(list(jieba.cut(sentence))) for sentence in sentences]
        avg_length = np.mean(words_per_sentence)
        
        # 理想的句子长度在10-25词之间
        if avg_length < 5:
            fluency = avg_length / 5
        elif avg_length > 30:
            fluency = 30 / avg_length
        else:
            fluency = 1.0
            
        return float(fluency)

    def evaluate_single_response(self, response: str, references: List[str]) -> Dict[str, Any]:
        """
        评估单个模型响应
        
        Args:
            response: 模型响应
            references: 参考答案列表
            
        Returns:
            评估结果字典
        """
        if not response.strip():
            return {
                'bleu': 0.0,
                'rouge': {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0},
                'semantic_similarity': 0.0,
                'coherence': 0.0,
                'character_consistency': 0.0,
                'fluency': 0.0
            }

        # 计算各项指标
        bleu_score = self.calculate_bleu(response, references)
        rouge_scores = self.calculate_rouge(response, references[0] if references else "")
        semantic_sim = max([self.calculate_semantic_similarity(response, ref) for ref in references]) \
                       if references else 0.0
        coherence = self.calculate_coherence(response)
        character_consistency = self.calculate_character_consistency(response)
        fluency = self.calculate_fluency(response)
        
        return {
            'bleu': bleu_score,
            'rouge': rouge_scores,
            'semantic_similarity': semantic_sim,
            'coherence': coherence,
            'character_consistency': character_consistency,
            'fluency': fluency
        }

    def evaluate_model(self, model_name: str, test_data: List[Dict], sample_size: int = None) -> Dict[str, Any]:
        """
        评估模型在测试数据上的表现
        
        Args:
            model_name: 模型名称
            test_data: 测试数据
            sample_size: 采样数量，None表示使用全部数据
            
        Returns:
            模型评估结果
        """
        if sample_size and sample_size < len(test_data):
            test_data = test_data[:sample_size]
        
        results = []
        responses = []
        
        for item in test_data:
            # 支持多种数据格式
            prompt = item.get('prompt') or item.get('input') or item.get('instruction', '')
            references = item.get('references') or item.get('outputs') or item.get('output', [])
            if isinstance(references, str):
                references = [references]
            
            # 获取模型响应
            response = self.get_model_response(model_name, prompt)
            responses.append(response)
            
            # 评估响应
            eval_result = self.evaluate_single_response(response, references)
            eval_result['prompt'] = prompt
            eval_result['response'] = response
            eval_result['references'] = references
            results.append(eval_result)
        
        # 计算平均分数
        avg_bleu = np.mean([r['bleu'] for r in results])
        avg_rouge_1 = np.mean([r['rouge']['rouge-1'] for r in results])
        avg_rouge_2 = np.mean([r['rouge']['rouge-2'] for r in results])
        avg_rouge_l = np.mean([r['rouge']['rouge-l'] for r in results])
        avg_semantic_sim = np.mean([r['semantic_similarity'] for r in results])
        avg_coherence = np.mean([r['coherence'] for r in results])
        avg_character_consistency = np.mean([r['character_consistency'] for r in results])
        avg_fluency = np.mean([r['fluency'] for r in results])
        
        # 计算整体多样性
        diversity = self.calculate_diversity(responses)
        
        return {
            'model_name': model_name,
            'total_samples': len(results),
            'average_scores': {
                'bleu': float(avg_bleu),
                'rouge-1': float(avg_rouge_1),
                'rouge-2': float(avg_rouge_2),
                'rouge-l': float(avg_rouge_l),
                'semantic_similarity': float(avg_semantic_sim),
                'coherence': float(avg_coherence),
                'character_consistency': float(avg_character_consistency),
                'fluency': float(avg_fluency),
                'diversity': float(diversity)
            },
            'individual_results': results
        }

    def compare_models(self, model_names: List[str], test_data_path: str, 
                      sample_size: int = None) -> Dict[str, Any]:
        """
        比较多个模型的评估结果
        
        Args:
            model_names: 模型名称列表
            test_data_path: 测试数据路径
            sample_size: 采样数量
            
        Returns:
            模型比较结果
        """
        test_data = self.load_test_data(test_data_path)
        
        comparison_results = {}
        for model_name in model_names:
            print(f"正在评估模型: {model_name}")
            result = self.evaluate_model(model_name, test_data, sample_size)
            comparison_results[model_name] = result
        
        return comparison_results

    def generate_report(self, comparison_results: Dict[str, Any], output_path: str = None) -> str:
        """
        生成评估报告
        
        Args:
            comparison_results: 比较结果
            output_path: 输出路径，None表示不保存到文件
            
        Returns:
            评估报告字符串
        """
        report = []
        report.append("# 大模型评估报告\n")
        report.append(f"评估时间: {np.datetime_as_string(np.datetime64('now'), unit='s')}\n")
        report.append("---\n")
        
        # 模型性能对比表
        report.append("## 模型性能对比\n")
        report.append("| 模型名称 | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | 语义相似度 | 连贯性 | 角色一致性 | 流畅度 | 多样性 | 样本数 |")
        report.append("|---------|------|---------|---------|---------|------------|--------|------------|--------|--------|--------|")
        
        for model_name, result in comparison_results.items():
            scores = result['average_scores']
            report.append(
                f"| {model_name} | {scores['bleu']:.4f} | {scores['rouge-1']:.4f} | "
                f"{scores['rouge-2']:.4f} | {scores['rouge-l']:.4f} | "
                f"{scores['semantic_similarity']:.4f} | {scores['coherence']:.4f} | "
                f"{scores['character_consistency']:.4f} | {scores['fluency']:.4f} | "
                f"{scores['diversity']:.4f} | {result['total_samples']} |"
            )
        
        report.append("\n")
        
        # 详细指标说明
        report.append("## 评估指标说明\n")
        report.append("1. **BLEU**: 用于评估生成文本与参考文本之间的n-gram重叠度，广泛应用于机器翻译和文本生成评估。\n")
        report.append("2. **ROUGE-1/2/L**: 用于评估文本摘要质量，分别基于单词、二元组和最长公共子序列的召回率。\n")
        report.append("3. **语义相似度**: 基于TF-IDF向量和余弦相似度计算生成文本与参考文本的语义接近程度。\n")
        report.append("4. **连贯性**: 衡量生成文本内部句子之间的逻辑连贯程度。\n")
        report.append("5. **角色一致性**: 衡量生成文本是否符合特定角色的语言风格和特征。\n")
        report.append("6. **流畅度**: 衡量生成文本的语言流畅程度。\n")
        report.append("7. **多样性**: 衡量模型生成不同响应的能力，避免重复和单调。\n")
        
        # 各模型详细结果
        report.append("## 各模型详细结果\n")
        for model_name, result in comparison_results.items():
            report.append(f"### {model_name}\n")
            report.append("#### 平均分数\n")
            scores = result['average_scores']
            report.append(f"- BLEU: {scores['bleu']:.4f}\n")
            report.append(f"- ROUGE-1: {scores['rouge-1']:.4f}\n")
            report.append(f"- ROUGE-2: {scores['rouge-2']:.4f}\n")
            report.append(f"- ROUGE-L: {scores['rouge-l']:.4f}\n")
            report.append(f"- 语义相似度: {scores['semantic_similarity']:.4f}\n")
            report.append(f"- 连贯性: {scores['coherence']:.4f}\n")
            report.append(f"- 角色一致性: {scores['character_consistency']:.4f}\n")
            report.append(f"- 流畅度: {scores['fluency']:.4f}\n")
            report.append(f"- 多样性: {scores['diversity']:.4f}\n")
            report.append("\n")
        
        report_str = '\n'.join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
        
        return report_str


def main():
    """
    主函数 - 用于命令行运行评估器
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='大模型自动化评估工具')
    parser.add_argument('--models', nargs='+', required=True, help='要评估的模型名称列表')
    parser.add_argument('--data', required=True, help='测试数据路径')
    parser.add_argument('--output', help='评估报告输出路径')
    parser.add_argument('--samples', type=int, help='采样数量')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator()
    results = evaluator.compare_models(args.models, args.data, args.samples)
    report = evaluator.generate_report(results, args.output)
    
    print(report)


if __name__ == "__main__":
    main()