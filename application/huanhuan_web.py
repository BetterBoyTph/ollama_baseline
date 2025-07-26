#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
甄嬛角色Web对话界面

基于Streamlit的甄嬛角色对话Web应用
参考: https://github.com/KMnO4-zx/huanhuan-chat

使用方法:
    streamlit run application/huanhuan_web.py
    streamlit run application/huanhuan_web.py --server.port 8501
"""

import os
import sys
import json
import requests
import streamlit as st
from pathlib import Path
from typing import List, Dict
import time
from datetime import datetime
import uuid

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入反馈处理模块
from application.feedback_handler import FeedbackHandler

# 页面配置
st.set_page_config(
    page_title="Chat-嬛嬛 - 甄嬛传角色对话",
    page_icon="👸",
    layout="wide",
    initial_sidebar_state="expanded"
)

class HuanHuanWebApp:
    """
    甄嬛Web应用
    """
    
    def __init__(self):
        self.ollama_host = "http://localhost:11434"
        self.model_name = "huanhuan-qwen"
        
        # 初始化反馈处理器
        self.feedback_handler = FeedbackHandler()
        
        # 初始化session state
        self.init_session_state()
    
    def init_session_state(self):
        """
        初始化会话状态
        """
        # 对话历史
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Ollama连接状态
        if "ollama_connected" not in st.session_state:
            st.session_state.ollama_connected = False
        
        # 可用模型列表
        if "available_models" not in st.session_state:
            st.session_state.available_models = []
        
        # 当前选择的模型
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = None
        
        # 生成参数
        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.7
        if "top_p" not in st.session_state:
            st.session_state.top_p = 0.9
        if "top_k" not in st.session_state:
            st.session_state.top_k = 40
        if "max_tokens" not in st.session_state:
            st.session_state.max_tokens = 256
        
        # 对话历史记录
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # 会话ID
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        # 当前对话的反馈状态
        if 'current_feedback' not in st.session_state:
            st.session_state.current_feedback = {}
    
    def check_ollama_connection(self) -> bool:
        """
        检查Ollama服务连接状态
        """
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                st.session_state.ollama_connected = True
                return True
        except requests.exceptions.RequestException:
            pass
        
        st.session_state.ollama_connected = False
        return False
    
    def get_available_models(self) -> List[str]:
        """
        获取可用的模型列表
        """
        if not self.check_ollama_connection():
            return []
        
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data['models']]
                st.session_state.available_models = models
                return models
        except Exception as e:
            st.error(f"获取模型列表失败: {e}")
        
        return []
    
    def chat_with_model(self, prompt: str, model: str) -> str:
        """
        与模型对话
        
        Args:
            prompt: 用户输入
            model: 模型名称
            
        Returns:
            模型回复
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": st.session_state.temperature,
                    "top_p": st.session_state.top_p,
                    "top_k": st.session_state.top_k,
                    "num_predict": st.session_state.max_tokens
                }
            }
            
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                return f"错误: HTTP {response.status_code}"
        except Exception as e:
            return f"请求失败: {str(e)}"
    
    def save_chat_history(self):
        """
        保存聊天历史到文件
        """
        if not st.session_state.messages:
            return
        
        try:
            # 创建聊天历史目录
            history_dir = Path(__file__).parent / "chat_history"
            history_dir.mkdir(exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"huanhuan_chat_{timestamp}.json"
            filepath = history_dir / filename
            
            # 保存聊天历史
            chat_data = {
                "session_id": st.session_state.session_id,
                "timestamp": timestamp,
                "messages": st.session_state.messages
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
            
            st.success(f"聊天历史已保存: {filename}")
        except Exception as e:
            st.error(f"保存聊天历史失败: {e}")
    
    def render_sidebar(self):
        """
        渲染侧边栏
        """
        with st.sidebar:
            st.title("👑 Chat-嬛嬛")
            st.markdown("---")
            
            # 连接状态
            if self.check_ollama_connection():
                st.success("🟢 Ollama服务已连接")
            else:
                st.error("🔴 Ollama服务未连接")
                st.info("请确保Ollama服务正在运行")
            
            # 模型选择
            available_models = self.get_available_models()
            if available_models:
                st.session_state.selected_model = st.selectbox(
                    "选择模型",
                    options=available_models,
                    index=0 if st.session_state.selected_model is None else 
                          available_models.index(st.session_state.selected_model) 
                          if st.session_state.selected_model in available_models else 0
                )
            else:
                st.warning("未找到可用模型")
                st.session_state.selected_model = None
            
            st.markdown("---")
            
            # 参数调节
            st.subheader("⚙️ 参数调节")
            st.session_state.temperature = st.slider(
                "Temperature", 0.0, 1.0, st.session_state.temperature, 0.1
            )
            st.session_state.top_p = st.slider(
                "Top-p", 0.0, 1.0, st.session_state.top_p, 0.1
            )
            st.session_state.top_k = st.slider(
                "Top-k", 1, 100, st.session_state.top_k, 1
            )
            st.session_state.max_tokens = st.slider(
                "最大生成长度", 50, 1000, st.session_state.max_tokens, 50
            )
            
            st.markdown("---")
            
            # 功能按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 保存对话"):
                    self.save_chat_history()
            
            with col2:
                if st.button("🗑️ 清空对话"):
                    st.session_state.messages = []
                    st.session_state.session_id = str(uuid.uuid4())
                    st.rerun()
            
            # 反馈统计
            st.markdown("---")
            st.subheader("📊 反馈统计")
            feedback_stats = self.feedback_handler.get_feedback_stats()
            st.metric("总反馈数", feedback_stats['total_feedback'])
            st.metric("正面反馈率", f"{feedback_stats['positive_rate']:.2%}")
            st.metric("平均评分", feedback_stats['avg_rating'])
    
    def render_feedback_section(self, message_index: int):
        """
        渲染反馈部分
        
        Args:
            message_index: 消息索引
        """
        # 获取消息
        if message_index >= len(st.session_state.messages):
            return
        
        message = st.session_state.messages[message_index]
        if message['role'] != 'assistant':
            return
        
        # 生成反馈组件的唯一键
        feedback_key = f"feedback_{message_index}"
        
        # 检查是否已经有反馈
        if feedback_key in st.session_state.current_feedback:
            st.success("✅ 感谢您的反馈！")
            return
        
        # 显示反馈组件
        st.markdown("---")
        st.markdown("#### 请对这个回答进行评价：")
        
        # 评分
        rating = st.radio(
            "评分",
            options=[("⭐", 1), ("⭐⭐", 2), ("⭐⭐⭐", 3), ("⭐⭐⭐⭐", 4), ("⭐⭐⭐⭐⭐", 5)],
            format_func=lambda x: x[0],
            key=f"rating_{message_index}",
            horizontal=True
        )
        
        # 评论
        comment = st.text_area(
            "详细评论（可选）",
            key=f"comment_{message_index}",
            placeholder="您觉得这个回答怎么样？有什么建议吗？"
        )
        
        # 提交按钮
        if st.button("提交反馈", key=f"submit_{message_index}"):
            if rating:
                # 准备反馈数据
                user_message = st.session_state.messages[message_index-1] if message_index > 0 else {"content": ""}
                
                feedback_data = {
                    "session_id": st.session_state.session_id,
                    "model_name": st.session_state.selected_model or "unknown",
                    "user_input": user_message.get("content", ""),
                    "model_response": message.get("content", ""),
                    "rating": rating[1],  # 获取评分值
                    "comment": comment
                }
                
                # 保存反馈
                if self.feedback_handler.save_feedback(feedback_data):
                    # 记录已提交反馈
                    st.session_state.current_feedback[feedback_key] = True
                    st.success("✅ 感谢您的反馈！")
                    st.rerun()
                else:
                    st.error("❌ 提交反馈失败，请重试")
    
    def render_chat_interface(self):
        """
        渲染聊天界面
        """
        # 显示聊天历史
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # 如果是助手回复，显示反馈部分
                if message["role"] == "assistant":
                    self.render_feedback_section(i)
        
        # 用户输入
        if prompt := st.chat_input("与甄嬛对话..."):
            # 添加用户消息
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 显示用户消息
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 获取模型回复
            if st.session_state.selected_model:
                with st.chat_message("assistant"):
                    with st.spinner("甄嬛正在思考..."):
                        response = self.chat_with_model(prompt, st.session_state.selected_model)
                    
                    st.markdown(response)
                    
                    # 添加助手回复到历史
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # 显示反馈部分
                    self.render_feedback_section(len(st.session_state.messages) - 1)
            else:
                st.warning("请先选择一个模型")
    
    def render_feedback_analysis(self):
        """
        渲染反馈分析页面
        """
        st.title("📈 反馈分析")
        
        # 获取反馈统计数据
        feedback_stats = self.feedback_handler.get_feedback_stats()
        
        # 显示总体统计
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总反馈数", feedback_stats['total_feedback'])
        with col2:
            st.metric("正面反馈", feedback_stats['positive_feedback'])
        with col3:
            st.metric("负面反馈", feedback_stats['negative_feedback'])
        with col4:
            st.metric("平均评分", feedback_stats['avg_rating'])
        
        st.progress(feedback_stats['positive_rate'], 
                   f"正面反馈率: {feedback_stats['positive_rate']:.2%}")
        
        # 按模型分组的统计
        st.subheader("各模型反馈统计")
        model_stats = self.feedback_handler.get_feedback_by_model()
        
        if model_stats:
            for model_name, stats in model_stats.items():
                with st.expander(f"🤖 {model_name}"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("反馈数", stats['stats']['total'])
                    with col2:
                        st.metric("正面反馈", stats['stats']['positive'])
                    with col3:
                        st.metric("负面反馈", stats['stats']['negative'])
                    with col4:
                        st.metric("平均评分", stats['stats']['avg_rating'])
        else:
            st.info("暂无模型反馈数据")
        
        # 最近反馈
        st.subheader("最近反馈")
        recent_feedback = self.feedback_handler.get_recent_feedback(10)
        
        if recent_feedback:
            for feedback in recent_feedback:
                with st.expander(f"⭐ {feedback.get('rating', 0)}星 - {feedback.get('timestamp', '')}"):
                    st.markdown(f"**模型**: {feedback.get('model_name', 'unknown')}")
                    st.markdown(f"**用户输入**: {feedback.get('user_input', '')}")
                    st.markdown(f"**模型回复**: {feedback.get('model_response', '')}")
                    if feedback.get('comment'):
                        st.markdown(f"**评论**: {feedback.get('comment', '')}")
        else:
            st.info("暂无反馈数据")
    
    def render_training_data_export(self):
        """
        渲染训练数据导出页面
        """
        st.title("📤 训练数据导出")
        
        st.markdown("""
        本页面允许您将用户正面反馈导出为训练数据，用于模型的持续优化。
        只有评分4星及以上的反馈会被导出。
        """)
        
        # 获取反馈统计数据
        feedback_stats = self.feedback_handler.get_feedback_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("总反馈数", feedback_stats['total_feedback'])
        with col2:
            st.metric("可用于训练的反馈数", 
                     sum(1 for f in self.feedback_handler.load_feedback() if f.get('rating', 0) >= 4))
        
        # 导出选项
        st.subheader("导出设置")
        output_filename = st.text_input("输出文件名", "user_feedback_training_data.jsonl")
        
        # 导出按钮
        if st.button("导出训练数据"):
            try:
                # 导出数据
                training_data = self.feedback_handler.export_feedback_for_training(
                    output_file=str(Path(__file__).parent.parent / "data" / output_filename)
                )
                
                st.success(f"✅ 成功导出 {len(training_data)} 条训练数据到 {output_filename}")
                
                # 显示示例数据
                if training_data:
                    st.subheader("导出数据示例")
                    for i, item in enumerate(training_data[:3]):
                        st.markdown(f"**示例 {i+1}:**")
                        st.json(item)
            except Exception as e:
                st.error(f"导出失败: {e}")
    
    def run(self):
        """
        运行应用
        """
        # 页面选择
        page = st.sidebar.radio("页面导航", ["💬 对话", "📈 反馈分析", "📤 训练数据导出"])
        
        if page == "💬 对话":
            self.render_sidebar()
            self.render_chat_interface()
        elif page == "📈 反馈分析":
            self.render_feedback_analysis()
        elif page == "📤 训练数据导出":
            self.render_training_data_export()


def main():
    """
    主函数
    """
    app = HuanHuanWebApp()
    app.run()


if __name__ == "__main__":
    main()