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
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import psutil

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
        self.vllm_host = "http://localhost:8000"
        self.model_name = "huanhuan-qwen"
        
        # 根据系统配置设置并发参数 (RTX 3090 24G, 16核心CPU, 120G内存)
        # CPU核心数的一半作为并发请求数，但不超过8
        cpu_count = psutil.cpu_count(logical=False) or 4
        self.max_concurrent_requests = min(8, max(1, cpu_count // 2))
        
        # 创建线程池用于并发请求
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
        
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
        
        # vLLM连接状态
        if "vllm_connected" not in st.session_state:
            st.session_state.vllm_connected = False
        
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
        
        # 反馈相关状态 - 存储当前对话中各消息的反馈评分
        if 'current_feedback' not in st.session_state:
            # 格式: {message_index: rating}
            st.session_state.current_feedback = {}
        
        # 会话ID
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(datetime.now().timestamp())
    
    def check_vllm_connection(self) -> bool:
        """
        检查vLLM服务连接状态
        """
        try:
            response = requests.get(f"{self.vllm_host}/v1/models", timeout=5)
            if response.status_code == 200:
                st.session_state.vllm_connected = True
                return True
        except requests.exceptions.RequestException:
            pass
        
        st.session_state.vllm_connected = False
        return False
    
    def get_available_models(self) -> List[str]:
        """
        获取可用的模型列表
        """
        if not self.check_vllm_connection():
            return []
        
        try:
            response = requests.get(f"{self.vllm_host}/v1/models")
            if response.status_code == 200:
                data = response.json()
                models = [model['id'] for model in data.get('data', [])]
                st.session_state.available_models = models
                return models
        except Exception as e:
            st.error(f"获取模型列表失败: {e}")
        
        return []
    
    def stream_chat(self, messages, model):
        """
        流式对话生成 (模拟流式，vLLM API不直接支持流式)
        """
        # 构建完整的对话历史
        conversation_history = ""
        for msg in messages:
            if msg["role"] == "user":
                conversation_history += f"用户: {msg['content']}\n"
            else:
                conversation_history += f"甄嬛: {msg['content']}\n"
        
        # 构建请求数据
        request_data = {
            "model": model,
            "prompt": conversation_history + "甄嬛:",
            "max_tokens": st.session_state.max_tokens,
            "temperature": st.session_state.temperature,
            "top_p": st.session_state.top_p,
            "top_k": st.session_state.top_k,
            "stream": False
        }
        
        try:
            # 使用线程池执行同步请求以支持并发
            future = self.executor.submit(self._chat_sync, request_data)
            full_response = future.result()
            
            # 模拟流式输出
            for i in range(len(full_response)):
                yield full_response[:i+1]
                time.sleep(0.01)  # 模拟打字效果
                        
        except Exception as e:
            yield f"连接错误: {e}"

    def _chat_sync(self, request_data):
        """
        同步聊天请求方法，用于在线程池中执行
        """
        try:
            response = requests.post(
                f"{self.vllm_host}/v1/completions",
                json=request_data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['text'] if result.get('choices') else ""
        except Exception as e:
            raise e
    
    def render_sidebar(self):
        """
        渲染侧边栏
        """
        with st.sidebar:
            st.title("⚙️ 设置")
            
            # 连接状态
            if self.check_vllm_connection():
                st.success("🟢 vLLM服务已连接")
            else:
                st.error("🔴 vLLM服务未连接")
                st.info("请确保vLLM服务正在运行")
            
            st.divider()
            
            # 模型选择
            st.subheader("🤖 模型设置")
            available_models = self.get_available_models()
            
            if available_models:
                if self.model_name in available_models:
                    default_index = available_models.index(self.model_name)
                else:
                    default_index = 0
                
                selected_model = st.selectbox(
                    "选择模型",
                    available_models,
                    index=default_index
                )
                st.session_state.selected_model = selected_model
                self.model_name = selected_model
            else:
                st.warning("未找到可用模型")
                st.info("请先部署甄嬛模型")
            
            st.divider()
            
            # 参数调节
            st.subheader("🎛️ 生成参数")
            
            st.session_state.temperature = st.slider(
                "Temperature (创造性)",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.temperature,
                step=0.1,
                help="控制回答的随机性，值越高越有创造性"
            )
            
            st.session_state.top_p = st.slider(
                "Top P (多样性)",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.top_p,
                step=0.1,
                help="控制词汇选择的多样性"
            )
            
            st.session_state.top_k = st.slider(
                "Top K (词汇范围)",
                min_value=1,
                max_value=100,
                value=st.session_state.top_k,
                step=1,
                help="限制每步选择的词汇数量"
            )
            
            st.session_state.max_tokens = st.slider(
                "Max Tokens (回答长度)",
                min_value=50,
                max_value=500,
                value=st.session_state.max_tokens,
                step=10,
                help="控制回答的最大长度"
            )
            
            st.divider()
            
            # 反馈统计
            st.subheader("📊 反馈统计")
            feedback_stats = self.feedback_handler.get_feedback_stats()
            
            if feedback_stats["total_feedback"] > 0:
                st.metric("总反馈数", feedback_stats["total_feedback"])
                st.metric("平均评分", f"{feedback_stats['avg_rating']:.2f} ⭐")
                st.progress(feedback_stats["positive_rate"], f"好评率: {feedback_stats['positive_rate']*100:.1f}%")
                
                # 评分分布
                st.write("评分分布:")
                for rating in range(5, 0, -1):
                    count = feedback_stats["rating_distribution"].get(str(rating), 0)
                    if count > 0:
                        st.progress(count / feedback_stats["total_feedback"], 
                                   f"{'⭐' * rating}: {count}")
            else:
                st.info("暂无反馈数据")
            
            st.divider()
            
            # 功能按钮
            st.subheader("🛠️ 功能")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🗑️ 清空对话", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                    st.session_state.current_feedback = {}
                    st.rerun()
            
            with col2:
                if st.button("💾 保存对话", use_container_width=True):
                    if st.session_state.chat_history:
                        self.save_chat_history()
                        st.success("对话已保存！")
                    else:
                        st.warning("没有对话内容可保存")
            
            with col3:
                if st.button("📂 加载对话", use_container_width=True):
                    self.load_chat_history()
    
    def render_main_content(self):
        """
        渲染主要内容
        """
        # 标题和介绍
        st.title("👸 Chat-嬛嬛")
        st.markdown("""
        欢迎来到甄嬛传角色对话系统！我是甄嬛，大理寺少卿甄远道之女。
        臣妾愿与您畅谈宫廷生活、诗词歌赋，分享人生感悟。
        """)
        
        # 角色信息卡片
        with st.expander("📖 角色信息", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **基本信息**
                - 姓名：甄嬛（甄玉嬛）
                - 身份：熹贵妃
                - 出身：大理寺少卿甄远道之女
                - 特长：诗词歌赋、琴棋书画
                """)
            
            with col2:
                st.markdown("""
                **性格特点**
                - 聪慧机智，善于应变
                - 温婉贤淑，知书达理
                - 坚韧不拔，重情重义
                - 语言典雅，谦逊有礼
                """)
        
        # 示例问题
        st.subheader("💡 示例问题")
        example_questions = [
            "你好，请介绍一下自己",
            "你觉得宫廷生活如何？",
            "如何看待友情？",
            "能为我作一首诗吗？",
            "给后人一些人生建议",
            "你最喜欢什么？"
        ]
        
        cols = st.columns(3)
        for i, question in enumerate(example_questions):
            with cols[i % 3]:
                if st.button(question, key=f"example_{i}", use_container_width=True):
                    st.session_state.current_question = question
        
        st.divider()
        
        # 对话历史
        st.subheader("💬 对话历史")
        
        # 显示对话消息并添加反馈功能
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # 为助手消息添加反馈功能
                if message["role"] == "assistant":
                    # 获取消息索引（助手消息在奇数位置）
                    message_index = i
                    
                    # 确保反馈状态已初始化
                    self.init_message_feedback_state(message_index)
                    
                    # 显示已有的反馈状态
                    if st.session_state.current_feedback.get(message_index) is not None:
                        rating = st.session_state.current_feedback[message_index]
                        st.markdown(f"您的评分: {'⭐' * rating}")
                    else:
                        # 显示评分按钮
                        st.markdown("**请为此回复评分：**")
                        cols = st.columns(5)
                        for j, col in enumerate(cols):
                            rating_value = j + 1
                            if col.button(f"{'⭐' * rating_value}", key=f"rate_{message_index}_{rating_value}"):
                                # 保存反馈
                                self.save_message_feedback(message_index, rating_value, st.session_state.messages)
                                st.session_state.current_feedback[message_index] = rating_value
                                st.rerun()
        
        # 处理示例问题
        if hasattr(st.session_state, 'current_question'):
            user_input = st.session_state.current_question
            delattr(st.session_state, 'current_question')
        else:
            user_input = None
        
        # 聊天输入
        if prompt := st.chat_input("请输入您的问题...") or user_input:
            # 添加用户消息
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 生成回复
            with st.chat_message("assistant"):
                with st.spinner("甄嬛正在思考..."):
                    # 使用流式生成
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    # 构建消息历史
                    messages = []
                    for msg in st.session_state.messages:
                        messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    
                    for chunk in self.stream_chat(messages, st.session_state.selected_model):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
            
            # 添加助手消息
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # 保存到历史记录
            st.session_state.chat_history.append({
                "timestamp": datetime.now().isoformat(),
                "user": prompt,
                "assistant": full_response,
                "params": {
                    "temperature": st.session_state.temperature,
                    "top_p": st.session_state.top_p,
                    "top_k": st.session_state.top_k,
                    "max_tokens": st.session_state.max_tokens
                }
            })
            
            # 初始化当前回复的反馈状态
            latest_message_index = len(st.session_state.messages) - 1
            self.init_message_feedback_state(latest_message_index)
            
            # 刷新页面以显示新消息及其反馈界面
            st.rerun()
    
    def init_message_feedback_state(self, message_index):
        """
        初始化特定消息的反馈状态
        
        Args:
            message_index: 消息索引
        """
        if message_index not in st.session_state.current_feedback:
            st.session_state.current_feedback[message_index] = None
    
    def save_message_feedback(self, message_index, rating, messages):
        """
        保存特定消息的反馈
        
        Args:
            message_index: 消息索引
            rating: 评分 (1-5)
            messages: 所有消息
        """
        try:
            # 获取用户输入和模型回复
            user_input = ""
            model_response = ""
            
            # 查找对应的用户输入和模型回复
            if message_index > 0:
                user_msg_index = message_index - 1
                if user_msg_index < len(messages) and messages[user_msg_index]["role"] == "user":
                    user_input = messages[user_msg_index]["content"]
            
            if message_index < len(messages) and messages[message_index]["role"] == "assistant":
                model_response = messages[message_index]["content"]
            
            # 获取当前使用的模型名称
            current_model = st.session_state.get("selected_model", self.model_name)
            if current_model is None:
                current_model = "unknown_model"
            
            # 构造反馈数据
            feedback_data = {
                "session_id": st.session_state.get("session_id", str(datetime.now().timestamp())),
                "model_name": current_model,
                "user_input": user_input,
                "model_response": model_response,
                "rating": rating,
                "temperature": st.session_state.temperature,
                "top_p": st.session_state.top_p,
                "top_k": st.session_state.top_k,
                "max_tokens": st.session_state.max_tokens
            }
            
            # 保存反馈
            success = self.feedback_handler.save_feedback(feedback_data)
            if success:
                st.success("感谢您的反馈！")
                # 更新反馈状态
                st.session_state.current_feedback[message_index] = rating
                # 延迟刷新以显示成功消息
                time.sleep(1)
                st.rerun()
            else:
                st.error("反馈保存失败，请重试。")
                
        except Exception as e:
            st.error(f"保存反馈时出错: {e}")
    
    def save_chat_history(self):
        """保存聊天历史到文件"""
        if not st.session_state.chat_history:
            return
        
        try:
            # 创建保存目录
            save_dir = project_root / "data" / "chat_history"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"huanhuan_chat_{timestamp}.json"
            filepath = save_dir / filename
            
            # 保存数据
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
            
            st.success(f"对话历史已保存至: {filepath}")
        except Exception as e:
            st.error(f"保存对话历史失败: {e}")
    
    def load_chat_history(self):
        """从文件加载聊天历史"""
        try:
            # 查找最新的聊天历史文件
            chat_history_dir = project_root / "data" / "chat_history"
            if not chat_history_dir.exists():
                st.warning("未找到聊天历史目录")
                return
            
            # 获取所有聊天历史文件
            history_files = list(chat_history_dir.glob("*.json"))
            if not history_files:
                st.warning("未找到聊天历史文件")
                return
            
            # 选择最新的文件
            latest_file = max(history_files, key=os.path.getctime)
            
            # 加载数据
            with open(latest_file, 'r', encoding='utf-8') as f:
                chat_history = json.load(f)
            
            # 转换为消息格式
            messages = []
            for item in chat_history:
                messages.append({"role": "user", "content": item["user"]})
                messages.append({"role": "assistant", "content": item["assistant"]})
            
            # 更新状态
            st.session_state.messages = messages
            st.session_state.chat_history = chat_history
            
            st.success(f"已加载聊天历史: {latest_file.name}")
        except Exception as e:
            st.error(f"加载聊天历史失败: {e}")

def main():
    """主函数"""
    app = HuanHuanWebApp()
    
    # 渲染侧边栏
    app.render_sidebar()
    
    # 渲染主内容
    app.render_main_content()

if __name__ == "__main__":
    main()