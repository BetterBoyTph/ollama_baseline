
import streamlit as st
from feedback_handler import FeedbackHandler

# 初始化FeedbackHandler
feedback_handler = FeedbackHandler()

def main():
    st.title("Chat-嬛嬛")

    # 欢迎信息
    st.write("欢迎来到甄嬛传角色对话系统！我是甄嬛，大理寺少卿甄远道之女。臣妾愿与您畅谈宫廷生活、诗词歌赋，分享人生感悟。")

    # 角色信息输入框
    role_info = st.text_input("角色信息", "")

    # 示例问题
    st.subheader("示例问题")
    example_questions = [
        "你好，请介绍一下自己",
        "你觉得宫廷生活如何？",
        "如何看待友情？",
        "能为我作一首诗吗？",
        "给后人一些人生建议",
        "你最喜欢什么？"
    ]
    
    # 创建示例问题按钮
    cols = st.columns(3)
    for i, question in enumerate(example_questions):
        if cols[i % 3].button(question):
            # 处理示例问题点击事件
            st.write(f"你问了: {question}")
            # 这里应该调用模型获取回复
            # model_response = get_model_response(question)
            # st.write(model_response)

    # 对话历史
    st.subheader("对话历史")
    # 这里应该显示之前的对话记录
    # conversation_history = get_conversation_history()
    # for message in conversation_history:
    #     st.write(message)

    # 用户输入框
    user_input = st.text_input("请输入您的问题...", "")
    if st.button("提交"):
        if user_input:
            st.write(f"你问了: {user_input}")
            # 这里应该调用模型获取回复
            # model_response = get_model_response(user_input)
            # st.write(model_response)

            # 保存反馈（示例）
            feedback_data = {
                "user_input": user_input,
                "model_response": "这是模型的回复",  # 实际上应该是模型的真实回复
                "rating": 5,  # 示例评分
                "comment": "测试反馈"  # 示例评论
            }
            if feedback_handler.save_feedback(feedback_data):
                st.write("✅ 反馈保存成功")
            else:
                st.write("❌ 反馈保存失败")

if __name__ == "__main__":
    main()