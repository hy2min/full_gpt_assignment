import streamlit as st
import openai

# Streamlit app 초기화
st.set_page_config(page_title="OpenAI Assistant", layout="wide")

# 사이드바 구성
st.sidebar.header("OpenAI Assistant")
st.sidebar.write("이 앱은 OpenAI Assistant 기능을 활용하여 대화형 에이전트를 제공합니다.")

# OpenAI API 키 입력
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key

# GitHub 리포지토리 링크
st.sidebar.markdown("[📂 GitHub Repository](https://github.com/hy2min)")

# 세션 상태 초기화
def init_session():
    if "assistant" not in st.session_state:
        st.session_state.assistant = None
    if "thread" not in st.session_state:
        st.session_state.thread = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

init_session()

# OpenAI Assistant 생성 함수
def create_assistant():
    if not st.session_state.assistant:
        st.session_state.assistant = openai.Assistant()

# Thread 생성 함수
def create_thread():
    if not st.session_state.thread:
        st.session_state.thread = st.session_state.assistant.create_thread()

# 메시지 처리 함수
def send_message(message):
    if st.session_state.thread:
        st.session_state.messages.append({"role": "user", "content": message})
        response = st.session_state.thread.run(message)
        st.session_state.messages.append({"role": "assistant", "content": response})

# 대화 기록 UI 렌더링
def render_chat():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.write(f"**You:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.write(f"**Assistant:** {msg['content']}")

# API 키가 있을 때만 Assistant와 Thread 생성
if api_key:
    create_assistant()
    create_thread()

    # 사용자 입력 받기
    user_input = st.text_input("Send a message:")
    if st.button("Send") and user_input:
        send_message(user_input)
        st.experimental_rerun()

# 대화 기록 렌더링
st.header("Chat History")
render_chat()
