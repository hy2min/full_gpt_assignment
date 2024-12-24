import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# Streamlit 페이지 설정
st.set_page_config(page_title="DocumentGPT", page_icon="📜")

# Chat Callback Handler
class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

# LLM 초기화
llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

# 파일 임베딩 함수
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.read())

    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()

# 메시지 관리 함수
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for msg in st.session_state["messages"]:
        send_message(msg["message"], msg["role"], save=False)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 초기 세션 상태 설정
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Prompt 설정
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
        
        Context: {context}
        """,
    ),
    ("human", "{question}"),
])

# Streamlit UI 구성
st.title("DocumentGPT")
st.markdown(
    """
    Welcome!

    Use this chatbot to ask questions about your files! Upload a file in the sidebar to get started.
    """
)

# 사이드바 구성
with st.sidebar:
    st.header("Upload Your File")
    file = st.file_uploader("Upload a .txt, .pdf, or .docx file", type=["pdf", "txt", "docx"])

# 파일 업로드 처리
if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()

    # 채팅 입력 처리
    user_input = st.chat_input("Ask anything about your file...")
    if user_input:
        send_message(user_input, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(user_input)
else:
    st.info("Please upload a file to start.")
