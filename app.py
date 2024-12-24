import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.document_loaders import SitemapLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import os

# Streamlit app title
st.title("Cloudflare Documentation Chatbot")

# Sidebar inputs for API key and GitHub repo link
st.sidebar.header("Configuration")
openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key:", type="password"
)
st.sidebar.markdown("[GitHub Repo](https://github.com/your-repo-link)")

if not openai_api_key:
    st.warning("Please enter your OpenAI API Key to proceed.")
    st.stop()

# Constants
SITEMAPS = [
    "https://developers.cloudflare.com/ai-gateway/sitemap.xml",
    "https://developers.cloudflare.com/vectorize/sitemap.xml",
    "https://developers.cloudflare.com/workers-ai/sitemap.xml",
]

@st.cache_resource
def load_and_embed_documents():
    """Loads documents from sitemaps, creates embeddings, and stores them in a vectorstore."""
    loader = SitemapLoader(web_paths=SITEMAPS)
    docs = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Load documents and create vectorstore
vectorstore = load_and_embed_documents()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Initialize the chat model
chat_model = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)
qa_chain = ConversationalRetrievalChain.from_llm(chat_model, retriever=retriever)

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Chat interface
st.header("Cloudflare Docs Chatbot")
user_input = st.text_input("Ask a question about Cloudflare's documentation:")
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response
    response = qa_chain.run({"question": user_input, "chat_history": st.session_state.messages})

    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")
