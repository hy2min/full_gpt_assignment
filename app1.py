import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Set the page config first
st.set_page_config(
    page_title="SiteGPT - Cloudflare Docs",
    page_icon="🖥️",
)

# Predefined URL for Cloudflare's documentation sitemap
cloudflare_sitemap_url = "https://developers.cloudflare.com/sitemap.xml"

# Initialize OpenAI model
llm = ChatOpenAI(
    temperature=0.1,
)

# Define the answer prompt template
answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  

    Context: {context}
    Examples:
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


# App description and UI
st.markdown(
    """
    # SiteGPT
    Ask questions about Cloudflare documentation.
    Start by exploring Cloudflare’s products through their documentation.
"""
)

# Load Cloudflare documentation using the predefined sitemap URL
retriever = load_website(cloudflare_sitemap_url)

# User input for query
query = st.text_input("Ask a question to the Cloudflare documentation:")
if query:
    chain = (
        {
            "docs": retriever,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
    )
    result = chain.invoke(query)
    st.markdown(result.content.replace("$", "\$"))

# Sidebar to accept OpenAI API Key from the user
with st.sidebar:
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    if openai_api_key:
        # Set API key for OpenAI integration
        import openai
        openai.api_key = openai_api_key
        st.success("OpenAI API Key set successfully!")

    st.markdown("### GitHub Repository")
    st.markdown("[View the code on GitHub](https://github.com/your-repository-link)")
