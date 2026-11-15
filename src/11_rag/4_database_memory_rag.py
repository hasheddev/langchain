import streamlit as st
import sqlite3
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pathlib import Path

from models import gemini_model

pdf_path = Path(__file__).parent.parent / "data/audit.pdf"

conn = sqlite3.connect("chat_history.db", check_same_thread=False)

cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history(
        session_id TEXT,
        role TEXT,
        content TEXT
    )
""")
conn.commit()


def save_chat_message(session_id: str, role: str, content: str):
    cursor.execute(
        """
        INSERT INTO chat_history VALUES(?, ?, ?)
""",
        (session_id, role, content),
    )


def load_chat_message(session_id: str):
    cursor.execute(
        """
        SELECT role, content from chat_history WHERE session_id = ?
""",
        (session_id,),
    )
    rows = cursor.fetchall()
    history: list[BaseMessage] = []
    for role, content in rows:
        if role == "human":
            history.append(HumanMessage(content=content))
        elif role == "ai":
            history.append(AIMessage(content=content))
    return history


def get_all_sessions():
    cursor.execute(
        """
            SELECT DISTINCT session_id from chat_history ORDER BY rowid desc
    """,
    )
    return [row[0] for row in cursor.fetchall()]


st.set_page_config(page_title="Conversational RAG", layout="wide")
st.title("Conversational RAG with memory")

st.sidebar.title("Chats")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []

if st.sidebar.button("New Chat"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []

st.sidebar.markdown("Previous conversations")

for sid in get_all_sessions():
    if st.sidebar.button(sid[:8]):
        st.session_state.session_id = sid
        st.session_state.chat_history = load_chat_message(sid)

session_id = st.session_state.session_id


@st.cache_resource
def load_vector_store():
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_store


vector_store = load_vector_store()
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 4,
    },
)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a helpful AI Assistant. Answer strictly from the provided context. If the"
            "answer is not present, just say you do not know"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context: {context}, question: {input}"),
    ],
)


def conversation_rag(user_input: str, chat_history: list[BaseMessage]):
    docs = retriever.invoke(user_input)
    context = "\n\n".join(
        f'[Page {d.metadata.get("page", "N/A")}]\n{d.page_content}' for d in docs
    )
    messages = prompt.invoke(
        {"input": user_input, "context": context, "chat_history": chat_history}
    )
    response = gemini_model.invoke(messages)
    return response, docs


if not st.session_state.chat_history:
    st.session_state.chat_history = load_chat_message(session_id)

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("AI").write(msg.content)

user_input = st.chat_input("Ask a question from a PDF")
if user_input:
    st.chat_message("user").write(user_input)
    save_chat_message(session_id, "human", user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    response, docs = conversation_rag(user_input, st.session_state.chat_history)
    st.chat_message("AI").write(response.content)
    save_chat_message(session_id, "ai", response.content)  # type: ignore
    st.session_state.chat_history.append(AIMessage(content=response.content))
