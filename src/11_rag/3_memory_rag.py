from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from models import qwen_model

pdf_path = Path(__file__).parent.parent / "data/audit.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

docs = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

store = Chroma.from_documents(
    documents=docs, embedding=embeddings, collection_name="source_aware_rag_collection"
)

retriever = store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 5,
        "lambda_mult": 0.45,
    },
)

query = "What is the audit about"


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="""
You are an AI assistant with access to a document context and a chat history.
- For questions about the audit document, use the provided context and state the chunk and page number.
- For questions about the conversation itself (e.g., past messages, who spoke first, user details), use the chat history provided in the history placeholder.
- If the answer is not present in either source, say you do not know politely.
"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "\ncontext:\n{context}\n\nquestion:\n{input}\n"),
    ]
)


def conversational_rag(user_input: str, chat_history: list[BaseMessage]):
    docs = retriever.invoke(user_input)

    context = "\n\n".join(
        f"--- Chunk (Page {doc.metadata.get('page', 'N/A')}) ---\n{doc.page_content}"
        for doc in docs
    )

    messages = prompt.invoke(
        {"context": context, "input": user_input, "history": chat_history}
    )

    response = qwen_model.invoke(messages)
    return response


chat_history: list[BaseMessage] = []

print("Conversational: RAG:  ")
while True:
    user_input = input("you: ")
    if user_input.lower() == "exit":
        break

    response = conversational_rag(user_input, chat_history)
    print(f"AI: {response.content}")

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))
