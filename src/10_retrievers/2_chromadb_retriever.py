from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = [
    Document(
        page_content="Large Language Models are advanced neural networks trained on vast amounts of text data."
    ),
    Document(
        page_content="Retrieval-Augmented Generation helps LLMs access external databases to reduce hallucinations."
    ),
    Document(
        page_content="Vector databases like Chroma store high-dimensional embeddings for fast semantic similarity search."
    ),
    Document(
        page_content="Fine-tuning adapts a pre-trained language model to specific downstream tasks and domains."
    ),
    Document(
        page_content="Attention mechanisms allow transformer-based models to weigh the significance of different words in context."
    ),
]

store = Chroma.from_documents(
    documents=docs, embedding=embeddings, collection_name="chroma_retriever_langchain"
)

query = "tell me more about llm"

retriever = store.as_retriever(search_kwargs={"k": 2})

result = retriever.invoke(query)

print(result)
