from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "Large Language Models are advanced neural networks trained on vast amounts of text data.",
    "Retrieval-Augmented Generation helps LLMs access external databases to reduce hallucinations.",
    "Vector databases like Chroma store high-dimensional embeddings for fast semantic similarity search.",
    "Fine-tuning adapts a pre-trained language model to specific downstream tasks and domains.",
    "Attention mechanisms allow transformer-based models to weigh the significance of different words in context.",
]

store = FAISS.from_texts(texts=texts, embedding=embeddings)

query = "tell me more about llm"

result = store.similarity_search(query, k=2)

print(result)
