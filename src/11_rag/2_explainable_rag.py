from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

search = store.similarity_search(query=query, k=3)

prompt = PromptTemplate(
    template="""
You are an AI assistant. Use the following context to answer the question.
Also provide actual chunk along with the page number of the document where the chunk can be found
If the answer is not present, say you do not know politely.
context: ```{context}```

question: ```{question}```
""",
    input_variables=["question", "context"],
)

parser = StrOutputParser()


def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", 0) + 1
        formatted.append(f"--- Chunk {i} (Page {page}) ---\n{doc.page_content}")
    return "\n\n".join(formatted)


chain = (
    {"context": retriever | format_docs, "question": lambda x: x}
    | prompt
    | qwen_model
    | parser
)


response = chain.invoke(query)
print(response)
