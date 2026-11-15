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
    documents=docs, embedding=embeddings, collection_name="basic_rag_collection"
)

retriever = store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 5,
        "lambda_mult": 0.45,
    },
)

prompt = PromptTemplate(
    template="""
You are an AI assistant. Use the following context to answer the question.
If the answer is not present, say you do not know politely.
context: ```{context}```

question: ```{question}```
""",
    input_variables=["question", "context"],
)

parser = StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": lambda x: x}
    | prompt
    | qwen_model
    | parser
)

query = "Is the audit considered good or bad"

response = chain.invoke(query)
print(response)
