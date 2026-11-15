# unstructured data ocr(optical code reading) does not provide semantics
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

from pathlib import Path

p = Path(__file__).parent.parent
pdf = p / "data/sample.pdf"

loader = PyPDFLoader(pdf)
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="")
result = splitter.split_documents([docs[0]])
# print(result[0])
print(result[0].metadata)
print(len(result[0].page_content))
