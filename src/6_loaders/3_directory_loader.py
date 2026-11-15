from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

from pathlib import Path

p = Path(__file__).parent.parent
dir = p / "data/"

loader = DirectoryLoader(path=str(dir.resolve()), loader_cls=PyPDFLoader, glob="*.pdf")

# docs = loader.load()
# print(docs[0].metadata)

for doc in loader.lazy_load():
    print(doc.metadata)
