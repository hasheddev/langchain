from langchain_community.document_loaders import PyPDFLoader

from pathlib import Path

p = Path(__file__).parent.parent
pdf = p / "data/sample.pdf"

loader = PyPDFLoader(pdf)
file = loader.load()
print(file[0].metadata)
