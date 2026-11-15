from langchain_community.document_loaders import TextLoader

from pathlib import Path

p = Path(__file__).parent.parent
text = p / "data/state_of_the_union.txt"

loader = TextLoader(text, encoding="utf-8")
docs = loader.load()
print(type(docs))

print(docs[0].metadata)
print(docs[0].page_content)
