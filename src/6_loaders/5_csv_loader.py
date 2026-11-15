from langchain_community.document_loaders import CSVLoader

from pathlib import Path

p = Path(__file__).parent.parent
csv = p / "data/customers.csv"

loader = CSVLoader(csv)
file = loader.load()
print(file[0].metadata)
print(type(file[0]))
# print(file[0])
