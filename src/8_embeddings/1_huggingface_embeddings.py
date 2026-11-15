from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "New york is a busy city with a lot of population"

result = embeddings.embed_query(text)
print(result)
