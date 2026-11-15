from langchain_community.retrievers import WikipediaRetriever
import wikipedia

wikipedia.set_user_agent("MyLangChainApp/1.0 (your_email@example.com)")

retriever = WikipediaRetriever(lang="en", top_k_results=2)
docs = retriever.invoke("Artificial Intelligence")
print(docs[0])
