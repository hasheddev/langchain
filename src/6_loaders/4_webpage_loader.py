from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from models import qwen_model

prompt = PromptTemplate(
    template="answer the following {question} from the following text {text}",
    input_variables=["text", "question"],
)

url = "https://en.wikipedia.org/wiki/Support_Vector_Machine"

loader = WebBaseLoader(url)

docs = loader.load()
parser = StrOutputParser()

chain = prompt | qwen_model | parser

result = chain.invoke(
    {"question": "What is the page talking about", "text": docs[0].page_content[:500]}
)

print(result)
