# Multiple steps one after the other. combines multiple simple chains connected to one another
# shape -> refine-> polish -> finishing details
# e.g extract data, clean data, summarize. Turning points into content from outline. Multistage reasoning with isolation to avoid context bloat
# It gives a predictable and reliable flow and debuggable flow, composable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from models import qwen_model

model = qwen_model

prompt1 = PromptTemplate(
    template="Generate detailed report on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a three point summary of the following text {text}",
    input_variables=["text"],
)

parser = StrOutputParser()

chain = (
    prompt1
    | model
    | parser
    | {"text": RunnablePassthrough()}
    | prompt2
    | model
    | parser
)


response = chain.invoke({"topic": "aliens"})
print(response)
