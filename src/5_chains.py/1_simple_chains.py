# single input single output
# input -> prompt -> model => parser -> output   like a conveyor belt
# great for summarization, text generation, extraction (single step operations)
# IT IS COMPOSABLE, parts can be changed but chain behaves the same, it is easily debuggable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models import gemini_model

parser = StrOutputParser()
template = PromptTemplate(
    template="Generate three facts about {topic}", input_variables=["topic"]
)

chain = template | gemini_model | parser

response = chain.invoke({"topic": "aliens"})
print(response)
