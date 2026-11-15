# Response with no structure
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models import qwen_model

template1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)

template2 = template = PromptTemplate(
    template="Write a five line summary on the following {text}",
    input_variables=["text"],
)

parser = StrOutputParser()

chain = (
    template1
    | qwen_model
    | parser
    | (lambda output: {"text": output})
    | template2
    | qwen_model
    | parser
)

response = chain.invoke({"topic": "Black Hole"})
print(response)
