from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from models import qwen_model

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a fictional person, the name and city has to be russian. {format}",
    input_variables=[],
    partial_variables={"format": parser.get_format_instructions()},
)

chain = template | qwen_model | parser
result = chain.invoke({})
print(result)
print(parser.get_format_instructions())
print(type(result))
