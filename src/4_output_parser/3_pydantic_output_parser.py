from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from models import qwen_model


class Schema(BaseModel):
    name: str = Field(description="The persons full name")
    age: int = Field(
        description="The person's age must not be less than 18 or greater than 60"
    )
    city: str = Field(description="The city where the person lives")


parser = PydanticOutputParser(pydantic_object=Schema)

template = PromptTemplate(
    template="Give me the name, age and city of a fictional {place} person, make sure the age is greater than 18 and return the following format:\n\n {format}\n",
    input_variables=["place"],
    partial_variables={"format": parser.get_format_instructions()},
)

chain = template | qwen_model | parser
result = chain.invoke({"place": "chinese"})
print("\n\n-----------------------------------")
print(result)
print("\n\n-----------------------------------")
print(parser.get_format_instructions())
print("\n\n-----------------------------------")
print(type(result))
