from langchain_core.prompts import PromptTemplate

static_prompt = PromptTemplate(
    input_variables=[], template="Write a short fun fact about AI"
)

prompt = static_prompt.format()
print(prompt)
