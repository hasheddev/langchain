from langchain_core.prompts import PromptTemplate

static_prompt = PromptTemplate(
    input_variables=["topic", "style"],
    template="Write a short paragraph about {topic} in a {style} style",
)

prompt = static_prompt.format(topic="AI", style="humorous")
print(prompt)
