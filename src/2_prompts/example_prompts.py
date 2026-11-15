from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Setup chat template containing a placeholder for history
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Dynamic history injection slot
        ("human", "{user_input}"),
    ]
)

# Simulate prior conversation history
history = [
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi there! How can I help you today?"),
]

# Invoke the template with history and new input
prompt_value = chat_template.invoke(
    {"chat_history": history, "user_input": "What is the capital of France?"}
)

print(prompt_value)
