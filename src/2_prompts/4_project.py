from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_version="v1beta")

print("Blog post generator")
print("Provide ideas or topics for the blog post. Type (exit) to finish")
topic = input("Enter blog post topic: ")

template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are a professional blog writer. Help generate informative, engaging and well"
            "Structured blog post about a {topic}."
        ),
        HumanMessagePromptTemplate.from_template(
            "Write a detailed blog post about {topic}"
        ),
    ]
)

history = []

while True:
    user_input = input("ideas or instruction or type exit:  ")
    if user_input.lower() == "exit":
        break
    messages = template.format_messages(topic=topic)
    for message in history:
        messages.append(message)

    user_message = HumanMessagePromptTemplate.from_template(user_input).format_messages(
        user_input=user_input
    )[0]
    messages.append(user_message)
    response = gemini_model.invoke(messages)
    print("Blog Post content \n", response.content)

    history.append(
        AIMessagePromptTemplate.from_template(response.content).format_messages()[0]
    )
