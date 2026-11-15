from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()


class Review(TypedDict):
    summary: str
    sentiment: str


model = ChatGroq(model="qwen/qwen3.8-27b")

prompt = """
"Extremely disappointed with the battery life"
I bought the NovaSound ANC Headphones last week after reading glowing reviews. The sound quality is honestly fantastic—deep bass and very clear highs, and the active noise cancellation works great on my daily subway commute. However, the battery life is atrocious.
The box claims 30 hours, but I barely get 8 hours on a full charge with ANC turned on. For a $250 pair of headphones, having to charge them twice a day is unacceptable. Customer support was polite when I called, but they couldn't offer a replacement without me
shipping them back at my own expense. I'll likely be returning these.
"""
structured_model = model.with_structured_output(Review)  # type: ignore[arg-type]
response = structured_model.invoke(prompt)
print(response)
