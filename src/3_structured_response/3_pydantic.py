from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal

load_dotenv()


class Schema(BaseModel):
    key_themes: list[str] = Field(
        description="write down the key themes discussed in the review in a list"
    )
    summary: str = Field(description="must write down a brief summary of the review")
    sentiment: Literal["positive", "negative"] = Field(
        description="Return sentiment of the review"
    )
    pros: Optional[list[str]] = Field(None, description="Write down the pros in a list")
    cons: Optional[list[str]] = Field(None, description="Write down the cons in a")


model = ChatGroq(model="qwen/qwen3.8-27b")

prompt = """
"Extremely disappointed with the battery life"
I bought the NovaSound ANC Headphones last week after reading glowing reviews. The sound quality is honestly fantastic—deep bass and very clear highs, and the active noise cancellation works great on my daily subway commute. However, the battery life is atrocious.
The box claims 30 hours, but I barely get 8 hours on a full charge with ANC turned on. For a $250 pair of headphones, having to charge them twice a day is unacceptable. Customer support was polite when I called, but they couldn't offer a replacement without me
shipping them back at my own expense. I'll likely be returning these.
"""

structured_model = model.with_structured_output(Schema)
response = structured_model.invoke(prompt)
print(response)
print(type(response))
