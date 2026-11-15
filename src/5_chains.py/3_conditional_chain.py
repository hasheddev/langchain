# Branches based on rules
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
)
from pydantic import BaseModel, Field
from typing import Literal

from models import qwen_model, gemini_model

parser = StrOutputParser()


class FeedBack(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of user feedback must be either positive or negative"
    )


def is_positive(x: FeedBack) -> bool:
    return x.sentiment == "positive"


def is_negative(x: FeedBack) -> bool:
    return x.sentiment == "negative"


parser2 = PydanticOutputParser(pydantic_object=FeedBack)

user_message = """
"Extremely disappointed with the battery life"
I bought the NovaSound ANC Headphones last week after reading glowing reviews. The sound quality is honestly fantastic—deep bass and very clear highs, and the active noise cancellation works great on my daily subway commute. However, the battery life is atrocious.
The box claims 30 hours, but I barely get 8 hours on a full charge with ANC turned on. For a $250 pair of headphones, having to charge them twice a day is unacceptable. Customer support was polite when I called, but they couldn't offer a replacement without me
shipping them back at my own expense. I'll likely be returning these.
"""

prompt1 = PromptTemplate(
    template="classify the sentiment of the following text into positive or negative {feedback}\n{format}",
    input_variables=["feedback"],
    partial_variables={"format": parser2.get_format_instructions()},
)

classify = prompt1 | qwen_model | parser2

prompt2 = PromptTemplate(
    template="write an appropriate response to this positive feedback{feedback}",
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template="write an appropriate response to this negative feedback{feedback}",
    input_variables=["feedback"],
)

branch = RunnableBranch(
    (
        is_positive,
        RunnableLambda(
            lambda x: print(f"DEBUG - value of x(pos): {x}")
            or {"feedback": user_message}
        )
        | prompt2
        | gemini_model
        | parser,
    ),
    (
        is_negative,
        RunnableLambda(
            lambda x: print(f"DEBUG - value of x(neg): {x}")
            or {"feedback": user_message}
        )
        | prompt3
        | gemini_model
        | parser,
    ),
    (RunnableLambda(lambda x: "No valid sentiment found")),
)

chain = classify | branch

result = chain.invoke({"feedback": user_message})
print(result)
