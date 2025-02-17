import json
import tiktoken
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from prediction_prophet.autonolas.research import clean_completion_json
from langchain.prompts import ChatPromptTemplate


QUESTION_REPHRASE_PROMPT = """We have the following question: `{question}`

Write a dictionary with following keys, don't answer the question, only rewrite it in the following ways:

```
- open_ended_question: Ask the question universally
- negated_question Ask the question in negation
```
"""


class RephrasedQuestion(BaseModel):
    original_question: str
    negated_question: str
    open_ended_question: str


def rephrase_question(
    question: str,
    engine: str = "gpt-4o"
) -> RephrasedQuestion:
    """
    Rephrase the original question, by asking it in negation and universally, for example:

    original_question: Is the sky blue?
    negated_question: Is the sky not blue?
    open_ended_question: What is the color of the sky?
    """
    tokenizer = tiktoken.encoding_for_model(engine)
    llm = ChatOpenAI(model=engine, temperature=0.0)

    prompt = ChatPromptTemplate.from_template(template=QUESTION_REPHRASE_PROMPT)
    messages = prompt.format_messages(question=question)

    max_tokens = 2 * len(tokenizer.encode(question)) + 50 # Max tokens as the question two times + some buffer for formatting.
    completion = str(llm(messages, max_tokens=max_tokens).content)

    try:
        return RephrasedQuestion(
            original_question=question, 
            **json.loads(clean_completion_json(completion))
        )
    except json.decoder.JSONDecodeError as e:
        raise ValueError(f"Error in rephrase_question for `{question}`: {completion}") from e
        
