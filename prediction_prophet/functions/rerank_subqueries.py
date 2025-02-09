import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from pydantic.types import SecretStr
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import secretstr_to_v1_secretstr
from prediction_market_agent_tooling.tools.langfuse_ import get_langfuse_langchain_config, observe

rerank_queries_template = """
I will present you with a list of queries to search the web for, for answers to the question: {goal}.

The queries are divided by '---query---'

Evaluate the queries in order that will provide the best data to answer the question. Do not modify the queries.
Return them, in order of relevance, as a comma separated list of strings.

Queries: {queries}
"""
@observe()
def rerank_subqueries(queries: list[str], goal: str, model: str, temperature: float, api_key: SecretStr | None = None) -> list[str]:
    if api_key == None:
        api_key = APIKeys().openai_api_key
            
    rerank_results_prompt = ChatPromptTemplate.from_template(template=rerank_queries_template)

    rerank_results_chain = (
        rerank_results_prompt |
        ChatOpenAI(model=model, temperature=temperature, api_key=secretstr_to_v1_secretstr(api_key)) |
        StrOutputParser()
    )

    responses: str = rerank_results_chain.invoke({
        "goal": goal,
        "queries": "\n---query---\n".join(queries)
    }, config=get_langfuse_langchain_config())

    return responses.split(",")