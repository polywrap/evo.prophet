from pydantic_ai import Agent
from prediction_market_agent_tooling.tools.langfuse_ import observe
from langchain.output_parsers import CommaSeparatedListOutputParser

rerank_queries_template = """I will present you with a list of queries to search the web for, for answers to the question: {goal}.

The queries are divided by '---query---'

Evaluate the queries in order that will provide the best data to answer the question. Do not modify the queries.
Return them, in order of relevance, as a comma separated list of strings.

Queries: {queries}
"""
@observe()
def rerank_subqueries(queries: list[str], goal: str, agent: Agent) -> list[str]:
    result = agent.run_sync(rerank_queries_template.format(goal=goal, queries="\n---query---\n".join(queries)))
    responses = CommaSeparatedListOutputParser().parse(result.data)
    return responses
