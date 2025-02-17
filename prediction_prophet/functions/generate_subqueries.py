from langchain.output_parsers import CommaSeparatedListOutputParser
from pydantic_ai import Agent
from prediction_market_agent_tooling.tools.langfuse_ import observe


subquery_generation_template = """You are a professional researcher. Your goal is to prepare a research plan for {query}.

The plan will consist of multiple web searches separated by commas.
Return ONLY the web searches, separated by commas and without quotes.

Limit your searches to {search_limit}.
"""
@observe()
def generate_subqueries(query: str, limit: int, agent: Agent) -> list[str]:
    if limit == 0:
        return [query]

    result = agent.run_sync(subquery_generation_template.format(query=query, search_limit=limit))
    subqueries = CommaSeparatedListOutputParser().parse(result.data)

    return [query] + [subquery.strip('\"') for subquery in subqueries]
