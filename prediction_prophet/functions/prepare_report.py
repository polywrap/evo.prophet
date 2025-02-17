import typing as t
from pydantic_ai import Agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from prediction_prophet.functions.utils import trim_to_n_tokens
from prediction_market_agent_tooling.config import APIKeys
from pydantic.types import SecretStr
from prediction_market_agent_tooling.gtypes import secretstr_to_v1_secretstr
from prediction_market_agent_tooling.tools.langfuse_ import get_langfuse_langchain_config, observe

@observe()
def prepare_summary(goal: str, content: str, model: str, api_key: SecretStr | None = None, trim_content_to_tokens: t.Optional[int] = None) -> str:
    if api_key == None:
        api_key = APIKeys().openai_api_key
    
    prompt_template = """Write comprehensive summary of the following web content, that provides relevant information to answer the question: '{goal}'.
But cut the fluff and keep it up to the point.
Write in bullet points.
    
Content:

{content}
"""
    content = trim_to_n_tokens(content, trim_content_to_tokens, model) if trim_content_to_tokens else content
    evaluation_prompt = ChatPromptTemplate.from_template(template=prompt_template)

    research_evaluation_chain = (
        evaluation_prompt |
        ChatOpenAI(model=model, api_key=secretstr_to_v1_secretstr(api_key)) |
        StrOutputParser()
    )

    response: str = research_evaluation_chain.invoke({
        "goal": goal,
        "content": content
    }, config=get_langfuse_langchain_config())

    return response


@observe()
def prepare_report(goal: str, scraped: list[str], agent: Agent) -> str:
    evaluation_prompt_template = """You are a professional researcher. Your goal is to provide a relevant information report
    in order to make an informed prediction for the question: '{goal}'.
    
    Here are the results of relevant web searches:
    
    {search_results}
    
    Prepare a full comprehensive report that provides relevant information to answer the aforementioned question.
    If that is not possible, state why.
    You will structure your report in the following sections:
    
    - Introduction
    - Background
    - Findings and Analysis
    - Conclusion
    - Caveats
    
    Don't limit yourself to just stating each finding; provide a thorough, full and comprehensive analysis of each finding.
    Use markdown syntax. Include as much relevant information as possible and try not to summarize.
    """
    result = agent.run_sync(evaluation_prompt_template.format(goal=goal, search_results="\n".join(scraped)))
    response = result.data

    return response
