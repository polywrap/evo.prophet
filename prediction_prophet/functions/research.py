import logging
import typing as t

from langchain.text_splitter import RecursiveCharacterTextSplitter
from prediction_prophet.functions.create_embeddings_from_results import create_embeddings_from_results
from prediction_prophet.functions.generate_subqueries import generate_subqueries
from prediction_prophet.functions.prepare_report import prepare_report, prepare_summary
from prediction_prophet.models.WebScrapeResult import WebScrapeResult
from prediction_prophet.functions.rerank_subqueries import rerank_subqueries
from prediction_prophet.functions.scrape_results import scrape_results
from prediction_prophet.functions.search import search
from pydantic.types import SecretStr
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from prediction_market_agent_tooling.tools.langfuse_ import observe

if t.TYPE_CHECKING:
    from loguru import Logger


class Research(BaseModel):
    report: str
    all_queries: list[str]
    reranked_queries: list[str]
    websites_to_scrape: list[str]
    websites_scraped: list[WebScrapeResult]


class NoResulsFoundError(ValueError):
    pass


class NotEnoughScrapedSitesError(ValueError):
    pass


@observe()
def research(
    goal: str,
    use_summaries: bool = False,
    agent: Agent | None = None,
    initial_subqueries_limit: int = 20,
    subqueries_limit: int = 4,
    max_results_per_search: int = 5,
    min_scraped_sites: int = 0,
    scrape_content_split_chunk_size: int = 800,
    scrape_content_split_chunk_overlap: int = 225,
    top_k_per_query: int = 8,
    use_tavily_raw_content: bool = False,
    openai_api_key: SecretStr | None = None,
    tavily_api_key: SecretStr | None = None,
    logger: t.Union[logging.Logger, "Logger"] = logging.getLogger(),
) -> Research:
    # Validate args
    if min_scraped_sites > max_results_per_search * subqueries_limit:
        raise ValueError(
            f"min_scraped_sites ({min_scraped_sites}) must be less than or "
            f"equal to max_results_per_search ({max_results_per_search}) * "
            f"subqueries_limit ({subqueries_limit})."
        )
    
    agent = agent or Agent("gpt-4o", model_settings=ModelSettings(temperature=0.7))

    logger.info("Started subqueries generation")
    all_queries = generate_subqueries(query=goal, limit=initial_subqueries_limit, agent=agent)
    
    stringified_queries = '\n- ' + '\n- '.join(all_queries)
    logger.info(f"Generated subqueries: {stringified_queries}")
    
    logger.info("Started subqueries reranking")
    queries = rerank_subqueries(queries=all_queries, goal=goal, agent=agent)[:subqueries_limit] if initial_subqueries_limit > subqueries_limit else all_queries

    stringified_queries = '\n- ' + '\n- '.join(queries)
    logger.info(f"Reranked subqueries. Will use top {subqueries_limit}: {stringified_queries}")
    
    logger.info(f"Started web searching")
    search_results_with_queries = search(
        queries,
        lambda result: not result.url.startswith("https://www.youtube"),
        tavily_api_key=tavily_api_key,
        max_results_per_search=max_results_per_search,
    )

    if not search_results_with_queries:
        raise NoResulsFoundError(f"No search results found for the goal {goal}.")

    scrape_args = [result for (_, result) in search_results_with_queries]
    websites_to_scrape = set(result.url for result in scrape_args)
    
    stringified_websites = '\n- ' + '\n- '.join(websites_to_scrape)
    logger.info(f"Found the following relevant results: {stringified_websites}")
    
    logger.info(f"Started scraping of web results")
    scraped = scrape_results(scrape_args) if not use_tavily_raw_content else [WebScrapeResult(
        query=result.query,
        url=result.url,
        title=result.title,
        content=result.raw_content,
    ) for result in scrape_args if result.raw_content]
    scraped = [result for result in scraped if result.content != ""]

    unique_scraped_websites = set([result.url for result in scraped])
    if len(scraped) < min_scraped_sites:
        # Get urls that were not scraped
        raise NotEnoughScrapedSitesError(
            f"Only successfully scraped content from "
            f"{len(unique_scraped_websites)} websites, out of a possible "
            f"{len(websites_to_scrape)} websites, which is less than the "
            f"minimum required ({min_scraped_sites}). The following websites "
            f"were not scraped: {websites_to_scrape - unique_scraped_websites}"
        )

    logger.info(f"Scraped content from {len(scraped)} websites")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "  "],
        chunk_size=scrape_content_split_chunk_size,
        chunk_overlap=scrape_content_split_chunk_overlap
    )
    
    logger.info("Started embeddings creation")
    collection = create_embeddings_from_results(scraped, text_splitter, api_key=openai_api_key)
    logger.info("Embeddings created")

    vector_result_texts: list[str] = []
    url_to_content_deemed_most_useful: dict[str, str] = {}

    stringified_queries = '\n- ' + '\n- '.join(queries)
    logger.info(f"Started similarity searches for: {stringified_queries}")
    for query in queries:
        top_k_per_query_results = collection.similarity_search(query, k=top_k_per_query)
        vector_result_texts += [result.page_content for result in top_k_per_query_results if result.page_content not in vector_result_texts]

        for x in top_k_per_query_results:
            # `x.metadata["content"]` holds the whole url's web page, so it's ok to overwrite the value of the same url.
            url_to_content_deemed_most_useful[x.metadata["url"]] = x.metadata["content"]
    
    stringified_urls = '\n- ' + '\n- '.join(url_to_content_deemed_most_useful.keys())
    logger.info(f"Found {len(vector_result_texts)} information chunks across the following sites: {stringified_urls}")

    if use_summaries:
        logger.info(f"Started summarizing information")
        vector_result_texts = [
            prepare_summary(
                goal,
                content,
                "gpt-3.5-turbo-0125",
                api_key=openai_api_key,
                trim_content_to_tokens=14_000,
            )
            for content in url_to_content_deemed_most_useful.values()
        ]
        logger.info(f"Information summarized")

    logger.info(f"Started preparing report")
    report = prepare_report(goal, vector_result_texts, agent=agent)
    logger.info(f"Report prepared")
    logger.info(report)

    return Research(
        all_queries=all_queries,
        reranked_queries=queries,
        report=report,
        websites_to_scrape=list(websites_to_scrape),
        websites_scraped=scraped,
    )
