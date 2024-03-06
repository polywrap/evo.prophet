import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from evo_researcher.functions.create_embeddings_from_results import create_embeddings_from_results
from evo_researcher.functions.generate_subqueries import generate_subqueries
from evo_researcher.functions.prepare_report import prepare_report, prepare_summary
from evo_researcher.models.WebScrapeResult import WebScrapeResult
from evo_researcher.functions.rerank_subqueries import rerank_subqueries
from evo_researcher.functions.scrape_results import scrape_results
from evo_researcher.functions.search import search

def research(
    goal: str,
    use_summaries: bool,
    model: str = "gpt-4-1106-preview",
    initial_subqueries_limit: int = 20,
    subqueries_limit: int = 4,
    scrape_content_split_chunk_size: int = 800,
    scrape_content_split_chunk_overlap: int = 225,
    top_k_per_query: int = 8,
    use_tavily_raw_content: bool = False,
) -> str:
    logging.info("Generating subqueries")
    queries = generate_subqueries(query=goal, limit=initial_subqueries_limit, model=model)
    
    logging.info("Reranking subqueries")
    queries = rerank_subqueries(queries=queries, goal=goal, model=model)[:subqueries_limit] if initial_subqueries_limit > subqueries_limit else queries

    stringified_queries = '\n- ' + '\n- '.join(queries)
    logging.info(f"Searching the web for: {stringified_queries}")
    search_results_with_queries = search(
        queries, 
        lambda result: not result.url.startswith("https://www.youtube")
    )

    if not search_results_with_queries:
        raise ValueError(f"No search results found for the goal {goal}.")

    scrape_args = [result for (_, result) in search_results_with_queries]
    websites_to_scrape = set([result.url for result in scrape_args])
    
    stringified_websites = '\n- ' + '\n- '.join(websites_to_scrape)
    logging.info(f"Scraping websites: {stringified_websites}")
    scraped = scrape_results(scrape_args) if not use_tavily_raw_content else [WebScrapeResult(
        query=result.query,
        url=result.url,
        title=result.title,
        content=result.raw_content,
    ) for result in scrape_args if result.raw_content]
    scraped = [result for result in scraped if result.content != ""]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "  "],
        chunk_size=scrape_content_split_chunk_size,
        chunk_overlap=scrape_content_split_chunk_overlap
    )
    
    logging.info("Creating embeddings")
    collection = create_embeddings_from_results(scraped, text_splitter)

    vector_result_texts: list[str] = []
    url_to_content_deemed_most_useful: dict[str, str] = {}

    stringified_queries = '\n- ' + '\n- '.join(queries)
    logging.info(f"Similarity searching for: {stringified_queries}")
    for query in queries:
        top_k_per_query_results = collection.similarity_search(query, k=top_k_per_query)
        vector_result_texts += [result.page_content for result in top_k_per_query_results if result.page_content not in vector_result_texts]

        for x in top_k_per_query_results:
            # `x.metadata["content"]` holds the whole url's web page, so it's ok to overwrite the value of the same url.
            url_to_content_deemed_most_useful[x.metadata["url"]] = x.metadata["content"]

    logging.info(f"Summarizing information")
    vector_result_texts = [
        prepare_summary(goal, content, "gpt-3.5-turbo-0125", trim_content_to_tokens=14_000)  # Hard code gpt-3.5-turbo-0125, because it would be very costly with gpt-4.
        for content in url_to_content_deemed_most_useful.values()
    ] if use_summaries else vector_result_texts

    logging.info(f"Preparing report")
    report = prepare_report(goal, vector_result_texts, model=model)

    return report
