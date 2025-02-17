import time
from prediction_prophet.benchmark.agents import _make_prediction
from prediction_market_agent_tooling.tools.is_predictable import is_predictable_binary
from pydantic import SecretStr
import streamlit as st
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName
from pydantic_ai.settings import ModelSettings
from prediction_market_agent_tooling.tools.utils import secret_str_from_env
from langchain.text_splitter import RecursiveCharacterTextSplitter
from prediction_prophet.functions.create_embeddings_from_results import create_embeddings_from_results
from prediction_prophet.functions.generate_subqueries import generate_subqueries
from prediction_prophet.functions.prepare_report import prepare_report
from prediction_prophet.functions.rerank_subqueries import rerank_subqueries
from prediction_prophet.functions.scrape_results import scrape_results
from prediction_prophet.functions.search import search

def research(
    goal: str,
    tavily_api_key: SecretStr,
    model: KnownModelName = "gpt-4o",
    temperature: float = 0.7, 
    initial_subqueries_limit: int = 20,
    subqueries_limit: int = 4,
    scrape_content_split_chunk_size: int = 800,
    scrape_content_split_chunk_overlap: int = 225,
    top_k_per_query: int = 8
) -> str:
    agent = Agent(model=model, model_settings=ModelSettings(temperature=temperature))

    with st.status("Generating subqueries"):
        queries = generate_subqueries(query=goal, limit=initial_subqueries_limit, agent=agent)
    
        stringified_queries = '\n- ' + '\n- '.join(queries)
        st.write(f"Generated subqueries: {stringified_queries}")
        
    with st.status("Reranking subqueries"):
        queries = rerank_subqueries(queries=queries, goal=goal, agent=agent)[:subqueries_limit] if initial_subqueries_limit > subqueries_limit else queries

        stringified_queries = '\n- ' + '\n- '.join(queries)
        st.write(f"Reranked subqueries. Will use top {subqueries_limit}: {stringified_queries}")
    
    with st.status("Searching the web"):
        search_results_with_queries = search(
            queries, 
            lambda result: not result.url.startswith("https://www.youtube"),
            tavily_api_key=tavily_api_key
        )

        if not search_results_with_queries:
            raise ValueError(f"No search results found for the goal {goal}.")

        scrape_args = [result for (_, result) in search_results_with_queries]
        websites_to_scrape = set([result.url for result in scrape_args])
        
        stringified_websites = '\n- ' + '\n- '.join(websites_to_scrape)
        st.write(f"Found the following relevant results: {stringified_websites}")
    
    with st.status(f"Scraping web results"):
        scraped = scrape_results(scrape_args)
        scraped = [result for result in scraped if result.content != ""]
        
        st.write(f"Scraped content from {len(scraped)} websites")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "  "],
        chunk_size=scrape_content_split_chunk_size,
        chunk_overlap=scrape_content_split_chunk_overlap
    )
    
    with st.status(f"Performing similarity searches"):
        collection = create_embeddings_from_results(scraped, text_splitter)
        st.write("Created embeddings")

        vector_result_texts: list[str] = []
        url_to_content_deemed_most_useful: dict[str, str] = {}
        
        for query in queries:
            top_k_per_query_results = collection.similarity_search(query, k=top_k_per_query)
            vector_result_texts += [result.page_content for result in top_k_per_query_results if result.page_content not in vector_result_texts]

            for x in top_k_per_query_results:
                url_to_content_deemed_most_useful[x.metadata["url"]] = x.metadata["content"]
        
            st.write(f"Similarity searched for: {query}")

        st.write(f"Found {len(vector_result_texts)} relevant information chunks")

    with st.status(f"Preparing report"):
        report = prepare_report(goal, vector_result_texts, agent=agent)
        st.markdown(report)

    return report

tavily_api_key = secret_str_from_env('TAVILY_API_KEY')

if tavily_api_key == None:
    st.container().error("No Tavily API Key provided")
    st.stop()

st.set_page_config(layout="wide")
st.title("Prediction Prophet")
st.write('Ask any yes-or-no question about a future outcome')

with st.sidebar:
    st.title('Prediction Prophet')
    st.markdown("A web3 agent by [Agentcoin](https://www.agentcoin.tv/)")
    st.image('https://raw.githubusercontent.com/agentcoinorg/predictionprophet/main/docs/imgs/banner.png')
    
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    st.markdown('-------')
    st.caption('View the source code on our [github](https://github.com/agentcoinorg/predictionprophet)')
    st.caption('Learn more on our [substack](https://www.agentcoin.tv/blog/prediction-prophet)')
    st.caption('Join our [discord](https://discord.com/invite/6gk85fetcT)')
    

# TODO: find a better way to clear the history
progress_placeholder = st.empty()

if question := st.chat_input(placeholder='Will Twitter implement a new misinformation policy before the end of 2024?'):
    progress_placeholder.empty()
    time.sleep(0.1) # https://github.com/streamlit/streamlit/issues/5044
    
    with progress_placeholder.container():
        st.chat_message("user").write(question)
        
        with st.chat_message("assistant"):
            st.write(f"I will evaluate the probability of '{question}' occurring")
            
            with st.status("Evaluating question") as status:
                is_predictable = is_predictable_binary(question=question) 
                if not is_predictable:
                    st.container().error(f"The agent thinks this question is not predictable.")
                    status.update(label="Error evaluating question", state="error", expanded=True)
                    st.stop()
            
            report = research(
                goal=question,
                subqueries_limit=6,
                top_k_per_query=15,
                tavily_api_key=tavily_api_key,
            )
                    
            with st.status("Making prediction"):
                prediction = _make_prediction(market_question=question, additional_information=report, agent=Agent("gpt-4o", model_settings=ModelSettings(temperature=0.0)))

                if prediction.outcome_prediction == None:
                    st.container().error("The agent failed to generate a prediction")
                    st.stop()
                
                outcome_prediction = prediction.outcome_prediction
            
                st.write(f"Probability: {outcome_prediction.p_yes * 100}%. Confidence: {outcome_prediction.confidence * 100}%")
                if outcome_prediction.reasoning:
                    st.write(f"Reasoning: {outcome_prediction.reasoning}")
                if not prediction:
                    st.container().error("No prediction was generated.")
                    st.stop()
                    
            st.markdown(f"With **{outcome_prediction.confidence * 100}% confidence**, I'd say this outcome has a **{outcome_prediction.p_yes * 100}% probability** of happening")

