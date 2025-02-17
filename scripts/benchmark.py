import typing as t

import typer
from prediction_market_agent_tooling.benchmark.agents import FixedAgent, RandomAgent
from prediction_market_agent_tooling.benchmark.benchmark import Benchmarker
from prediction_market_agent_tooling.markets.markets import MarketType, get_binary_markets, FilterBy, SortBy

from pydantic_ai import Agent
from prediction_prophet.autonolas.research import EmbeddingModel
from prediction_prophet.benchmark.agents import PredictionProphetAgent, OlasAgent, QuestionOnlyAgent
from prediction_market_agent_tooling.config import APIKeys


def main(
    n: int = 10,
    output: str = "./benchmark_report.md",
    reference: MarketType = MarketType.MANIFOLD,
    filter: FilterBy = FilterBy.OPEN,
    sort: SortBy = SortBy.NONE,
    max_workers: int = 1,
    cache_path: t.Optional[str] = "predictions_cache.json",
    only_cached: bool = False,
) -> None:
    """
    Polymarket usually contains higher quality questions, 
    but on Manifold, additionally to filtering by MarketFilter.resolved, you can sort by MarketSort.newest.
    """
    markets = get_binary_markets(n, reference, filter_by=filter, sort_by=sort)
    markets_deduplicated = list(({m.question: m for m in markets}.values()))
    if len(markets) != len(markets_deduplicated):
        print(
            f"Warning: Deduplicated markets from {len(markets)} to {len(markets_deduplicated)}."
        )

    print(f"Found {len(markets_deduplicated)} markets.")

    benchmarker = Benchmarker(
        markets=markets_deduplicated,
        agents=[
            RandomAgent(agent_name="random", max_workers=max_workers),
            FixedAgent(
                fixed_answer=False, agent_name="fixed-no", max_workers=max_workers
            ),
            FixedAgent(
                fixed_answer=True, agent_name="fixed-yes", max_workers=max_workers
            ),
            QuestionOnlyAgent(
                agent=Agent("gpt-3.5-turbo-0125"),
                agent_name="question-only_gpt-3.5-turbo-0125",
                max_workers=max_workers,
            ),
            OlasAgent(
                research_agent=Agent("gpt-3.5-turbo-0125"),
                prediction_agent=Agent("gpt-3.5-turbo-0125"),
                max_workers=max_workers,
                agent_name="olas_gpt-3.5-turbo-0125",
            ),
            OlasAgent(
                research_agent=Agent("gpt-3.5-turbo-0125"),
                prediction_agent=Agent("gpt-3.5-turbo-0125"),
                max_workers=max_workers,
                agent_name="olas_gpt-3.5-turbo-0125_openai-embeddings",
                embedding_model=EmbeddingModel.openai,
            ),
            PredictionProphetAgent(
                research_agent=Agent("gpt-3.5-turbo-0125"),
                prediction_agent=Agent("gpt-3.5-turbo-0125"),
                max_workers=max_workers,
                agent_name="prediction_prophet_gpt-3.5-turbo-0125_summary",
                use_summaries=True,
            ),
            PredictionProphetAgent(
                research_agent=Agent("gpt-3.5-turbo-0125"),
                prediction_agent=Agent("gpt-3.5-turbo-0125"),
                max_workers=max_workers,
                agent_name="prediction_prophet_gpt-3.5-turbo-0125",
            ),
            PredictionProphetAgent(
                research_agent=Agent("gpt-3.5-turbo-0125"),
                prediction_agent=Agent("gpt-3.5-turbo-0125"),
                max_workers=max_workers,
                agent_name="prediction_prophet_gpt-3.5-turbo-0125_summary_tavilyrawcontent",
                use_summaries=True,
                use_tavily_raw_content=True,
            ),
            PredictionProphetAgent(
                research_agent=Agent("gpt-3.5-turbo-0125"),
                prediction_agent=Agent("gpt-3.5-turbo-0125"),
                max_workers=max_workers,
                agent_name="prediction_prophet_gpt-3.5-turbo-0125_tavilyrawcontent",
                use_tavily_raw_content=True,
            ),
            # PredictionProphetAgent(research_agent=Agent("gpt-4-0125-preview"), prediction_agent=Agent("gpt-4-0125-preview"), max_workers=max_workers, agent_name="prediction_prophet_gpt-4-0125-preview"),  # Too expensive to be enabled by default.
        ],
        cache_path=cache_path,
        only_cached=only_cached,
    )

    benchmarker.run_agents(
        enable_timing=not APIKeys().ENABLE_CACHE,
    )  # Caching of search etc. can distort timings
    md = benchmarker.generate_markdown_report()

    with open(output, "w") as f:
        print(f"Writing benchmark report to: {output}")
        f.write(md)


if __name__ == "__main__":
    typer.run(main)
