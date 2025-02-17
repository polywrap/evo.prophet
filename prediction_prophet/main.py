import logging
from typing import cast
import click
import time
from dotenv import load_dotenv
from prediction_prophet.functions.debate_prediction import make_debated_prediction
from langchain_community.callbacks import get_openai_callback
from prediction_prophet.functions.research import research as prophet_research
from prediction_market_agent_tooling.benchmark.utils import (
    OutcomePrediction
)

load_dotenv()
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')

def create_output_file(info: str, path: str) -> None:
    with open(path, 'w') as file:
        file.write(info)

def read_text_file(file_path: str) -> str:
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"

@click.group()
def cli() -> None:
    pass

@cli.command()
@click.argument('prompt')
@click.option('--file', '-f', default=None)
def research(
    prompt: str,
    file: str | None = None
) -> None:
    start = time.time()
    
    with get_openai_callback() as cb:
      research = prophet_research(goal=prompt)
    
    report = research.report
    end = time.time()
    
    if file:
        create_output_file(report, file)
        print(f"Output saved to '{file}'")
        print(f"\n\nTime elapsed: {end - start}\n\n{cb}\n\n")
        return
    
    print(f"Research results:\n\n{report}")
    print(f"\n\nTime elapsed: {end - start}\n\n{cb}\n\n")


@cli.command()
@click.argument('prompt')
@click.option('--path', '-p', default=None)
def predict(prompt: str, path: str | None = None) -> None:
    start = time.time()

    with get_openai_callback() as cb:
        if path:
            report = read_text_file(path)
        else:
            logger = logging.getLogger("research")
            logger.setLevel(logging.INFO)
            report = prophet_research(goal=prompt, logger=logger).report
        
        prediction = make_debated_prediction(prompt=prompt, additional_information=report)

    end = time.time()
    
    outcome_prediction = prediction.outcome_prediction
    if outcome_prediction == None:
        raise ValueError("The agent failed to generate a prediction")
    
    outcome_prediction = cast(OutcomePrediction, prediction.outcome_prediction)
    
    print(f"\n\nQuestion: '{prompt}'\nProbability of ocurring: {outcome_prediction.p_yes * 100}%\nConfidence in prediction: {outcome_prediction.confidence * 100}%\nTime elapsed: {end - start}")


if __name__ == '__main__':
    cli()