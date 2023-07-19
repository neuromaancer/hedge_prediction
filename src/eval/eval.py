"""
@Created Date: Thursday March 9th 2023
@Author: Alafate Abulimiti at alafate.abulimiti@gmail.com
@Company: INRIA
@Lab: CoML/Articulab
@School: PSL/ENS
@Description: conduct the evaluation from command line.
"""

import click
from evaluator import Evaluator
from rich import print as rprint

@click.command()
@click.option("--source_file_path", help=".txt file containing the source texts.")
@click.option(
    "--hypos_file_path", help=".txt file containing the hedge style transferred texts."
)
@click.option(
    "--source_labels_file_path", help=".txt file containing the source labels."
)
@click.option(
    "--hypo_labels_file_path",
    help=".txt file containing the hedge style transferred labels.",
)
@click.option(
    "--bart_score_model_path",
    help="Path to the BART model used for the bart score evaluation.",
)
def eval(
    source_file_path: str,
    hypos_file_path: str,
    source_labels_file_path: str,
    hypo_labels_file_path: str,
    bart_score_model_path: str,
):
    """
    eval: This function is the command line interface for the evaluation of the hedge style transfer task.
    Args:
        source_file_path (str): .txt file containing the source texts.
        hypos_file_path (str): .txt file containing the hedge style transferred texts.
        source_labels_file_path (str): .txt file containing the source labels.
        hypo_labels_file_path (str): .txt file containing the hedge style transferred labels.
        bart_score_model_path (str): Path to the BART model used for the bart score evaluation.
    """
    sources = open(source_file_path, "r").readlines()
    hypos = open(hypos_file_path, "r").readlines()
    source_labels = open(source_labels_file_path, "r").readlines()
    hypo_labels = open(hypo_labels_file_path, "r").readlines()
    source_labels = [int(label) for label in source_labels]
    hypo_labels = [int(label) for label in hypo_labels]

    evaluator = Evaluator(
        sources=sources,
        hypotheses=hypos,
        source_labels=source_labels,
        hypothesis_labels=hypo_labels,
        bart_score_model_path=bart_score_model_path,
    )
    scores_dict = evaluator.compute()
    for k, v in scores_dict.items():
        rprint(k.replace("_", " "), f": [deep_sky_blue1]{v}")  # rich print
    rprint("[green]Done!")


if __name__ == "__main__":
    eval()