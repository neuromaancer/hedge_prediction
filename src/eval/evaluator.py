"""
@Created Date: Thursday March 9th 2023
@Author: Alafate Abulimiti at alafate.abulimiti@gmail.com
@Company: INRIA
@Lab: CoML/Articulab
@School: PSL/ENS
@Description: Evaluator for RL agent.
"""

from typing import Tuple
from clf_scores import f1_score, accuracy_score, precision_score, recall_score
from pathlib import Path
from enum import Enum
import pandas as pd


class MetricsEnum(str, Enum):
    f1_score = "f1_score"
    accuracy_score = "accuracy_score"
    precision_score = "precision_score"
    recall_score = "recall_score"
    accumulated_reward = "accumulated_reward"


class Evaluator:
    def __init__(
        self, test_files_folder: Path, metrics: list[str], preds_folder: Path
    ) -> None:
        self.test_files_folder = test_files_folder
        self.preds_folder = preds_folder
        self.metrics = metrics
        self._validate_inputs()

    def compute(self) -> dict[str, float]:
        # return results with a dict of metrics.
        metrics_result = {}
        preds, pred_rewards = self.read_preds(self.preds_folder)
        targets, target_rewards = self.read_preds(self.test_files_folder)
        for metric in self.metrics:
            if metric == MetricsEnum.f1_score:
                metrics_result["f1_score"] = f1_score(targets, preds)
            elif metric == MetricsEnum.accuracy_score:
                metrics_result["accuracy_score"] = accuracy_score(targets, preds)
            elif metric == MetricsEnum.precision_score:
                metrics_result["precision_score"] = precision_score(targets, preds)
            elif metric == MetricsEnum.recall_score:
                metrics_result["recall_score"] = recall_score(targets, preds)
            elif metric == MetricsEnum.accumulated_reward:
                metrics_result["accumulated_reward"] = self._compare_rewards(
                    target_rewards, pred_rewards
                )
            else:
                raise ValueError(f"metric {metric} is not supported.")
        return metrics_result

    def read_preds(self, folder: Path) -> Tuple[list[int], list[float]]:
        csv_files = list(folder.glob("*.csv"))
        labels = []
        rewards = []
        for csv in csv_files:
            df = pd.read_csv(csv)
            # test csv files must have two columns: label and reward.
            try:
                assert "label" in df.columns
                assert "reward" in df.columns
            except AssertionError:
                raise AssertionError("label and reward must be in the csv file.")
            labels.extend(df["label"].values.tolist())
            rewards.extend(df["reward"].values.tolist())
        return labels, rewards

    def _validate_inputs(self) -> None:
        # check the inputs: model, test_files, metrics dict.
        # check the test_files_folder is a folder. and also preds_folder is a folder.
        try:
            assert self.test_files_folder.is_dir()
            assert self.preds_folder.is_dir()
        except AssertionError:
            raise AssertionError("test_files_folder and preds_folder must be a folder.")
        # check the test_files_folder and preds_folder is not empty.
        try:
            assert len(list(self.test_files_folder.iterdir())) > 0
            assert len(list(self.preds_folder.iterdir())) > 0
        except AssertionError:
            raise AssertionError(
                "test_files_folder and preds_folder must not be empty."
            )
        # the number of files in test_files_folder and preds_folder must be the same.
        try:
            assert len(list(self.test_files_folder.iterdir())) == len(
                list(self.preds_folder.iterdir())
            )
        except AssertionError:
            raise AssertionError(
                "the number of files in test_files_folder and preds_folder must be the same."
            )
        # check the metrics is a list of str.
        try:
            assert isinstance(self.metrics, list)
            assert all(isinstance(metric, str) for metric in self.metrics)
        except:
            raise AssertionError("metrics must be a list of str.")
        # check the metrics' values are in MetricsEnum.
        try:
            assert all(metric in MetricsEnum.__members__ for metric in self.metrics)
        except:
            raise ValueError(f"metrics must be in {MetricsEnum.__members__}")

    def _compare_rewards(self, target_rewards: list[float], pred_rewards: list[float]):
        # compare the rewards of the agent with the rewards of the test set.
        # rewards is a list of rewards.
        # return the difference between the rewards of the agent and the rewards of the test set.
        #! TODO: this function is not used in the current version, needs to be refactored.
        return sum(pred_rewards) - sum(target_rewards)
