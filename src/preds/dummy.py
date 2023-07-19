"""
@Created Date: Friday March 17th 2023
@Author: Alafate Abulimiti at alafate.abulimiti@gmail.com
@Company: INRIA
@Lab: CoML/Articulab
@School: PSL/ENS
@Description: Dummy classifier that just predicts along with class distribution.
"""

import torch
import numpy as np
from rich import print as rprint
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryStatScores,
    F1Score,
    Precision,
    Recall,
)
from sklearn.metrics import classification_report
from torch.utils.data import Dataset

import math

from eval.ci import f1_score_confidence_interval, normal_ci_scoring


class DummyClassifier:
    def __init__(self, train_set: Dataset, test_set: Dataset):
        self.class_distribution = None
        self.train_set = train_set
        self.test_set = test_set
        self.accuracy = BinaryAccuracy()
        self.f1_score = F1Score(task="binary", num_classes=2, average="weighted")
        self.precis = Precision(task="binary", num_classes=2, average="weighted")
        self.recall = Recall(task="binary", num_classes=2, average="weighted")
        self.stat_scores = BinaryStatScores()

    def fit(self):
        # Calculate the class distribution in the training dataset
        targets = [sample["label"].cpu() for sample in self.train_set]
        targets = torch.stack(targets).numpy()
        class_counts = np.bincount(targets)
        self.class_distribution = torch.Tensor(class_counts) / len(targets)
        self.class_distribution = {
            index: round(value.item(), 3)
            for index, value in enumerate(self.class_distribution)
        }

    def predict(self):
        # Sample from the class distribution
        classes = list(self.class_distribution.keys())
        probabilities = list(self.class_distribution.values())
        y_hat = np.random.choice(classes, size=len(self.test_set), p=probabilities)
        y = [sample["label"].item() for sample in self.test_set]
        accuracy = self.accuracy(torch.Tensor(y_hat), torch.Tensor(y))
        accuracy_interval_range =normal_ci_scoring(accuracy, len(y))
        f1 = self.f1_score(torch.Tensor(y_hat), torch.Tensor(y))
        precision_score = self.precis(torch.Tensor(y_hat), torch.Tensor(y))
        recall = self.recall(torch.Tensor(y_hat), torch.Tensor(y))
        states_scores = self.stat_scores(torch.Tensor(y_hat), torch.Tensor(y))
        rprint(classification_report(y, y_hat, zero_division=0))
        tp, fp, tn, fn, support = states_scores
        n_recall = tp + fn
        n_precision = tp + fp
        recall_std_error = math.sqrt((recall * (1 - recall)) / n_recall)
        precision_std_error = math.sqrt(
            (precision_score * (1 - precision_score)) / n_precision
        )
        #! confidence interval 𝛼=0.05 yields 𝑧=1.96
        recall_interval_range = 1.96 * recall_std_error
        precision_interval_range = 1.96 * precision_std_error
        f1_score, f1_interval_range, _, _ = f1_score_confidence_interval(
            recall.cpu().numpy(),
            precision_score.cpu().numpy(),
            recall_interval_range,
            precision_interval_range,
        )
        result = {
            "accuracy": accuracy,
            "accuracy_ci_range": accuracy_interval_range,
            "f1_score": f1,
            "f1_ci_range": f1_interval_range,
            "precision": precision_score,
            "precision_ci_range": precision_interval_range,
            "recall": recall,
            "recall_ci_range": recall_interval_range,
        }
        return result
