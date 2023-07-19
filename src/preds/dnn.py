"""
@Created Date: Monday February 13th 2023
@Author: Alafate Abulimiti at alafate.abulimiti@gmail.com
@Company: INRIA
@Lab: CoML/Articulab
@School: PSL/ENS
@Description: DNN model for tutor's hedge prediction.
"""

from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print as rprint
from sklearn.metrics import classification_report
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryStatScores,
    F1Score,
    Precision,
    Recall,
)

import math
from eval.ci import f1_score_confidence_interval, normal_ci_scoring

class DNNModel(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 4 * 512,
        eval_dir: str = "eval",
        data_hierarchical: bool = False,
    ) -> None:
        super().__init__()
        self.eval_dir = eval_dir
        self.name = "mlp"
        self.data_hierarchical = data_hierarchical
        self.fc1 = nn.Linear(input_size, 512)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(p=0.5)
        self.output = nn.Linear(64, 1)
        # Initialize the loss function
        self.loss = nn.BCELoss()
        # initialize the metrics
        self.accuracy = BinaryAccuracy()
        self.f1_score = F1Score(task="binary", num_classes=2, average="weighted")
        self.precis = Precision(task="binary", num_classes=2, average="weighted")
        self.recall = Recall(task="binary", num_classes=2, average="weighted")
        self.stat_scores = BinaryStatScores()
        self.predictions = []
        self.targets = []
        self.save_hyperparameters()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input tensor
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = torch.sigmoid(self.output(x))
        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=10e-4)

    def _step(self, batch, batch_idx, mode="train"):
        x, y = batch
        y = y.to(torch.float32)
        y_pred = self(x)
        loss = self.loss(y_pred, y.unsqueeze(1))
        self.log(
            f"{mode}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch, batch_idx) -> float:
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx) -> float:
        return self._step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx) -> float:
        x, y = batch
        y = y.to(torch.float32).unsqueeze(1)
        # batch size is 1.
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        y_pred = torch.round(y_pred)
        self.predictions.append(y_pred)
        self.targets.append(y)

    def on_test_epoch_end(self):
        y_hat = torch.cat(self.predictions)
        y = torch.cat(self.targets)
        acc = self.accuracy(y_hat, y)
        f1 = self.f1_score(y_hat, y)
        precision = self.precis(y_hat, y)
        recall = self.recall(y_hat, y)
        states_scores = self.stat_scores(y_hat, y)
        tp, fp, tn, fn, support = states_scores
        n_recall = tp + fn
        n_precision = tp + fp
        accuracy_ci = normal_ci_scoring(acc.cpu().numpy(), support)
        recall_std_error = math.sqrt((recall * (1 - recall)) / n_recall)
        precision_std_error = math.sqrt((precision * (1 - precision)) / n_precision)
        #! confidence interval 𝛼=0.05 yields 𝑧=1.96
        recall_interval_range = 1.96 * recall_std_error
        precision_interval_range = 1.96 * precision_std_error
        f1_score, f1_interval_range, _, _ = f1_score_confidence_interval(
            recall.cpu().numpy(), precision.cpu().numpy(), recall_interval_range, precision_interval_range
        )
        y_hat = y_hat.cpu().numpy()
        y = y.cpu().numpy()
        report = classification_report(y, y_hat, output_dict=True, zero_division=0)
        self.log("accuracy", acc)
        self.log("accuracy_ci", accuracy_ci)
        self.log("f1_score", f1)
        self.log("f1_score_by_function", f1_score)
        self.log("f1_score_interval_range", f1_interval_range)
        self.log("precision", precision)
        self.log("precision_interval_range", precision_interval_range)
        self.log("recall", recall)
        self.log("recall_interval_range", recall_interval_range)
        rprint(report)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_pred = self(x).squeeze(1)
        y_pred = torch.round(y_pred)

        # Write predictions to file for the current dataset
        preds = y_pred.cpu().numpy()
        output_dir = Path(f"{self.eval_dir}/{self.name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        to_save = output_dir / f"{dataloader_idx}.csv"
        df = pd.DataFrame(preds, columns=["label"])
        df.to_csv(to_save, index=False)

        return preds
