"""
@Created Date: Saturday February 25th 2023
@Author: Alafate Abulimiti at alafate.abulimiti@gmail.com
@Company: INRIA
@Lab: CoML/Articulab
@School: PSL/ENS
@Description: LSTM model for tutor's hedge prediction.
"""

from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rich import print as rprint
from sklearn.metrics import classification_report
from torch.nn import functional as F
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryStatScores,
    F1Score,
    Precision,
    Recall,
)
import math

from eval.ci import f1_score_confidence_interval, normal_ci_scoring

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        attention_output = torch.sum(attention_weights * x, dim=1)
        return attention_output


class LSTMModel(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 300,
        num_layers: int = 1,
        dropout_rate: float = 0.5,
        eval_dir: str = "eval",
        if_attention: bool = False,
    ):
        super().__init__()
        self.eval_dir = eval_dir
        self.name = "lstm"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.if_attention = if_attention
        self.predictions = []
        self.targets = []
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.attention = Attention(hidden_dim)

        # Output a single binary value.
        self.fc = nn.Linear(hidden_dim, output_dim)

        # initialize the metrics
        self.accuracy = BinaryAccuracy()
        self.f1_score = F1Score(task="binary", num_classes=2, average="weighted")
        self.precis = Precision(task="binary", num_classes=2, average="weighted")
        self.recall = Recall(task="binary", num_classes=2, average="weighted")
        self.stat_scores = BinaryStatScores()

        # Initialize the loss function
        self.loss = nn.BCELoss()

    def forward(self, x):
        x = x.to(torch.float32)
        # Pass the input through the LSTM
        output, (h_n, c_n) = self.lstm(x)
        attention_output = self.attention(output)
        if self.if_attention:
            output = self.fc(attention_output)
        else:
            output = self.fc(output[:, -1])
        output = torch.sigmoid(output)
        return output.squeeze()

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3)

    def _step(self, batch, batch_idx, mode):
        x, y = batch
        y = y.to(torch.float32)
        y_hat = self(x)
        if y_hat.shape != y.shape:
            y_hat = y_hat.unsqueeze(0)
        loss = self.loss(y_hat, y)
        self.log(
            f"{mode}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(torch.float32)
        y_hat = self(x)
        y_hat = y_hat.unsqueeze(0)
        loss = self.loss(y_hat, y)
        # Calculate metrics
        y_pred = torch.round(y_hat)
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
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
            recall.cpu().numpy(),
            precision.cpu().numpy(),
            recall_interval_range,
            precision_interval_range,
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
        dataset = self.test_set.datasets[dataloader_idx]
        rprint(f"Predicting on {dataset.name} dataset...")
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
