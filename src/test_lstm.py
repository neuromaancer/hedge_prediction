"""
@Created Date: Saturday February 25th 2023
@Author: Alafate Abulimiti at alafate.abulimiti@gmail.com
@Company: INRIA
@Lab: CoML/Articulab
@School: PSL/ENS
@Description: test file for LSTM model.
"""


import logging
import random
import warnings
from ast import literal_eval
from pathlib import Path

import click
import lightning.pytorch as pl
import torch
from dotenv import dotenv_values
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from rich import print as rprint
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold

from dataset import HedgingPredDataModule
from interpret.shap_explain import explain_deep_model
from preds.lstm import LSTMModel

warnings.filterwarnings("ignore")

config = dotenv_values("../.env")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EVAL_DATA_FOLDER = config["EVAL_DATA_FOLDER"]
TENSOR_LIST = literal_eval(config["TENSOR_LIST"])
assert isinstance(TENSOR_LIST, list), "TENSOR_LIST must be a list of strings."
# shape explainer
SHAP_IMAGE_FOLDER = Path(config["SHAP_IMAGE_FOLDER"])
SHAP_IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
FEATURE_DICT = literal_eval(config["FEATURE_DICT"])
assert isinstance(FEATURE_DICT, dict), "FEATURE_DICT must be a dict of strings."
## Init DataModule
sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")
train_data_folder = "/scratch2/aabulimiti/hedge_prediction/data/trainset"
test_data_folder = "/scratch2/aabulimiti/hedge_prediction/data/testset"


@click.command()
@click.option(
    "-a",
    "--if_attention",
    default=False,
    type=bool,
    help="Enable or disable attention mechanism.",
)
@click.option(
    "-f",
    "--feature_mode",
    default="features_only",
    type=str,
    help="Feature mode to use.",
)
@click.option("-e", "--epochs", default=10, type=int, help="Number of training epochs.")
@click.option(
    "-k",
    "--k_folds",
    default=10,
    type=int,
    help="Number of folds for cross validation.",
)
@click.option(
    "-d", "--debug_mode", default=False, type=bool, help="Enable or disable debug mode."
)
@click.option(
    "--av", default=["rapport"], multiple=True, type=str, help="ablation value"
)
@click.option(
    "-s", "--shap", default=False, type=bool, help="Enable or disable SHAP analysis."
)
@click.option(
    "-l",
    "--log_file",
    default="../logs/lstm/attn.log",
    type=str,
    help="Name of the log file.",
)
def train_lstm(
    if_attention, feature_mode, epochs, k_folds, debug_mode, shap, av, log_file
):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    rprint("[red]*" * 50)
    logging.info(f"use attention: {if_attention}")
    logging.info(f"use feature_mode: {feature_mode}")
    logging.info(f"epochs number: {epochs}")
    logging.info(f"use debug_mode: {debug_mode}")
    logging.info(f"use shap: {shap}")
    # Your LSTM training code here
    data_module = HedgingPredDataModule(
        train_data_folder=train_data_folder,
        test_data_folder=test_data_folder,
        batch_size=6,
        num_workers=0,
        sentence_embed_model=sent_transformer,
        tensor_list=TENSOR_LIST,
    )

    data_module.prepare_data()
    data_module.setup()

    input_size = data_module.get_input_size()
    logging.info(f"[green]Input size:[/green] ", input_size)

    # init model
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min")
    # # init callbacks
    checkpoint_callback = ModelCheckpoint(
        filename="{feature_mode}_{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    if if_attention:
        logger_name = "attn_lstm"
    else:  # no attention
        logger_name = "lstm"
    # init tensorboard logger
    logger = pl_loggers.TensorBoardLogger(save_dir="../models/", name=f"{logger_name}")
    # k-fold cross validation
    dataset = data_module.data
    logging.info("Dataset length:", len(dataset))
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    best_model_path = None
    best_val_loss = float("inf")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        rprint(f"train index: {train_idx}")
        rprint(f"val index: {val_idx}")
        logging.info(f"Fold {fold+1}")
        # Update train and validation DataLoaders
        data_module.update_train_val_datasets(train_idx, val_idx)
        rprint("Max train index:", max(train_idx))
        rprint("Max val index:", max(val_idx))
        # Instantiate your LightningModule and Trainer
        lstm = LSTMModel(
            input_dim=input_size,
            hidden_dim=300,
            num_layers=1,
            output_dim=1,
            dropout_rate=0.5,
            eval_dir=EVAL_DATA_FOLDER,
            if_attention=if_attention,
        )
        trainer = pl.Trainer(
            logger=logger,
            accelerator="auto",
            strategy="auto",
            devices="auto",
            max_epochs=epochs,
            callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar()],
            fast_dev_run=debug_mode,
        )

        # Fit the model
        trainer.fit(lstm, datamodule=data_module)
        # Test model on test set
        trainer.test(lstm, datamodule=data_module)
        current_best_val_loss = checkpoint_callback.best_model_score.item()
        if current_best_val_loss < best_val_loss:
            best_val_loss = current_best_val_loss
            best_model_path = checkpoint_callback.best_model_path

    # ----------------------------shape analysis--------------------------------#
    if shap:
        rprint(f"Best model path after {k_folds} folds: {best_model_path}")
        lstm.load_from_checkpoint(best_model_path)
        lstm = lstm.to(device)
        test_dataloader = data_module.test_dataloader()
        train_dataloader = data_module.train_dataloader()
        # Create a list of samples from the DataLoader
        train_samples = []
        for data_batch, target_batch in train_dataloader:
            for data_sample, _ in zip(data_batch, target_batch):
                train_samples.append((data_sample))
        logging.info("[bold green]Train samples:[/bold green] ", train_samples[0])
        test_samples = []
        for data_batch, target_batch in test_dataloader:
            for data_sample, _ in zip(data_batch, target_batch):
                test_samples.append((data_sample))
        # Randomly select 500 train samples and 30 test samples
        train_samples_subset = random.sample(train_samples, 500)
        test_samples_subset = random.sample(test_samples, 30)

        train_samples_tensor = torch.stack(train_samples).to(device)
        test_samples_tensor = torch.stack(test_samples).to(device)

        logging.info(train_samples_tensor.shape)
        logging.info(test_samples_tensor.shape)

        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super(ModelWrapper, self).__init__()
                self.model = model

            def forward(self, x):
                out = self.model(x)
                return out.unsqueeze(1)

        lstm_wrapped = ModelWrapper(lstm)
        torch.backends.cudnn.enabled = False
        special_str = logger_name
        features_list = list(FEATURE_DICT.values())
        features = [i for i in features_list if i not in av]
        explain_deep_model(
            lstm_wrapped,
            features,
            train_samples_tensor,
            test_samples_tensor,
            SHAP_IMAGE_FOLDER,
            special_str,
        )


if __name__ == "__main__":
    train_lstm()
