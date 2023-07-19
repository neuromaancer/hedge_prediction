"""
@Created Date: Monday February 13th 2023
@Author: Alafate Abulimiti at alafate.abulimiti@gmail.com
@Company: INRIA
@Lab: CoML/Articulab
@School: PSL/ENS
@Description: test file for DNN model.
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
from preds.dnn import DNNModel

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
    "-s", "--shap", default=False, type=bool, help="Enable or disable SHAP analysis."
)
@click.option(
    "-l",
    "--log_file",
    default="../logs/mlp/mlp.log",
    type=str,
    help="Name of the log file.",
)
def train_mlp(feature_mode, epochs, k_folds, debug_mode, shap, log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    rprint("[red]*" * 50)
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
    mlp_input_size = input_size * 4
    logging.info("[green]Input size:[/green] ", input_size)

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
    # init tensorboard logger
    logger = pl_loggers.TensorBoardLogger(save_dir="../models/", name="mlp")
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
        mlp = DNNModel(
            input_size=mlp_input_size,
            eval_dir=EVAL_DATA_FOLDER,
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
        trainer.fit(mlp, datamodule=data_module)
        # Test model on test set
        trainer.test(mlp, datamodule=data_module)
        current_best_val_loss = checkpoint_callback.best_model_score.item()
        if current_best_val_loss < best_val_loss:
            best_val_loss = current_best_val_loss
            best_model_path = checkpoint_callback.best_model_path

    # ----------------------------shape analysis--------------------------------#
    if shap:
        rprint(f"Best model path after {k_folds} folds: {best_model_path}")
        mlp.load_from_checkpoint(best_model_path)
        mlp = mlp.to(device)
        test_dataloader = data_module.test_dataloader()
        train_dataloader = data_module.train_dataloader()
        # Create a list of samples from the DataLoader
        train_samples = []
        for data_batch, target_batch in train_dataloader:
            train_samples.extend(
                data_sample for data_sample, _ in zip(data_batch, target_batch)
            )
        logging.info("[bold green]Train samples:[/bold green] ", train_samples[0])
        test_samples = []
        for data_batch, target_batch in test_dataloader:
            test_samples.extend(
                data_sample for data_sample, _ in zip(data_batch, target_batch)
            )
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

        mlp_wrapped = ModelWrapper(mlp)
        torch.backends.cudnn.enabled = False

        explain_deep_model(
            mlp_wrapped,
            FEATURE_DICT,
            train_samples_tensor,
            test_samples_tensor,
            SHAP_IMAGE_FOLDER,
        )


if __name__ == "__main__":
    train_mlp()
