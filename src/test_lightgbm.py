"""
@Created Date: Friday April 14th 2023
@Author: Alafate Abulimiti at alafate.abulimiti@gmail.com
@Company: INRIA
@Lab: CoML/Articulab
@School: PSL/ENS
@Description: Test LightGBM classifier.
"""
import warnings
from ast import literal_eval
from datetime import datetime
from pathlib import Path

import click
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import torch
from dotenv import dotenv_values
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from rich import print as rprint
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import StratifiedKFold

from dataset import HedgingPredDataModule
from eval.ci import f1_score_confidence_interval, normal_ci_scoring
from interpret.shap_explain import draw_bar

warnings.filterwarnings("ignore")

config = dotenv_values("../.env")
EVAL_DATA_FOLDER = config["EVAL_DATA_FOLDER"]
TENSOR_LIST = literal_eval(config["TENSOR_LIST"])
assert isinstance(TENSOR_LIST, list), "TENSOR_LIST must be a list of strings."
rprint(TENSOR_LIST)
FEATURE_DICT = literal_eval(config["FEATURE_DICT"])
assert isinstance(FEATURE_DICT, dict), "FEATURE_DICT must be a dictionary."

train_data_folder = "/scratch2/aabulimiti/hedge_prediction/data/trainset"
test_data_folder = "/scratch2/aabulimiti/hedge_prediction/data/testset"
sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")


@click.command()
@click.option(
    "-k",
    "--k_folds",
    default=10,
    type=int,
    help="Number of folds for cross validation.",
)
@click.option(
    "-s",
    "--compute_shap",
    default=False,
    type=bool,
    help="Enable or disable SHAP analysis.",
)
@click.option(
    "-l",
    "--log_file",
    default="../logs/lightgbm/lightgbm.log",
    type=str,
    help="Name of the log file.",
)
def train_lightgbm(k_folds, compute_shap, log_file):
    data_module = HedgingPredDataModule(
        train_data_folder=train_data_folder,
        test_data_folder=test_data_folder,
        batch_size=6,
        num_workers=0,
        sentence_embed_model=sent_transformer,
        tensor_list=TENSOR_LIST,
    )

    data_module.setup()
    data_module.prepare_data()

    train_dataset = data_module.train_set
    test_dataset = data_module.test_set
    rprint("Dataset loaded.")
    rprint(f"Train dataset size: {len(train_dataset)}")
    rprint(f"Test dataset size: {len(test_dataset)}")

    y_train = [sample["label"].cpu().numpy() for sample in train_dataset]

    x_train = []
    for sample in train_dataset:
        del sample["label"]
        del sample["state"]
        del sample["record"]
        del sample["turn_tensor"]
        values = [t.cpu() for t in sample.values()]
        values = torch.cat(values, dim=-1).cpu().numpy()
        x_train.append(values)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    y_test = [sample["label"].cpu().numpy() for sample in test_dataset]

    x_test = []

    for sample in test_dataset:
        del sample["label"]
        del sample["state"]
        del sample["record"]
        del sample["turn_tensor"]
        values = [t.cpu() for t in sample.values()]
        values = torch.cat(values, dim=-1).cpu().numpy()
        x_test.append(values)
    rprint(len(x_train), len(y_train))
    rprint(len(x_test), len(y_test))

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = np.array([x.flatten() for x in x_train])
    x_test = np.array([x.flatten() for x in x_test])
    rprint(x_train.shape, y_train.shape)
    rprint(x_train)
    rprint(y_train)
    svmsmote = SMOTE(random_state=42)
    x_train, y_train = svmsmote.fit_resample(x_train, y_train)

    #  Prepare data for cross-validation
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies, fold_recalls, fold_precisions, fold_f1_scores = [], [], [], []
    best_model = None
    best_f1 = 0
    # Train the model
    num_round = 50
    # Perform k-fold cross-validation
    for train_idx, val_idx in skf.split(x_train, y_train):
        x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Convert the data into Dataset format
        dtrain = lgb.Dataset(x_train_fold, label=y_train_fold)
        dval = lgb.Dataset(x_val_fold, label=y_val_fold)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.0001,
            "verbose": -1,
            "max_depth": -1,
            "num_leaves": 100,
        }
        bst = lgb.train(
            params,
            dtrain,
            num_round,
            valid_sets=[dval],
        )
        # Make predictions
        y_pred = bst.predict(x_val_fold)
        y_pred = np.round(y_pred)

        # Calculate metrics
        accuracy = accuracy_score(y_val_fold, y_pred)
        recall = recall_score(y_val_fold, y_pred)
        precision = precision_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred)

        # Append the metrics for this fold
        fold_accuracies.append(accuracy)
        fold_recalls.append(recall)
        fold_precisions.append(precision)
        fold_f1_scores.append(f1)
        rprint(f"best model type: {type(best_model)}")
        if  best_model is None or f1 > best_f1:
            best_f1 = f1
            rprint(f"Best F1: {best_f1}")
            best_model = bst
    rprint(f"best model type: {type(best_model)}")
    # Compute the average metrics across all folds
    avg_accuracy = np.mean(fold_accuracies)
    avg_recall = np.mean(fold_recalls)
    avg_precision = np.mean(fold_precisions)
    avg_f1 = np.mean(fold_f1_scores)

    print(f"Average accuracy: {avg_accuracy}")
    print(f"Average recall: {avg_recall}")
    print(f"Average precision: {avg_precision}")
    print(f"Average F1: {avg_f1}")

    rprint("[cyan]LightGBM classifier trained.")
    rprint("[cyan]Predicting...")
    y_pred = best_model.predict(x_test)
    y_pred = np.round(y_pred)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_mat.ravel()
    n_recall = tp + fn
    n_precision = tp + fp
    rprint(f"Accuracy: {accuracy}")
    rprint(f"ci accuracy: {normal_ci_scoring(accuracy, len(y_test))}")
    rprint(f"Recall: {recall}")
    rprint(f"ci recall: {normal_ci_scoring(recall, n_recall)}")
    rprint(f"Precision: {precision}")
    rprint(f"ci precision: {normal_ci_scoring(precision, n_precision)}")
    rprint(f"F1: {f1}")
    rprint(
        f"ci f1: {f1_score_confidence_interval(recall,precision,normal_ci_scoring(recall, n_recall), normal_ci_scoring(precision, n_precision))}"
    )
    if compute_shap:
        explainer = shap.TreeExplainer(best_model)
        features = list(FEATURE_DICT.values())
        today = datetime.now().strftime("%Y-%m-%d")
        special_str = "non-hedge"
        save_to = Path("../docs/images/shap")
        # Calculate average SHAP values for all instances in the test set
        # all_shap_values = np.array([explainer.shap_values(x) for x in x_test])
        all_shap_values = explainer.shap_values(x_test)
        all_shap_values = np.array(all_shap_values)
        rprint(all_shap_values)

        rprint(f"all_shap_values shape: {all_shap_values.shape}")
        reshaped_average_shap_values = all_shap_values.reshape((2, 1287, -1, 51))
        shap_vals_avg = np.mean(reshaped_average_shap_values, axis=2)
        shap_vals_avg_list = [arr for arr in shap_vals_avg]
        shap_vals_avg_1 = shap_vals_avg[1]
        shap_vals_avg_1_df = pd.DataFrame(shap_vals_avg_1, columns=features)
        shap.summary_plot(
            shap_vals_avg_1,
            feature_names=features,
            show=False,
        )
        plt.savefig(save_to / f"lightgbm_shap_{today}_summary_1.png")
        # shap_vals_avg = np.mean(reshaped_average_shap_values, axis=1)
        # rprint("prediction on 0: non-hedge")
        # all_shap_values = all_shap_values[0].squeeze()
        # reshaped_average_shap_values = all_shap_values.reshape((1287, 4, 51))
        # shap_vals_avg = np.mean(reshaped_average_shap_values, axis=1)
        # rprint(f"shap before mean: {shap_vals_avg}")
        # rprint(f"shape before mean: {shap_vals_avg.shape}")
        # shap_vals_avg = np.mean(shap_vals_avg, axis=0)
        # rprint(f"shape after mean: {shap_vals_avg.shape}")
        # rprint(f"shap values after mean: {shap_vals_avg}")

        # draw_bar(features, shap_vals_avg, save_to, "lightgbm", today, special_str)


if __name__ == "__main__":
    train_lightgbm()
