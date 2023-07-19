"""
@Created Date: Friday April 14th 2023
@Author: Alafate Abulimiti at alafate.abulimiti@gmail.com
@Company: INRIA
@Lab: CoML/Articulab
@School: PSL/ENS
@Description: Test SVM classifier.
"""

from datetime import datetime
from pathlib import Path
import warnings
from ast import literal_eval

import matplotlib.pyplot as plt
import click
import numpy as np
import pandas as pd
import shap
import torch
from dotenv import dotenv_values
from imblearn.over_sampling import SVMSMOTE
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from rich import print as rprint
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.svm import SVC

from dataset import HedgingPredDataModule
from eval.ci import f1_score_confidence_interval, normal_ci_scoring

warnings.filterwarnings("ignore")

config = dotenv_values("../.env")
EVAL_DATA_FOLDER = config["EVAL_DATA_FOLDER"]
TENSOR_LIST = literal_eval(config["TENSOR_LIST"])
assert isinstance(TENSOR_LIST, list), "TENSOR_LIST must be a list of strings."
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
    default="../logs/svm/svm.log",
    type=str,
    help="Name of the log file.",
)
def train_svm(k_folds, compute_shap, log_file):
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

    rprint(f"[red]tensor input size: {data_module.get_input_size()}")

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
    x_train = x_train.reshape(x_train.shape[0], -1)
    rprint(f"X_train shape: {x_train.shape}")

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
    x_test = np.array(x_test)
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_train = np.array([x.flatten() for x in x_train])
    x_test = np.array([x.flatten() for x in x_test])
    svmsmote = SVMSMOTE(random_state=42)
    x_train, y_train = svmsmote.fit_resample(x_train, y_train)
    # ------------------Cross validation------------------#
    k = 5  # Number of folds
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],  # Optional, for polynomial kernel
    }

    svm_classifier = SVC(kernel="linear", probability=True, random_state=42)
    grid_search = GridSearchCV(
        svm_classifier, param_grid, scoring="f1_macro", cv=kf, verbose=1, n_jobs=-1
    )
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    rprint("SVM classifier trained.")
    rprint("Predicting...")

    y_ = best_estimator.predict(x_test)
    accuracy = accuracy_score(y_test, y_)
    recall = recall_score(y_test, y_)
    precision = precision_score(y_test, y_)
    f1 = f1_score(y_test, y_)
    conf_mat = confusion_matrix(y_test, y_)
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
        explainer = shap.KernelExplainer(best_estimator.predict_proba, x_train)
        rprint("[green]Kernel explainer created.")
        features = list(FEATURE_DICT.values())
        today = datetime.today().strftime("%Y-%m-%d")
        special_str = "hedge"
        save_to = Path("../docs/images/shap")
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
        plt.savefig(save_to / f"svm_shap_{today}_summary_1.png")

if __name__ == "__main__":
    train_svm()

# ------------------SHAP analysis------------------#

# # Select K representative samples using shap.kmeans
# X_summary_kmeans = shap.kmeans(x_train, 100)
# rprint("[green]K representative samples selected.")
# explainer = shap.KernelExplainer(svm_classifier.predict_proba, X_summary_kmeans)
# rprint("[green]Kernel explainer created.")
# # Compute Shapley values for all instances in the test set
# x_test = x_test[:10]
# all_shap_values = np.array([explainer.shap_values(x) for x in x_test])
# rprint("[green]Shapley values computed.")
# # Calculate the average Shapley values for each feature
# average_shap_values = np.mean(all_shap_values, axis=0)

# # Print the average Shapley values

# rprint(average_shap_values)
# rprint(f"shape of average shap values: {average_shap_values.shape}")
# # reshape back to the original shape
# shap_values_3d = np.reshape(average_shap_values, x_train.shape)
# #print the shap values
# rprint(shap_values_3d)
# rprint(f"shape of shap values 3d: {shap_values_3d.shape}")

# # draw the shap values
# # shap.summary_plot(
# #     average_shap_values, x_test, feature_names=list(FEATURE_DICT.values())
# # )
# # plt.savefig("shap.png")
