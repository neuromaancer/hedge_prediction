"""
@Created Date: Tuesday March 14th 2023
@Author: Alafate Abulimiti at alafate.abulimiti@gmail.com
@Company: INRIA
@Lab: CoML/Articulab
@School: PSL/ENS
@Description: Shap explanations for the models.
"""


from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
from dotenv import dotenv_values
from rich import print as rprint

today = datetime.today().strftime("%Y-%m-%d")

config = dotenv_values("../.env")
shap_image_folder = Path(config["SHAP_IMAGE_FOLDER"])


def explain_deep_model(
    model, features, background_data, test_data, plot_folder_to_save, special_str=""
):
    # Ensure the model is in evaluation mode
    model.eval()
    rprint("model name is: ", model.model.name)
    model_name = model.model.name
    features_name = features
    # Create a DeepExplainer object
    explainer = shap.DeepExplainer(model, background_data)

    # Compute the SHAP values for the test dataset
    shap_values = explainer.shap_values(test_data)
    rprint("Shape of shap_values:", np.array(shap_values).shape)
    shap_values = np.array(shap_values)
    shap_values = np.mean(shap_values, axis=1)
    rprint("Shape of shap_values after mean:", shap_values.shape)
    shap.summary_plot(shap_values, feature_names=features_name, show=False)
    plot_folder = Path(plot_folder_to_save)
    plot_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_folder / f"{model_name}_shap_{today}_summary_1_{special_str}.png")
    plt.clf()

    # Create and save the bar plot
    # plt.bar(list(features_dict.values()), mean_shap_values)
    # plt.xticks(rotation=90, fontsize=5)
    # plt.xlabel("Features")
    # plt.ylabel("Mean SHAP Values")
    # plt.title("Feature Importance")
    # plt.savefig(plot_folder / f"{model.model.name}_bar_plot_mean_shap_{today}.svg")
    # plt.clf()
    rprint(f"[green]Bar plot for {model.model.name} saved to disk.")


def explain_3d_model(
    model, features_dict, background_data, test_data, plot_folder_to_save
):
    model_name = model.__class__.__name__
    rprint("model name is: ", model_name)
    feature_names = list(features_dict.values())
    explainer = shap.KernelExplainer(model.predict_proba, background_data)
    all_shap_values = np.array([explainer.shap_values(x) for x in test_data])
    average_shap_values = np.mean(all_shap_values, axis=0)
    shap_values_3d = np.reshape(average_shap_values, background_data.shape)
    rprint(shap_values_3d)
    rprint(f"shape of shap values 3d: {shap_values_3d.shape}")
    shap_values_avg = np.mean(shap_values_3d, axis=2)
    rprint(f"shape of shap values avg: {shap_values_avg.shape}")

    # Create a summary plot with the averaged SHAP values
    # Create and save the bar plot
    plt.bar(feature_names, shap_values_avg)
    plt.xticks(rotation=90, fontsize=5)
    plt.xlabel("Features")
    plt.ylabel("Mean SHAP Values")
    plt.title("Feature Importance")
    plot_folder = Path(plot_folder_to_save)
    plot_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_folder / f"{model_name}_bar_plot_mean_shap_{today}.svg")
    plt.clf()
    rprint(f"[green]Bar plot for {model_name} saved to disk.")


def draw_bar(
    features: list,
    mean_shap_values: np.ndarray,
    save_to: Path,
    model_name: str,
    date: str,
    special_str: str = "",
):
    plt.bar(features, mean_shap_values)
    plt.xticks(rotation=90, fontsize=5)
    plt.xlabel("Features")
    plt.ylabel("Mean SHAP Values")
    plt.title("Feature Importance")
    plt.savefig(save_to / f"{model_name}_bar_plot_mean_shap_{date}_{special_str}.svg")
    plt.clf()
    rprint(f"[green]Bar plot for {model_name} saved to disk.")
