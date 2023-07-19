import numpy as np
from rich import print as rprint
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from dataset import HedgingPredDataModule
from eval.ci import f1_score_confidence_interval, normal_ci_scoring


def random_classifier(len_x):
    """
    Generates random predictions for binary classification.

    Parameters:
    X: np.array
        Input array (n_samples, n_features)

    Returns:
    np.array
        Random predictions (n_samples, )
    """
    n_samples = len_x
    return np.random.choice([0, 1], size=n_samples)


# Assuming your input data (X_test) and ground truth labels (y_test) are available
train_data_folder = "/scratch2/aabulimiti/hedge_prediction/data/trainset"
test_data_folder = "/scratch2/aabulimiti/hedge_prediction/data/testset"
sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")
data_module = HedgingPredDataModule(
    train_data_folder=train_data_folder,
    test_data_folder=test_data_folder,
    batch_size=6,
    num_workers=0,
    sentence_embed_model=sent_transformer,
)
data_module.setup()
data_module.prepare_data()

train_dataset = data_module.train_set
test_dataset = data_module.test_set

y_test = [sample["label"].cpu().numpy() for sample in test_dataset]
random_predictions = random_classifier(len(test_dataset))
accuracy = accuracy_score(y_test, random_predictions)

print("Random Classifier Accuracy:", accuracy)
accuracy = accuracy_score(y_test, random_predictions)
recall = recall_score(random_predictions, y_test)
precision = precision_score(random_predictions, y_test)
f1 = f1_score(random_predictions, y_test)

# ------------------Confidence interval------------------#
conf_mat = confusion_matrix(random_predictions, y_test)
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
    f"ci f1: {f1_score_confidence_interval(recall,precision,normal_ci_scoring(recall, n_recall), normal_ci_scoring(precision, n_precision))}"  # noqa: E501
)
