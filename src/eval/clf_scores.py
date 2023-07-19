"""
@Created Date: Thursday March 9th 2023
@Author: Alafate Abulimiti at alafate.abulimiti@gmail.com
@Company: INRIA
@Lab: CoML/Articulab
@School: PSL/ENS
@Description: Classification scores calculations.
"""

import numpy as np
from sklearn import metrics


def f1_score(
    y_true: list[int] = None, y_pred: list[int] = None, average: str = "weighted"
) -> float:
    """
    f1_score: f1 score
    Args:
        y_true (list[int], optional): desired style labels from source text. Defaults to None.
        y_pred (list[int], optional): generated text's hedge label. Defaults to None.
        average (str, optional): type of f1 score. Defaults to "weighted".
    Raises:
        ValueError: y_true and y_pred must have the same length
    Returns:
        float: f1 score
    """
    if y_true is None:
        y_true = []
    if y_pred is None:
        y_pred = []
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return metrics.f1_score(y_true, y_pred, average=average)


def accuracy_score(y_true: list[int] = None, y_pred: list[int] = None) -> float:
    """
    accuracy_score: accuracy score
    Args:
        y_true (list[int], optional): desired style labels from source text. Defaults to None.
        y_pred (list[int], optional): generated text's hedge label. Defaults to None.
    Raises:
        ValueError: y_true and y_pred must have the same length
    Returns:
        float: accuracy score
    """
    if y_true is None:
        y_true = []
    if y_pred is None:
        y_pred = []
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return metrics.accuracy_score(y_true, y_pred)


def precision_score(
    y_true: list[int] = None, y_pred: list[int] = None, average: str = "weighted"
) -> float:
    """
    precision_score: precision score
    Args:
        y_true (list[int], optional): desired style labels from source text. Defaults to None.
        y_pred (list[int], optional): generated text's hedge label. Defaults to None.
        average (str, optional): type of precision score. Defaults to "weighted"
    Raises:
        ValueError: y_true and y_pred must have the same length
    Returns:
        float: precision score
    """
    if y_true is None:
        y_true = []
    if y_pred is None:
        y_pred = []
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return metrics.precision_score(y_true, y_pred, average=average)


def recall_score(
    y_true: list[int] = None, y_pred: list[int] = None, average: str = "weighted"
) -> float:
    """
    recall_score: recall score
    Args:
        y_true (list[int], optional): desired style labels from source text. Defaults to None.
        y_pred (list[int], optional): generated text's hedge label. Defaults to None.
        average (str, optional): type of recall score. Defaults to "weighted"
    Raises:
        ValueError: y_true and y_pred must have the same length
    Returns:
        float: recall score
    """
    if y_true is None:
        y_true = []
    if y_pred is None:
        y_pred = []
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return metrics.recall_score(y_true, y_pred, average=average)


def classification_report(y_true: list[int] = None, y_pred: list[int] = None) -> dict:
    """
    classification_report: classification report
    Args:
        y_true (list[int], optional): desired style labels from source text. Defaults to None.
        y_pred (list[int], optional): generated text's hedge label. Defaults to None.
    Raises:
        ValueError: y_true and y_pred must have the same length
    Returns:
        dict: classification report dict representation.
    """
    if y_true is None:
        y_true = []
    if y_pred is None:
        y_pred = []
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return metrics.classification_report(y_true, y_pred)


def confusion_matrix(y_true: list[int] = None, y_pred: list[int] = None) -> np.ndarray:
    """
    confusion_matrix: confusion matrix of the classification task.
    Args:
        y_true (list[int], optional): desired style labels from source text. Defaults to None.
        y_pred (list[int], optional): generated text's hedge label. Defaults to None.
    Raises:
        ValueError: y_true and y_pred must have the same length.
    Returns:
        np.ndarray: confusion matrix in numpy array format
    """
    if y_true is None:
        y_true = []
    if y_pred is None:
        y_pred = []
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return metrics.confusion_matrix(y_true, y_pred)


def roc_auc_score(
    y_true: list[int] = None, y_pred: list[int] = None, average: str = "weighted"
) -> np.ndarray:
    """
    roc_auc_score: roc auc score
    Args:
        y_true (list[int], optional): desired style labels from source text. Defaults to None.
        y_pred (list[int], optional): generated text's hedge label. Defaults to None.
        average (str, optional): type of roc auc score. Defaults to "weighted"
    Raises:
        ValueError: y_true and y_pred must have the same length.
    Returns:
        np.ndarray: roc auc score in numpy array format.
    """
    if y_true is None:
        y_true = []
    if y_pred is None:
        y_pred = []
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return metrics.roc_auc_score(y_true, y_pred, average=average)
