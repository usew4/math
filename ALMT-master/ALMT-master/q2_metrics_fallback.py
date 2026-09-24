"""Small NumPy equivalents for the three-class metrics used by Q2 scripts."""
from __future__ import annotations

import numpy as np


def accuracy_score(y_true, y_pred) -> float:
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    if true.size == 0 or true.shape != pred.shape:
        raise ValueError("Expected equally sized nonempty label arrays")
    return float(np.mean(true == pred))


def confusion_matrix(y_true, y_pred, labels):
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError("True and predicted label arrays have different lengths")
    labels = list(labels)
    result = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for row, actual in enumerate(labels):
        for column, predicted in enumerate(labels):
            result[row, column] = np.count_nonzero((true == actual) & (pred == predicted))
    return result


def recall_score(y_true, y_pred, labels, average=None, zero_division=0):
    if average is not None or zero_division != 0:
        raise ValueError("This fallback supports average=None and zero_division=0")
    counts = confusion_matrix(y_true, y_pred, labels)
    positives = np.diag(counts).astype(np.float64)
    support = counts.sum(axis=1)
    return np.divide(positives, support, out=np.zeros_like(positives), where=support != 0)


def f1_score(y_true, y_pred, labels, average="macro", zero_division=0):
    if average != "macro" or zero_division != 0:
        raise ValueError("This fallback supports average='macro' and zero_division=0")
    counts = confusion_matrix(y_true, y_pred, labels)
    positives = np.diag(counts).astype(np.float64)
    denominators = counts.sum(axis=1) + counts.sum(axis=0)
    per_class = np.divide(2 * positives, denominators,
                          out=np.zeros_like(positives), where=denominators != 0)
    return float(per_class.mean())
