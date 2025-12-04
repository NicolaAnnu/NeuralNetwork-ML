import numpy as np
import pandas as pd
from copy import deepcopy


class Metrics:
    accuracy_results = np.array([])
    def __init__(self, metrics=None):
        self.metrics = deepcopy(metrics or [])
        self.reset()

    def reset(self) -> None:
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
        self.confusion_matrix = pd.DataFrame()

    def compute_results(self, oracle: np.ndarray, out: np.ndarray) -> None:
        if not ("accuracy" in self.metrics or "accuracy_score" in self.metrics):
            return


        labels = np.unique(np.concatenate([oracle, out]))
        num_classes = len(labels)

        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(oracle, out):
            true_idx = np.where(labels == true)[0][0]
            pred_idx = np.where(labels == pred)[0][0]
            confusion_matrix[true_idx, pred_idx] += 1

        cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
        if num_classes == 2:
            self.true_pos = cm.loc[1, 1]
            self.true_neg = cm.loc[0, 0]
            self.false_pos = cm.loc[0, 1]
            self.false_neg = cm.loc[1, 0]
        else:
            for idx, label in enumerate(labels):
                self.true_pos += cm.loc[label, label]
                self.true_neg += np.sum(np.delete(np.delete(cm.values, idx, axis=0), idx, axis=1))
                self.false_pos += np.sum(cm.loc[:, label]) - cm.loc[label, label]
                self.false_neg += np.sum(cm.loc[label, :]) - cm.loc[label, label]

        self.confusion_matrix = cm

    def accuracy_score(self) -> float:
        correct = self.true_pos + self.true_neg
        total = self.true_pos + self.true_neg + self.false_pos + self.false_neg
        res = 0 if total == 0 else correct / total
        self.accuracy_results = np.append(self.accuracy_results, res)
        return res

    def get_best_accuracy(self) -> float:
        return float(np.max(self.accuracy_results)) if len(self.accuracy_results) else 0.0

    def precision(self) -> float:
        denom = self.true_pos + self.false_pos
        return 0 if denom == 0 else self.true_pos / denom

    def recall(self) -> float:
        denom = self.true_pos + self.false_neg
        return 0 if denom == 0 else self.true_pos / denom

    def f1(self) -> float:
        precision = self.precision()
        recall = self.recall()
        denom = precision + recall
        return 0 if denom == 0 else 2 * (precision * recall) / denom
    
    def error_rate(self) -> float:
        total = self.true_pos + self.true_neg + self.false_pos + self.false_neg
        return 0 if total == 0 else (self.false_pos + self.false_neg) / total

    def specificity(self) -> float:
        denom = self.true_neg + self.false_pos
        return 0 if denom == 0 else self.true_neg / denom

    def false_positive_rate(self) -> float:
        denom = self.true_neg + self.false_pos
        return 0 if denom == 0 else self.false_pos / denom

    def false_negative_rate(self) -> float:
        denom = self.true_pos + self.false_neg
        return 0 if denom == 0 else self.false_neg / denom

    def plot(self) -> None:
        if "accuracy" in self.metrics:
            import matplotlib.pyplot as plt

            plt.plot(self.accuracy_results)
            plt.show()