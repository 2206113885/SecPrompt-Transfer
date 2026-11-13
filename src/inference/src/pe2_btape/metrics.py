from __future__ import annotations

from dataclasses import asdict, dataclass


INVALID = "__INVALID__"


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    matthews_correlation: float
    per_class: dict[str, dict[str, float | int]]
    invalid_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_metrics(
    predictions: list[str], gold: list[str], labels: list[str]
) -> Metrics:
    if len(predictions) != len(gold):
        raise ValueError("Predictions and gold labels must have the same length")
    if not gold:
        raise ValueError("Cannot score an empty subset")

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    weighted_sum = 0.0

    for label in labels:
        tp = sum(p == label and y == label for p, y in zip(predictions, gold))
        fp = sum(p == label and y != label for p, y in zip(predictions, gold))
        fn = sum(p != label and y == label for p, y in zip(predictions, gold))
        support = sum(y == label for y in gold)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
        recall_values.append(recall)
        weighted_sum += f1 * support

    if len(labels) != 2:
        raise ValueError("The frozen study requires exactly two labels")
    negative, positive = labels
    tp = sum(p == positive and y == positive for p, y in zip(predictions, gold))
    tn = sum(p == negative and y == negative for p, y in zip(predictions, gold))
    fp = sum(p == positive and y == negative for p, y in zip(predictions, gold))
    fn = sum(p == negative and y == positive for p, y in zip(predictions, gold))
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0

    return Metrics(
        accuracy=sum(p == y for p, y in zip(predictions, gold)) / len(gold),
        macro_f1=sum(f1_values) / len(labels),
        weighted_f1=weighted_sum / len(gold),
        balanced_accuracy=sum(recall_values) / len(labels),
        matthews_correlation=mcc,
        per_class=per_class,
        invalid_count=sum(p == INVALID for p in predictions),
    )
