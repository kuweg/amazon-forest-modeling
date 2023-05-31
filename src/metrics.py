from torchmetrics import MetricCollection, F1Score, Precision, Recall


def get_metrics(**kwargs) -> MetricCollection:
    """Build a `MetricCollection`.

    Build a `MetricCollection` object with
    F1, Precision, Recall metrics.

    Args:
        kwargs: arguments for metric objects.

    Returns:
        MetricCollection:
        specified metrics `inside MetricCollection`
    """
    return MetricCollection(
        {
            'f1': F1Score(**kwargs),
            'precision': Precision(**kwargs),
            'recall': Recall(**kwargs),
        },
    )
