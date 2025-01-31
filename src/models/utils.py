import pandas as pd

def symmetric_mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE)."""
    denominator = (abs(y_true) + abs(y_pred)) / 2
    smape = (abs(y_true - y_pred) / denominator).mean() * 100
    return smape