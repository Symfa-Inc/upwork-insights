import logging
import os
import pickle
from typing import Tuple
from datetime import datetime

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from supervised.automl import AutoML

from src import PROJECT_DIR
from src.data.feature_processors import NumericProcessor

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset from a parquet file."""
    log.info(f"Loading dataset from: {file_path}")
    df = pd.read_parquet(file_path)
    log.info(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def load_scaler(column_name: str, pipeline_path: str) -> StandardScaler:
    """Load a scaler for the specified column from the saved pipeline."""
    if not column_name.startswith("wh_"):
        raise ValueError("Column name must have prefix 'wh_'.")

    log.info(f"Loading pipeline from: {pipeline_path}")
    with open(pipeline_path, 'rb') as file:
        pipeline = pickle.load(file)

    for processor in pipeline.processors:
        if isinstance(processor, NumericProcessor) and processor.column_name == column_name:
            log.info(f"Scaler for column '{column_name}' successfully loaded.")
            return processor.scaler

    raise ValueError(f"Scaler for column '{column_name}' not found in the pipeline.")


def prepare_train_test_split(
        df: pd.DataFrame,
        target_column: str,
        test_size: float,
        random_state: int,  # Default: 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into training and testing sets, dropping columns with prefix 'wh_'."""
    log.info("Performing train-test split.")
    x = df.drop(columns=[col for col in df.columns if col.startswith('wh_')])
    y = df[target_column]  # Adjusted to handle only one target column
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    log.info(
        f"Train-test split complete: {x_train.shape[0]} train samples, {x_test.shape[0]} test samples."
    )
    return x_train, x_test, y_train, y_test


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    save_dir: str,
    automl_params: dict,
    target: str,
    scaler: StandardScaler,
) -> None:
    """Train and evaluate AutoML model for the specific target variable."""
    log.info(f"Training model for target: {target}")
    timestamp = datetime.now().strftime("%H%M_%d%m%Y")
    target_save_dir = os.path.join(save_dir, f"{target}_{timestamp}")
    os.makedirs(target_save_dir, exist_ok=True)

    # Initialize and train AutoML for the specific target
    automl = AutoML(results_path=target_save_dir, **automl_params)
    automl.fit(x_train, y_train)

    # Inverse transform the target
    predictions = automl.predict(x_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    y_test = scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    log.info(f"Target: {target}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}")

    # Save predictions and evaluation
    predictions_file = os.path.join(target_save_dir, f"{target}_predictions.csv")
    pd.DataFrame({"Actual": y_test, "Predicted": predictions}).to_csv(
        predictions_file, index=False
    )
    log.info(f"Predictions for target '{target}' saved to {predictions_file}.")


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='train',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    data_path = cfg.files.final
    pipeline_path = cfg.files.pipeline
    save_dir = os.path.join(cfg.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    df = load_dataset(data_path)

    # Load the scaler
    scaler = load_scaler(cfg.target_column, pipeline_path)

    # Train-test split
    x_train, x_test, y_train, y_test = prepare_train_test_split(
        df, cfg.target_column, cfg.test_size, cfg.random_state
    )

    # Train MLJAR AutoML model
    train_model(
        x_train, y_train, x_test, y_test, save_dir, cfg.automl_params, cfg.target_column, scaler
    )

    log.info("Training complete!")


if __name__ == '__main__':
    main()
