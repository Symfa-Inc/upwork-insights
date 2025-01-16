import logging
import os
from typing import Tuple, List
from datetime import datetime

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML

from src import PROJECT_DIR

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset from a parquet file."""
    log.info(f"Loading dataset from: {file_path}")
    df = pd.read_parquet(file_path)
    log.info(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def prepare_train_test_split(
        df: pd.DataFrame,
        target_column: str,
        test_size: float,
        seed: int,  # Default: 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into training and testing sets, dropping columns with prefix 'wh_'."""
    log.info("Performing train-test split.")
    x = df.drop(columns=[col for col in df.columns if col.startswith('wh_')])
    y = df[target_column]  # Adjusted to handle only one target column
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed
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
) -> None:
    """Train and evaluate AutoML model for the specific target variable."""
    log.info(f"Training model for target: {target}")
    timestamp = datetime.now().strftime("%H%M_%d%m%Y")
    target_save_dir = os.path.join(save_dir, f"{target}_{timestamp}")
    os.makedirs(target_save_dir, exist_ok=True)

    # Initialize and train AutoML for the specific target
    automl = AutoML(results_path=target_save_dir, **automl_params)
    automl.fit(x_train, y_train)

    # Evaluate the model
    predictions = automl.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    log.info(f"Target: {target}, MSE: {mse:.4f}")

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
    data_path = str(os.path.join(PROJECT_DIR, cfg.data_path))
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    df = load_dataset(data_path)

    # Split dataset into training and testing sets
    x_train, x_test, y_train, y_test = prepare_train_test_split(
        df=df,
        target_column=cfg.target_column,
        test_size=cfg.test_size,
        seed=cfg.seed,
    )

    # Train MLJAR AutoML model
    train_model(x_train, y_train, x_test, y_test, save_dir, cfg.automl_params, cfg.target_column)

    log.info("Training complete!")


if __name__ == '__main__':
    main()
