import logging
import os
from typing import Tuple, List

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
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
        target_columns: List[str],
        test_size: float,
        random_state: int,  # 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into training and testing sets with multiple target variables."""
    log.info("Performing train-test split.")
    x = df.drop(columns=target_columns)
    y = df[target_columns]
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
    targets: List[str],
) -> None:
    """Train and evaluate separate AutoML models for each target variable."""
    for target in targets:
        log.info(f"Training model for target: {target}")
        target_save_dir = os.path.join(save_dir, target)
        os.makedirs(target_save_dir, exist_ok=True)

        # Initialize and train AutoML for the specific target
        automl = AutoML(results_path=target_save_dir, **automl_params)
        automl.fit(x_train, y_train[target])

        # Evaluate the model
        predictions = automl.predict(x_test)
        mse = ((y_test[target] - predictions) ** 2).mean()
        log.info(f"Target: {target}, MSE: {mse:.4f}")

        # Save predictions and evaluation
        predictions_file = os.path.join(target_save_dir, f"{target}_predictions.csv")
        pd.DataFrame({"Actual": y_test[target], "Predicted": predictions}).to_csv(
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
    save_dir = os.path.join(cfg.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    df = load_dataset(data_path)

    # Train-test split
    x_train, x_test, y_train, y_test = prepare_train_test_split(
        df, cfg.target_columns, cfg.test_size, cfg.random_state
    )

    # Train MLJAR AutoML model
    train_model(x_train, y_train, x_test, y_test, save_dir, cfg.automl_params, cfg.target_columns)

    log.info("Training complete!")


if __name__ == '__main__':
    main()