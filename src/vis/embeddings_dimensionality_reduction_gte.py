import logging
import os

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA

from src import PROJECT_DIR
from src.data.feature_processors.ppa import PCAWithPreProcessing
from src.data.utils import get_embeddings_gte
from src.vis.embeddings_dimensionality_reduction_openai import (
    calculate_and_save_embeddings,
    fit_dimensionality_reduction_methods,
    visualize_variances,
)

log = logging.getLogger()
log.setLevel(logging.INFO)
if log.hasHandlers():
    log.handlers.clear()

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s][%(levelname)s] - %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S',
)
console_handler.setFormatter(formatter)
log.addHandler(console_handler)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='embeddings_dimensionality_reduction_gte',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    data_path = str(os.path.join(PROJECT_DIR, cfg.data_path))
    embeddings_path = str(os.path.join(PROJECT_DIR, cfg.embeddings_path))
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    text_columns = cfg.text_columns
    missing_columns = [
        column
        for column in text_columns
        if not os.path.exists(os.path.join(embeddings_path, f"{column}_thenlper/gte-large.parquet"))
    ]

    if missing_columns:
        calculate_and_save_embeddings(
            data_path,
            embeddings_path,
            missing_columns,
            batch_size=500,
            model='thenlper/gte-large',
            model_function=get_embeddings_gte,
        )

    methods = {
        'PCA': PCA(n_components=3072),
        'PCAWithPreProcessing (normalize)': PCAWithPreProcessing(
            n_components=3072,
            preprocessing_method='normalize',
        ),
        'PCAWithPreProcessing (standardize)': PCAWithPreProcessing(
            n_components=3072,
            preprocessing_method='standardize',
        ),
    }
    for column in text_columns:
        df = pd.read_parquet(os.path.join(embeddings_path, f"{column}_thenlper/gte-large.parquet"))
        explained_variances, cumulative_variances = fit_dimensionality_reduction_methods(
            df,
            column,
            methods,
            save_dir,
        )
        save_path = os.path.join(save_dir, f'explained_variance_decay_{column}.png')
        visualize_variances(
            explained_variances,
            cumulative_variances,
            feature_name=column,
            save_path=save_path,
        )

    log.info('Training complete!')


if __name__ == '__main__':
    main()
