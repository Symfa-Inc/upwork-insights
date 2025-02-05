import os
from typing import Dict, Tuple, Type, Union

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA, KernelPCA

from src import PROJECT_DIR
from src.data.feature_processors.ppa import PCAWithPreProcessing
from src.data.utils import get_embeddings, get_embeddings_gte
from src.vis.utils import (
    calculate_and_save_embeddings,
    fit_dimensionality_reduction_methods,
    log,
    visualize_variances,
)


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='embeddings_dimensionality_reduction',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    data_path = str(os.path.join(PROJECT_DIR, cfg.data_path))
    embeddings_path = str(os.path.join(PROJECT_DIR, cfg.embeddings_path))

    embeddings_model_url = cfg.embeddings_model_url
    embeddings_model = embeddings_model_url.split('/')[-1]
    embeddings_function = (
        get_embeddings if embeddings_model.startswith('text-embedding-3') else get_embeddings_gte
    )

    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir + embeddings_model))

    max_n_components = cfg.max_n_components[embeddings_model_url]
    batch_size = cfg.batch_size[embeddings_model_url]

    os.makedirs(save_dir, exist_ok=True)

    text_columns = cfg.text_columns
    missing_columns = [
        column
        for column in text_columns
        if not os.path.exists(os.path.join(embeddings_path, f"{column}_{embeddings_model}.parquet"))
    ]

    if missing_columns:
        calculate_and_save_embeddings(
            data_path,
            embeddings_path,
            missing_columns,
            batch_size=batch_size,
            model=embeddings_model_url,
            model_function=embeddings_function,
        )

    methods: Dict[str, Tuple[Type[Union[PCA, KernelPCA, PCAWithPreProcessing]], dict]] = {
        'PCA': (PCA, dict(n_components=max_n_components)),
        'PCAWithPreProcessing (normalize)': (
            PCAWithPreProcessing,
            dict(n_components=max_n_components, preprocessing_method='normalize'),
        ),
        'PCAWithPreProcessing (standardize)': (
            PCAWithPreProcessing,
            dict(n_components=max_n_components, preprocessing_method='standardize'),
        ),
    }
    for column in text_columns:
        df = pd.read_parquet(os.path.join(embeddings_path, f"{column}_{embeddings_model}.parquet"))
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
