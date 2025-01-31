import os
from typing import Dict, Tuple, Type

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import KMeans, MiniBatchKMeans

from src import PROJECT_DIR
from src.data.utils import get_embeddings, get_embeddings_gte
from src.vis.utils import (
    CosineKMeans,
    calculate_and_save_embeddings,
    fit_clustering_methods,
    log,
    visualize_clustering_elbow,
)


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='embeddings_clustering',
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

    # Define the pipeline for KMeans with cosine distance

    # Define clustering methods
    methods: Dict[str, Tuple[Type[KMeans], dict]] = {
        # Standard KMeans with default algorithm
        'KMeans': (KMeans, dict(random_state=42)),
        # KMeans with 'elkan' algorithm (optimized for Euclidean distance)
        'KMeans (Elkan)': (KMeans, dict(random_state=42, algorithm='elkan')),
        # MiniBatch KMeans for faster computations on large datasets
        'MiniBatchKMeans': (MiniBatchKMeans, dict(random_state=42)),
        # KMeans with Cosine Distance
        'KMeans (Cosine)': (CosineKMeans, dict(random_state=42)),
        # KMeans with Cosine Distance and alternative algorithm ('lloyd')
        'KMeans (Cosine, Lloyd)': (CosineKMeans, dict(random_state=42, algorithm='lloyd')),
        # Bisecting KMeans for recursive splitting
        'BisectingKMeans': (KMeans, dict(random_state=42, n_init=10)),
    }

    for column in text_columns:
        # Load embeddings for the current column
        df = pd.read_parquet(os.path.join(embeddings_path, f"{column}_{embeddings_model}.parquet"))

        # Fit clustering methods and calculate WCSS
        wcss_scores = fit_clustering_methods(
            df,
            column,
            methods,
            save_dir,
        )

        # Visualize clustering elbow plots
        save_path = os.path.join(save_dir, f'clustering_elbow_{column}.png')
        visualize_clustering_elbow(
            wcss_scores,
            feature_name=column,
            save_path=save_path,
        )

    log.info('Training complete!')


if __name__ == '__main__':
    main()
