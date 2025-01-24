import logging
import os
from typing import Dict, List, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA, KernelPCA

from src import PROJECT_DIR
from src.data.feature_processors.ppa import PCAWithPreProcessing
from src.data.utils import get_embeddings, load_model_from_pickle, save_model_to_pickle

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


def calculate_and_save_embeddings(
    input_path: str,
    output_dir: str,
    text_columns: List[str],
    id_column: str = 'id',
    batch_size: int = 1000,
    model='text-embedding-3-large',
):
    # Load the dataset
    df = pd.read_csv(input_path, usecols=[id_column] + text_columns, index_col=id_column)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for column in text_columns:
        log.info(f"Processing embeddings for column: {column}")

        # Extract text data
        texts = df[column].tolist()

        # Generate embeddings
        embeddings = get_embeddings(texts, batch_size=batch_size, model=model)

        # Create a DataFrame for embeddings
        embedding_columns = [f"EMBEDDING_{i + 1}" for i in range(embeddings.shape[1])]
        embeddings_df = pd.DataFrame(embeddings, columns=embedding_columns, index=df.index)

        # Reset index to include ID column
        embeddings_df.reset_index(inplace=True)

        # Save the DataFrame to Parquet, preserving the index
        output_file = os.path.join(output_dir, f"{column}_embeddings.parquet")
        embeddings_df.to_parquet(output_file, index=True)
        log.info(f"Saved embeddings to {output_file}")


def fit_dimensionality_reduction_methods(
    df: pd.DataFrame,
    feature_name: str,
    methods: Dict[str, Union[PCA, KernelPCA, PCAWithPreProcessing]],
    save_dir: str,
) -> Tuple[dict, dict]:
    """Fit dimensionality reduction methods and save/load models as pickles."""
    # Extract embedding values as a NumPy array
    embeddings = df.values
    sample_size = embeddings.shape[0]

    explained_variances = {}
    cumulative_variances = {}

    os.makedirs(save_dir, exist_ok=True)

    for name, method in methods.items():
        model_path = os.path.join(save_dir, f"{name}_{feature_name}_{sample_size}.pkl")
        if os.path.exists(model_path):
            method = load_model_from_pickle(model_path)
            if method is None:
                log.info(f"Skipping method {name} due to loading error.")
                continue
        else:
            log.info(f"Fitting method {name}")
            try:
                method.fit(embeddings)
                save_model_to_pickle(method, model_path)
            except ValueError as e:
                log.error(f"Error with method {name}: {e}")
                continue

        if isinstance(method, PCA) or hasattr(method, 'pca_'):
            # Extract explained variance from PCA or PCAWithPreProcessing
            explained_variance = (
                method.explained_variance_ratio_
                if isinstance(method, PCA)
                else method.pca_.explained_variance_ratio_
            )
        else:
            # Approximation for KernelPCA
            embeddings_transformed = method.transform(embeddings)
            singular_values = np.linalg.svd(embeddings_transformed, compute_uv=False)
            explained_variance = singular_values**2 / np.sum(singular_values**2)

        explained_variances[name] = explained_variance
        cumulative_variances[name] = np.cumsum(explained_variance)

    return explained_variances, cumulative_variances


def visualize_variances(
    explained_variances: dict,
    cumulative_variances: dict,
    feature_name: str,
    save_path: str = None,
):
    """Visualize explained variance and cumulative variance for different methods.

    Args:
        explained_variances (dict): Dictionary of explained variances for each method.
        cumulative_variances (dict): Dictionary of cumulative variances for each method.
        feature_name (str): Name of the feature for labeling the plot.
        save_path (str, optional): Path to save the plot. If None, only visualizes in notebooks.
    """
    # Sort methods by the value of the first component (weakest to strongest)
    sorted_methods = sorted(
        explained_variances.keys(),
        key=lambda x: explained_variances[x][0],
    )

    # Find the best method (least components to reach 80% variance)
    best_method = min(
        cumulative_variances.keys(),
        key=lambda x: np.argmax(cumulative_variances[x] >= 0.8) + 1,
    )
    best_components = np.argmax(cumulative_variances[best_method] >= 0.8) + 1

    # Plotting
    plt.figure(figsize=(14, 6))

    # Subplot 1: Explained Variance Decay
    plt.subplot(1, 2, 1)
    for name in sorted_methods:
        variance = explained_variances[name][:500]  # Only plot the first 500 components
        plt.plot(
            range(1, len(variance) + 1),
            variance * 100,
            label=f"{name} (1st: {variance[0] * 100:.2f}%)",
        )
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance (%)')
    plt.title(f"Explained Variance Decay ({feature_name})")
    plt.ylim(0, 50)  # Set y-axis limits
    plt.grid(True)  # Add grid
    plt.legend()

    # Subplot 2: Cumulative Variance
    plt.subplot(1, 2, 2)
    for name in sorted_methods:
        cumulative_variance = cumulative_variances[name][:500]  # Only plot the first 500 components
        plt.plot(
            range(1, len(cumulative_variance) + 1),
            cumulative_variance * 100,
            label=name,
        )
    plt.axhline(80, color='red', linestyle='--', label='80% Threshold')
    plt.axvline(
        best_components,
        color='blue',
        linestyle='--',
        label=f"Best ({best_method}, {best_components})",
    )
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title(f"Cumulative Variance ({feature_name})")
    plt.ylim(0, 100)  # Set y-axis limits
    plt.grid(True)  # Add grid
    plt.legend()

    plt.tight_layout()

    if save_path:
        # Ensure the directory exists
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            log.info(f"Created directory: {save_dir}")
        try:
            plt.savefig(save_path, format='png', dpi=600)
            log.info(f"Plot saved to {save_path}")
        except Exception as e:
            log.error(f"Failed to save plot to {save_path}: {e}")
    else:
        plt.show()

    # Clear the figure to prevent overlapping in notebooks
    plt.close()


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
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    text_columns = cfg.text_columns
    missing_columns = [
        column
        for column in text_columns
        if not os.path.exists(os.path.join(embeddings_path, f"{column}_embeddings.parquet"))
    ]

    if missing_columns:
        calculate_and_save_embeddings(data_path, embeddings_path, missing_columns)

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
        df = pd.read_parquet(os.path.join(embeddings_path, f'{column}_embeddings.parquet'))
        explained_variances, cumulative_variances = fit_dimensionality_reduction_methods(
            df,
            column,
            methods,
            save_dir,
        )
        save_path = os.path.join(embeddings_path, f'explained_variance_decay_{column}.png')
        visualize_variances(
            explained_variances,
            cumulative_variances,
            feature_name=column,
            save_path=save_path,
        )

    log.info('Training complete!')


if __name__ == '__main__':
    main()
