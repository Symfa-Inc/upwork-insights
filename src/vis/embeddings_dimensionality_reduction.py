import logging
import os
from typing import List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import mean_squared_error

from src import PROJECT_DIR
from src.data.utils import get_embeddings

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


def calculate_umap_explained_variance(data, n_components):
    """Approximate explained variance for UMAP using reconstruction error.

    Args:
        data (np.ndarray): Original high-dimensional data.
        n_components (int): Number of UMAP components.

    Returns:
        float: Approximation of explained variance.
    """
    umap_model = umap.UMAP(n_components=n_components)
    low_dim_embedding = umap_model.fit_transform(data)
    reconstructed_data = umap_model.inverse_transform(low_dim_embedding)

    # Calculate reconstruction error
    reconstruction_error = mean_squared_error(data, reconstructed_data)
    total_variance = np.var(data, axis=0).sum()

    # Approximate explained variance
    explained_variance = 1 - (reconstruction_error / total_variance)
    return explained_variance


def fit_dimensionality_reduction_methods(df: pd.DataFrame) -> Tuple[dict, dict]:
    """Fit PCA, UMAP, and other dimensionality reduction methods and compute explained variances.

    Args:
        df (pd.DataFrame): The DataFrame containing embeddings (with 'id' as the index).

    Returns:
        dict: A dictionary containing explained variances for each method.
        dict: A dictionary containing cumulative variances for each method.
    """
    # Extract embedding values as a NumPy array
    embeddings = df.values

    # Limit n_components to 1500
    max_components = min(1500, embeddings.shape[1])

    methods = {
        'PCA': PCA(n_components=max_components),
        'KernelPCA (rbf)': KernelPCA(
            n_components=max_components,
            kernel='rbf',
            fit_inverse_transform=True,
        ),
        'KernelPCA (poly)': KernelPCA(
            n_components=max_components,
            kernel='poly',
            fit_inverse_transform=True,
        ),
        'KernelPCA (sigmoid)': KernelPCA(
            n_components=max_components,
            kernel='sigmoid',
            coef0=1,
            fit_inverse_transform=True,
        ),
        'KernelPCA (cosine)': KernelPCA(
            n_components=max_components,
            kernel='cosine',
            fit_inverse_transform=True,
        ),
    }

    explained_variances = {}
    cumulative_variances = {}

    for name, method in methods.items():
        log.info(f"Fitting method {name}")
        try:
            if isinstance(method, PCA):
                method.fit(embeddings)
                explained_variance = method.explained_variance_ratio_
            else:
                # Approximate explained variance for UMAP
                embeddings_transformed = method.fit_transform(embeddings)
                singular_values = np.linalg.svd(embeddings_transformed, compute_uv=False)
                explained_variance = singular_values**2 / np.sum(singular_values**2)

            explained_variances[name] = explained_variance
            cumulative_variances[name] = np.cumsum(explained_variance)
        except ValueError as e:
            logging.error(f"Error with method {name}: {e}")
            log.info(f"Skipping method {name} due to an error: {e}")
            continue

    # Special handling for UMAP
    log.info('Fitting method UMAP')
    try:
        # Step 1: Select a subset of size 1600
        subset_size = 1600
        subset = embeddings[
            :subset_size,
            :,
        ]  # Take the first `subset_size` rows and first 1500 features

        # Step 2: Validate dimensions
        if subset.shape[0] < 2 or subset.shape[1] < 2:
            raise ValueError('UMAP requires at least a 2D dataset with both rows and columns.')

        # Step 3: Define intervals for calculating explained variance
        intervals = (
            list(range(2, 11))  # First 10 components (step 1)
            + list(range(15, 101, 5))  # 15 to 100 components (step 5)
            + list(range(110, 501, 50))  # 110 to 500 components (step 10)
            + list(range(600, max_components + 1, 100))  # 600+ components (step 100)
        )
        log.info(f"Intervals for UMAP: {intervals}")

        # Step 4: Calculate explained variance for specified intervals
        cumulative_variances_umap = []
        for n_components in intervals:
            cumulative_variance = calculate_umap_explained_variance(
                subset,
                n_components=n_components,
            )
            cumulative_variances_umap.append(cumulative_variance)

        # Step 5: Compute explained variance for each interval
        explained_variances_umap = np.diff(
            [0] + cumulative_variances_umap,
        )  # Derive incremental variance

        # Step 6: Save results
        explained_variances['UMAP'] = explained_variances_umap
        cumulative_variances['UMAP'] = cumulative_variances_umap

    except ValueError as e:
        logging.error(f"Error with UMAP: {e}")
        log.info(f"Skipping UMAP due to an error: {e}")

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
    # Find the best method (least components to reach 80% variance)
    best_method = min(
        cumulative_variances.keys(),
        key=lambda x: np.argmax(cumulative_variances[x] >= 0.8) + 1,
    )

    # Plotting
    plt.figure(figsize=(14, 6))

    # Subplot 1: Explained Variance Decay
    plt.subplot(1, 2, 1)
    for name, variance in explained_variances.items():
        intervals = list(range(1, len(variance) + 1))
        if name == 'UMAP':
            intervals = (
                list(range(2, 11))  # First 10 components (step 1)
                + list(range(15, 101, 5))  # 15 to 100 components (step 5)
                + list(range(110, 501, 50))  # 110 to 500 components (step 10)
                + list(range(600, 1500 + 1, 100))  # 600+ components (step 100)
            )
        plt.plot(intervals, variance * 100, label=name)
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance (%)')
    plt.title(f"Explained Variance Decay ({feature_name})")
    plt.legend()

    # Subplot 2: Cumulative Variance
    plt.subplot(1, 2, 2)
    for name, cumulative_variance in cumulative_variances.items():
        intervals = list(range(1, len(cumulative_variance) + 1))
        if name == 'UMAP':
            intervals = (
                list(range(2, 11))  # First 10 components (step 1)
                + list(range(15, 101, 5))  # 15 to 100 components (step 5)
                + list(range(110, 501, 50))  # 110 to 500 components (step 10)
                + list(range(600, 1500 + 1, 100))  # 600+ components (step 100)
            )
        plt.plot(intervals, cumulative_variance * 100, label=name)
    plt.axhline(80, color='red', linestyle='--', label='80% Threshold')

    if best_method:
        best_components = np.argmax(cumulative_variances[best_method] >= 0.8) + 1
        plt.axvline(best_components, color='blue', linestyle='--', label=f"Best ({best_method})")
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title(f"Cumulative Variance ({feature_name})")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', dpi=600)
        log.info(f"Plot saved to {save_path}")
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

    for column in text_columns:
        df = pd.read_parquet(os.path.join(embeddings_path, f'{column}_embeddings.parquet'))
        explained_variances, cumulative_variances = fit_dimensionality_reduction_methods(df)
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
