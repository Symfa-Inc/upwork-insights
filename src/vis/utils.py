import logging
import os
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import normalize

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


class CosineKMeans(BaseEstimator, ClusterMixin):
    """A wrapper for KMeans that normalizes data to use cosine distance."""

    def __init__(self, n_clusters=8, random_state=None, algorithm='lloyd'):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.algorithm = algorithm
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            algorithm=self.algorithm,
        )

    def fit(self, X, y=None):
        """Normalize data and fit KMeans."""
        X_normalized = normalize(X)
        self.model.fit(X_normalized)
        return self

    def predict(self, X):
        """Normalize data and predict cluster labels."""
        X_normalized = normalize(X)
        return self.model.predict(X_normalized)

    def fit_predict(self, X, y=None):
        """Normalize data, fit KMeans, and predict cluster labels."""
        X_normalized = normalize(X)
        return self.model.fit_predict(X_normalized)

    @property
    def inertia_(self):
        """Return the inertia of the fitted model."""
        return self.model.inertia_


def calculate_and_save_embeddings(
    input_path: str,
    output_dir: str,
    text_columns: List[str],
    id_column: str = 'id',
    batch_size: int = 1000,
    model='text-embedding-3-large',
    model_function=get_embeddings,
):
    # Load the dataset
    df = pd.read_csv(input_path, usecols=[id_column] + text_columns, index_col=id_column)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    model_name = model.split('/')[-1]

    for column in text_columns:
        log.info(f"Processing embeddings for column: {column}")

        # Extract text data
        texts = df[column].tolist()

        # Generate embeddings
        embeddings = model_function(texts, batch_size=batch_size, model=model)

        # Create a DataFrame for embeddings
        embedding_columns = [f"EMBEDDING_{i + 1}" for i in range(embeddings.shape[1])]
        embeddings_df = pd.DataFrame(embeddings, columns=embedding_columns, index=df.index)

        # Reset index to include ID column
        embeddings_df.reset_index(inplace=True)

        # Save the DataFrame to Parquet, preserving the index
        output_file = os.path.join(output_dir, f"{column}_{model_name}.parquet")
        embeddings_df.to_parquet(output_file, index=True)
        log.info(f"Saved embeddings to {output_file}")


def calculate_best_k(wcss_scores: dict) -> dict:
    """Calculate the best number of clusters (k) for each method using the elbow method.

    Args:
        wcss_scores (dict): Dictionary with WCSS values for each method.

    Returns:
        dict: Best k values for each method.
    """
    best_k_values = {}
    for method, wcss in wcss_scores.items():
        # Calculate the second derivative (rate of change of WCSS)
        deltas = np.diff(wcss, n=1)  # First-order differences
        delta2 = np.diff(deltas)  # Second-order differences
        # The elbow point is where the second derivative is minimal
        elbow_k = np.argmin(delta2) + 2  # Add 2 because indexing starts from k=1
        best_k_values[method] = elbow_k
    return best_k_values


def fit_dimensionality_reduction_methods(
    df: pd.DataFrame,
    feature_name: str,
    methods: Dict[str, Tuple[Type[Union[PCA, KernelPCA, PCAWithPreProcessing]], dict]],
    save_dir: str,
) -> Tuple[dict, dict]:
    """Fit dimensionality reduction methods and save/load models as pickles."""
    # Extract embedding values as a NumPy array
    embeddings = df.values
    sample_size = embeddings.shape[0]

    explained_variances = {}
    cumulative_variances = {}

    os.makedirs(save_dir, exist_ok=True)

    for name, (method_class, params) in methods.items():
        model_path = os.path.join(save_dir, f"{name}_{feature_name}_{sample_size}.pkl")
        if os.path.exists(model_path):
            method = load_model_from_pickle(model_path)
            if method is None:
                log.info(f"Skipping method {name} due to loading error.")
                continue
        else:
            log.info(f"Fitting method {name}")
            try:
                method = method_class(**params)
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


def fit_clustering_methods(
    df: pd.DataFrame,
    feature_name: str,
    methods: Dict[str, Tuple[Type[KMeans], dict]],
    save_dir: str,
    max_k: int = 100,
) -> dict[str, list[Any]]:
    """Fit clustering methods and determine the best number of clusters using the elbow method.

    Args:
        df (pd.DataFrame): DataFrame containing embeddings.
        feature_name (str): Feature name for saving/loading models.
        methods (dict): Dictionary of clustering methods and parameters.
        save_dir (str): Directory to save models.
        max_k (int): Maximum number of clusters to evaluate.

    Returns:
        Tuple[dict]: WCSS values.
    """
    # Extract embedding values as a NumPy array
    embeddings = df.values
    os.makedirs(save_dir, exist_ok=True)

    wcss_scores = {}

    for method_name, (method_class, params) in methods.items():
        method_wcss = []

        # Create a directory for the method to save all k's pickles
        method_dir = os.path.join(save_dir, f"{method_name}_{feature_name}")
        os.makedirs(method_dir, exist_ok=True)

        log.info(f"Evaluating clustering method {method_name} for the feature {feature_name}")
        for k in range(1, max_k + 1):
            model_path = os.path.join(method_dir, f"{method_name}_{feature_name}_k_{k}.pkl")

            # Try to load the model if it exists
            if os.path.exists(model_path):
                log.info(f"Loading pre-fitted model for {method_name} with k={k} from {model_path}")
                clustering_model = load_model_from_pickle(model_path)
                if clustering_model is None:
                    log.warning(f"Failed to load model for {method_name} with k={k}. Refitting...")
                else:
                    # If model is loaded, append WCSS and skip fitting
                    wcss = clustering_model.inertia_
                    method_wcss.append(wcss)
                    continue

            # If model does not exist or loading failed, fit and save
            try:
                params['n_clusters'] = k
                clustering_model = method_class(**params)
                clustering_model.fit(embeddings)

                # Compute WCSS (Inertia)
                wcss = clustering_model.inertia_
                method_wcss.append(wcss)

                # Save the model for the current k
                save_model_to_pickle(clustering_model, model_path)

            except Exception as e:
                log.error(f"Error with method {method_name} and k={k}: {e}")
                continue

        # Store WCSS scores for the method
        wcss_scores[method_name] = method_wcss

    return wcss_scores


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


def visualize_clustering_elbow(
    wcss_scores: dict,
    feature_name: str,
    save_path: str = None,
):
    """Visualize the WCSS and highlight the best k for each clustering method.

    Args:
        wcss_scores (dict): WCSS scores for each method.
        feature_name (str): Feature name for labeling the plot.
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    # Calculate the best k values
    best_k_values = calculate_best_k(wcss_scores)

    # Find the best method (smallest best k)
    best_method = min(best_k_values, key=best_k_values.get)
    best_k = best_k_values[best_method]

    plt.figure(figsize=(10, 6))

    # Plot all methods
    for name, wcss in wcss_scores.items():
        plt.plot(
            range(1, len(wcss) + 1),
            wcss,
            label=f"{name} (Best k={best_k_values[name]})",
        )

    # Highlight the best method with a vertical line
    plt.axvline(
        best_k,
        color='red',
        linestyle='--',
        label=f"Best Method: {best_method} (k={best_k})",
    )

    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.title(f"Clustering Elbow Method ({feature_name})")
    plt.legend()
    plt.grid()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, format='png', dpi=600)
        log.info(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()
