from typing import Dict, Type, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA

from src.data.feature_processors.ppa import PCAWithPreProcessing
from src.data.feature_processors.text_processor import TextProcessor


class TextProcessorPCA(TextProcessor):
    """A processor for handling and transforming text data into PCA of embeddings.

    Attributes:
        pca_class (Type): The PCA class to be used (e.g., PCA, KernelPCA, custom PPA).
        pca_params (Dict): Dictionary of parameters to initialize the PCA class.
        pca_threshold (float): Explained variance ratio threshold for PCA.
        min_components (int): Minimum number of principal components.
        max_components (int): Maximum number of principal components.
        pca (Optional[PCA, KernelPCA, PCAWithPreProcessing]): PCA instance, fitted during the `fit` method.
    """

    def __init__(
        self,
        column_name: str,
        model: str,
        embeddings_dir: str,
        pca_class: Union[Type[PCA], Type[KernelPCA], Type[PCAWithPreProcessing]],
        pca_params: Dict = {},
        pca_threshold: float = 0.85,
        min_components: int = 5,
        max_components: int = 25,
    ):
        """Initializes the TextProcessorPCA with PCA parameters.

        Args:
            column_name (str): The name of the column to process.
            model (str): The embedding model to use.
            embeddings_dir (str): Directory where embeddings should be stored.
            pca_class (Type): The PCA class to use (default: PCA).
            pca_params (Dict): Parameters for the PCA class.
            pca_threshold (float): Explained variance ratio threshold for PCA.
            min_components (int): Minimum number of principal components.
            max_components (int): Maximum number of principal components.
        """
        super().__init__(column_name, model, embeddings_dir)

        if not (0 < pca_threshold <= 1):
            raise ValueError('pca_threshold must be between 0 and 1.')
        if min_components < 1 or max_components < min_components:
            raise ValueError(
                'Invalid component range. Ensure 1 <= min_components <= max_components.',
            )

        self.pca_threshold = pca_threshold
        self.min_components = min_components
        self.max_components = max_components
        self.pca_class = pca_class
        self.pca_params = pca_params

    def _fit(self, df: pd.DataFrame):
        """Fits the processor by generating embeddings and applying PCA.

        Args:
            df (pd.DataFrame): The input DataFrame to fit on.
        """
        texts = df[self.column_name]
        embeddings = self._load_or_compute_embeddings(texts).values

        # Initial PCA fit to compute explained variance
        self.pca = self.pca_class(n_components=self.max_components, **self.pca_params)
        self.pca.fit(embeddings)

        if isinstance(self.pca, PCAWithPreProcessing):
            cumulative_variance = np.cumsum(self.pca.pca_.explained_variance_ratio_)
        else:
            cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)

        # Determine optimal components
        threshold = self.pca_threshold
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        n_components = max(self.min_components, min(n_components, self.max_components))

        # Refit PCA with the optimal number of components
        self.pca = self.pca_class(n_components=n_components, **self.pca_params)
        self.pca.fit(embeddings)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data by generating embeddings and applying PCA.

        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame with reduced-dimensionality embeddings.
        """
        texts = df[self.column_name]
        embeddings = self._load_or_compute_embeddings(texts).values

        # Apply PCA
        embeddings_pca = self.pca.transform(embeddings)

        # Create DataFrame with principal components
        pc_columns = {
            f"{self.column_name}_PC{i+1}": embeddings_pca[:, i]
            for i in range(embeddings_pca.shape[1])
        }
        pc_df = pd.DataFrame(pc_columns, index=df.index)

        return df.drop(columns=[self.column_name]).join(pc_df)

    def get_params(self) -> dict:
        return {
            'PCA': {
                'n_params': int(
                    (
                        self.pca.pca_.n_components_
                        if isinstance(self.pca, PCAWithPreProcessing)
                        else self.pca.n_components_
                    ),
                ),
                'explained_variance': (
                    self.pca.pca_.explained_variance_ratio_.tolist()
                    if isinstance(self.pca, PCAWithPreProcessing)
                    else self.pca.explained_variance_ratio_.tolist()
                ),
                'components': (
                    self.pca.pca_.components_
                    if isinstance(self.pca, PCAWithPreProcessing)
                    else self.pca.components_
                ),
            },
            'config': {
                'pca_class': self.pca_class.__name__,
                'pca_params': self.pca_params,
                'pca_threshold': self.pca_threshold,
                'min_components': self.min_components,
                'max_components': self.max_components,
            },
        }
