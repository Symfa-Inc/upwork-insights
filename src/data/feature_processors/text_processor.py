from typing import List, Optional, Type

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA

from src.data.feature_processors.base_processor import BaseProcessor

load_dotenv()
openai_client = OpenAI()


class TextProcessor(BaseProcessor):
    """A processor for handling and transforming text data into PCA of embeddings.

    Attributes:
        column_name (str): The name of the column to process.
        pca_threshold (float): Explained variance ratio threshold for PCA.
        pca (Optional[PCA]): PCA instance, fitted during the `fit` method.
    """

    scaling_method: Optional[Type[TransformerMixin]]
    pca_threshold: float
    pca: Optional[PCA]

    def __init__(
        self,
        column_name: str,
        pca_threshold: float = 0.85,
    ):
        """Initializes the TextProcessor with OpenAI client, PCA threshold, and optional scaler.

        Args:
            column_name (str): The name of the column to process.
            pca_threshold (float): Explained variance ratio threshold for PCA.

        Raises:
            ValueError: If `pca_threshold` is not between 0 and 1.
        """
        if not (0 < pca_threshold <= 1):
            raise ValueError('pca_threshold must be between 0 and 1.')

        self.pca_threshold = pca_threshold
        self.pca = None  # PCA instance will be initialized during `fit`

        # Call parent class initializer
        super().__init__(column_name)

    def _get_embeddings(
        self,
        texts: List[str],
        batch_size: int = 1000,
        model: str = 'text-embedding-3-large',
    ) -> np.ndarray:
        """Generates embeddings for a list of strings using OpenAI's API.

        Args:
            texts (List[str]): A list of strings to generate embeddings for.
            batch_size (int): The size of the batches for API requests. Defaults to 100.

        Returns:
            np.ndarray: A numpy array of embeddings.
        """
        cleaned_texts = [text if isinstance(text, str) else 'Missing' for text in texts]
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = cleaned_texts[i : i + batch_size]
            response = openai_client.embeddings.create(input=batch, model=model)
            embeddings.extend([res.embedding for res in response.data])

        return np.array(embeddings)

    def _fit(self, df: pd.DataFrame):
        """Fits the processor by generating embeddings, scaling them, applying PCA.

        Args:
            df (pd.DataFrame): The input DataFrame to fit on.

        Raises:
            ValueError: If the column is not found in the DataFrame.
        """
        texts = df[self.column_name].tolist()
        embeddings = self._get_embeddings(texts)

        # Initial PCA fit to compute explained variance
        self.pca = PCA()
        self.pca.fit(embeddings)
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)

        # Select the optimal number of components
        n_components = np.argmax(cumulative_variance >= self.pca_threshold) + 1

        # Refit PCA with the optimal number of components
        if n_components < embeddings.shape[1]:
            self.pca = PCA(n_components=n_components)
            self.pca.fit(embeddings)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data by generating embeddings, scaling them, and applying PCA.

        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame with reduced-dimensionality embeddings.

        Raises:
            ValueError: If the processor has not been fitted or the column is missing.
        """
        # Generate embeddings
        texts = df[self.column_name].tolist()
        embeddings = self._get_embeddings(texts)

        # Apply PCA
        embeddings = self.pca.transform(embeddings)

        # Create a DataFrame for principal components
        pc_columns = {
            f"{self.column_name}_PC{i + 1}": embeddings[:, i] for i in range(embeddings.shape[1])
        }
        pc_df = pd.DataFrame(pc_columns, index=df.index)

        # Drop the original column and concatenate new principal components
        df = df.drop(columns=[self.column_name])
        df = pd.concat([df, pc_df], axis=1)
        return df

    def get_params(self) -> dict:
        return {
            'PCA': {
                'n_params': int(self.pca.n_components_),
                'explained_variance': self.pca.explained_variance_ratio_.tolist(),
                'components': self.pca.components_,
            },
        }


if __name__ == '__main__':
    data = pd.DataFrame(
        {
            '_text_column': [
                'This is the first test sentence.',
                'Here is another example for testing.',
                'Machine learning with embeddings is powerful.',
                'OpenAI API provides useful tools.',
                'Testing the PCA transformation process.',
            ],
            'text_column': [
                'This is the first test sentence.',
                'Here is another example for testing.',
                'Machine learning with embeddings is powerful.',
                'OpenAI API provides useful tools.',
                'Testing the PCA transformation process.',
            ],
        },
    )

    processor = TextProcessor(
        column_name='text_column',
        pca_threshold=0.85,
    )

    # Transform the data
    transformed_data = processor.process(data)

    # Output transformed data
    print('Transformed Data:')
    print(transformed_data)

    # Output processor parameters
    print('\nProcessor Parameters:')
    print(processor.get_params())
