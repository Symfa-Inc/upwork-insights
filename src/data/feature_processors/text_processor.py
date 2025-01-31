import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd

from src.data.feature_processors.base_processor import BaseProcessor
from src.data.utils import get_embeddings, get_embeddings_gte


class TextProcessor(BaseProcessor, ABC):
    """Abstract base class for text processing with different transformation techniques."""

    column_name: str

    def __init__(
        self,
        column_name: str,
        model: str = 'thenlper/gte-small',
        embeddings_dir: str = './embeddings',
    ):
        """Initializes the TextProcessor with a model for generating embeddings.

        Args:
            column_name (str): The name of the column to process.
            model (str): The embedding model to use. Defaults to 'thenlper/gte-small'.
            embeddings_dir (str): Directory where embeddings should be stored. Defaults to './embeddings'.
        """
        valid_models = {
            'text-embedding-3-small',
            'text-embedding-3-large',
            'thenlper/gte-small',
            'thenlper/gte-large',
        }
        if model not in valid_models:
            raise ValueError(f"Invalid model: {model}. Must be one of {valid_models}")

        self.model = model
        self.model_name = model.split('/')[-1]
        self.embeddings_dir = embeddings_dir
        super().__init__(column_name)

    @abstractmethod
    def _fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def _get_embeddings(
        self,
        texts: List[str],
        batch_size: int = 1000,
        model: str = 'text-embedding-3-large',
    ) -> np.ndarray:
        """Generates embeddings for a list of strings using the specified model.

        Args:
            texts (List[str]): A list of strings to generate embeddings for.
            batch_size (int): The size of the batches for API requests. Defaults to 1000.
            model (str): The embedding model to use. Defaults to 'text-embedding-3-large'.

        Returns:
            np.ndarray: A numpy array containing the generated embeddings.
        """
        if self.model in {'text-embedding-3-small', 'text-embedding-3-large'}:
            return get_embeddings(texts, batch_size, model=model)
        elif self.model in {'thenlper/gte-small', 'thenlper/gte-large'}:
            return get_embeddings_gte(texts, batch_size, model=model)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _load_or_compute_embeddings(self, texts: pd.Series) -> pd.DataFrame:
        """Loads embeddings from a file if available, otherwise computes and saves missing ones.

        Args:
            texts (pd.Series): A series containing text data.

        Returns:
            pd.DataFrame: A DataFrame with the same index as `texts`, and embedding features as columns.
        """
        file_path = os.path.join(
            self.embeddings_dir,
            f"{self.column_name}_{self.model_name}.parquet",
        )
        if os.path.exists(file_path):
            existing_embeddings = pd.read_parquet(file_path)
        else:
            existing_embeddings = pd.DataFrame()

        missing_indexes = texts.index.difference(existing_embeddings.index)
        if not missing_indexes.empty:
            new_texts = texts.loc[missing_indexes].tolist()
            new_embeddings = self._get_embeddings(new_texts)
            new_embedding_df = pd.DataFrame(
                new_embeddings,
                index=missing_indexes,
                columns=[f"Embedding_{i+1}" for i in range(new_embeddings.shape[1])],
            )
            existing_embeddings = pd.concat([existing_embeddings, new_embedding_df])
            existing_embeddings.to_parquet(file_path)

        return existing_embeddings.loc[texts.index]

    @abstractmethod
    def get_params(self) -> dict:
        pass
