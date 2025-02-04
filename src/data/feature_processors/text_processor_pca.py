from typing import Dict, Optional, Type, Union

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
        pca_class: Union[Type[PCA], Type[KernelPCA], Type[PCAWithPreProcessing]] = PCA,
        pca_params: Optional[Dict] = None,
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
        self.pca_params = pca_params if pca_params else {}

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


if __name__ == '__main__':
    data = pd.DataFrame(
        {
            '_text_column': [
                'This is the first test sentence.',
                'Here is another example for testing.',
                'Machine learning with embeddings is powerful.',
                'OpenAI API provides useful tools.',
                'Testing the PCA transformation process.',
                'Deep learning models are transforming AI applications.',
                'Natural language processing enables human-like text understanding.',
                'Embedding representations improve text classification accuracy.',
                'Sentence transformers create dense vector representations.',
                'Clustering text documents requires effective embeddings.',
                'Neural networks learn complex patterns in data.',
                'Transfer learning accelerates model training and fine-tuning.',
                'Feature extraction helps in dimensionality reduction.',
                'Word2Vec and BERT generate meaningful word embeddings.',
                'Pre-trained models save computation time in NLP tasks.',
                'Open-source AI libraries foster rapid development.',
                'PyTorch and TensorFlow are popular deep learning frameworks.',
                'Fine-tuning models enhances performance on specific tasks.',
                'The cosine similarity metric compares embedding distances.',
                'BERT embeddings capture contextual word meanings.',
                'Dimensionality reduction techniques like PCA improve efficiency.',
                'Word embeddings represent words in a dense space.',
                'Tokenization is a crucial step in text preprocessing.',
                'Sentence similarity is computed using vector distances.',
                'TF-IDF is a traditional method for text representation.',
                'Deep neural networks require large amounts of training data.',
                'Semantic search leverages embeddings for relevance ranking.',
                'Vector space models enable document comparison.',
                'The softmax function normalizes probability distributions.',
                'Attention mechanisms improve transformer model performance.',
                'Sequence-to-sequence models are used in machine translation.',
                'Text summarization extracts key information from documents.',
                'Sentiment analysis detects emotions in text.',
                'Reinforcement learning optimizes decision-making strategies.',
                'Named entity recognition identifies proper nouns in text.',
                'Part-of-speech tagging classifies words into grammatical categories.',
                'Topic modeling groups documents based on similar themes.',
                'Word frequency analysis helps in keyword extraction.',
                'Neural networks generalize better with diverse training data.',
                'Knowledge graphs represent relationships between entities.',
                'Text embeddings power recommendation systems.',
                'Encoder-decoder architectures enable generative NLP models.',
                'GPT models generate human-like text responses.',
                'Data augmentation improves model robustness.',
                'Hyperparameter tuning optimizes model performance.',
                'Text normalization reduces vocabulary size.',
                'Embedding alignment helps in cross-lingual NLP.',
                'Named entities can be linked to knowledge bases.',
                'Bag-of-words is a simple text representation method.',
                'Multilingual embeddings improve cross-language applications.',
                'Semantic similarity measures how close texts are in meaning.',
                'Siamese networks are useful for text pair comparisons.',
                'Hierarchical clustering groups text documents by similarity.',
                'Contextualized embeddings capture sentence-level meaning.',
                'Machine learning pipelines automate NLP workflows.',
                'AI-powered chatbots understand and respond to user queries.',
            ],
            'text_column': [
                'This is the first test sentence.',
                'Here is another example for testing.',
                'Machine learning with embeddings is powerful.',
                'OpenAI API provides useful tools.',
                'Testing the PCA transformation process.',
                'Deep learning models are transforming AI applications.',
                'Natural language processing enables human-like text understanding.',
                'Embedding representations improve text classification accuracy.',
                'Sentence transformers create dense vector representations.',
                'Clustering text documents requires effective embeddings.',
                'Neural networks learn complex patterns in data.',
                'Transfer learning accelerates model training and fine-tuning.',
                'Feature extraction helps in dimensionality reduction.',
                'Word2Vec and BERT generate meaningful word embeddings.',
                'Pre-trained models save computation time in NLP tasks.',
                'Open-source AI libraries foster rapid development.',
                'PyTorch and TensorFlow are popular deep learning frameworks.',
                'Fine-tuning models enhances performance on specific tasks.',
                'The cosine similarity metric compares embedding distances.',
                'BERT embeddings capture contextual word meanings.',
                'Dimensionality reduction techniques like PCA improve efficiency.',
                'Word embeddings represent words in a dense space.',
                'Tokenization is a crucial step in text preprocessing.',
                'Sentence similarity is computed using vector distances.',
                'TF-IDF is a traditional method for text representation.',
                'Deep neural networks require large amounts of training data.',
                'Semantic search leverages embeddings for relevance ranking.',
                'Vector space models enable document comparison.',
                'The softmax function normalizes probability distributions.',
                'Attention mechanisms improve transformer model performance.',
                'Sequence-to-sequence models are used in machine translation.',
                'Text summarization extracts key information from documents.',
                'Sentiment analysis detects emotions in text.',
                'Reinforcement learning optimizes decision-making strategies.',
                'Named entity recognition identifies proper nouns in text.',
                'Part-of-speech tagging classifies words into grammatical categories.',
                'Topic modeling groups documents based on similar themes.',
                'Word frequency analysis helps in keyword extraction.',
                'Neural networks generalize better with diverse training data.',
                'Knowledge graphs represent relationships between entities.',
                'Text embeddings power recommendation systems.',
                'Encoder-decoder architectures enable generative NLP models.',
                'GPT models generate human-like text responses.',
                'Data augmentation improves model robustness.',
                'Hyperparameter tuning optimizes model performance.',
                'Text normalization reduces vocabulary size.',
                'Embedding alignment helps in cross-lingual NLP.',
                'Named entities can be linked to knowledge bases.',
                'Bag-of-words is a simple text representation method.',
                'Multilingual embeddings improve cross-language applications.',
                'Semantic similarity measures how close texts are in meaning.',
                'Siamese networks are useful for text pair comparisons.',
                'Hierarchical clustering groups text documents by similarity.',
                'Contextualized embeddings capture sentence-level meaning.',
                'Machine learning pipelines automate NLP workflows.',
                'AI-powered chatbots understand and respond to user queries.',
            ],
        },
    )

    processor = TextProcessorPCA(
        column_name='text_column',
        model='thenlper/gte-small',
        embeddings_dir='./data/embeddings',
    )

    # Transform the data
    transformed_data = processor.process(data)

    # Output transformed data
    print('Transformed Data:')
    print(transformed_data)

    # Output processor parameters
    print('\nProcessor Parameters:')
    print(processor.get_params())
