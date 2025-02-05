from typing import Dict, Optional, Type, Union

import pandas as pd
from sklearn.cluster import KMeans

from src.data.feature_processors.cosine_k_means import CosineKMeans
from src.data.feature_processors.text_processor import TextProcessor


class TextProcessorClustering(TextProcessor):
    """A processor for handling and transforming text data into cluster labels using embeddings."""

    def __init__(
        self,
        column_name: str,
        model: str,
        embeddings_dir: str,
        clustering_class: Union[Type[KMeans], Type[CosineKMeans]] = KMeans,
        clustering_params: Optional[Dict] = None,
        cluster_column_name: Optional[str] = None,
        drop_column: bool = True,
    ):
        """Initializes the TextProcessorClustering with clustering parameters.

        Args:
            column_name (str): The name of the column to process.
            model (str): The embedding model to use.
            embeddings_dir (str): Directory where embeddings should be stored.
            clustering_class (Type): The clustering algorithm class to use.
            clustering_params (Dict): Parameters for the clustering algorithm.
            cluster_column_name (str): Custom name for the cluster feature column.
            drop_column (bool): Defines behaviour of the original column after transform. Used for combine processors.
        """
        super().__init__(column_name, model, embeddings_dir)
        self.clustering_class = clustering_class
        self.clustering_params = clustering_params or {}
        self.cluster_column_name = cluster_column_name or f"{column_name}_Cluster"
        self.drop_column = drop_column

    def _fit(self, df: pd.DataFrame):
        """Fits the clustering model on the text embeddings.

        Args:
            df (pd.DataFrame): The input DataFrame to fit on.
        """
        texts = df[self.column_name]
        embeddings = self._load_or_compute_embeddings(texts).values

        # Initialize and fit clustering model
        self.clustering_model = self.clustering_class(**self.clustering_params)
        self.clustering_model.fit(embeddings)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data by assigning cluster labels to the text data.

        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame with a new cluster label feature.
        """
        texts = df[self.column_name]
        embeddings = self._load_or_compute_embeddings(texts).values

        # Predict cluster labels
        cluster_labels = self.clustering_model.fit_predict(embeddings)

        # Create DataFrame with cluster assignments
        cluster_df = pd.DataFrame({self.cluster_column_name: cluster_labels}, index=df.index)

        if self.drop_column:
            df = df.drop(columns=[self.column_name])

        return df.join(cluster_df)

    def get_params(self) -> dict:
        return {
            'Clustering': {
                'algorithm': self.clustering_class.__name__,
                'params': self.clustering_params,
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

    processor = TextProcessorClustering(
        column_name='text_column',
        model='thenlper/gte-small',
        embeddings_dir='./data/embeddings',
        clustering_params={'n_clusters': 5},
    )

    # Transform the data
    transformed_data = processor.process(data)

    # Output transformed data
    print('Transformed Data:')
    print(transformed_data)

    # Output processor parameters
    print('\nProcessor Parameters:')
    print(processor.get_params())
