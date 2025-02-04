from typing import List, Optional

import numpy as np
import pandas as pd

from src.data.feature_processors.text_processor import TextProcessor


class TextProcessorStatistics(TextProcessor):
    """A processor for calculating statistical parameters of text embeddings."""

    def __init__(
        self,
        column_name: str,
        model: str,
        embeddings_dir: str,
        statistics: Optional[List[str]] = None,
    ):
        """Initializes the TextProcessorStatistics with statistical parameters.

        Args:
            column_name (str): The name of the column to process.
            model (str): The embedding model to use.
            embeddings_dir (str): Directory where embeddings should be stored.
            statistics (List[str]): List of statistical metrics to compute. Defaults to ['mean', 'std', 'min', 'max'].
        """
        super().__init__(column_name, model, embeddings_dir)
        self.statistics = statistics or ['mean', 'std', 'min', 'max']

    def _fit(self, df: pd.DataFrame):
        """No fitting required for statistical computation."""
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data by computing statistical metrics of embeddings.

        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame with statistical metrics as features.
        """
        texts = df[self.column_name]
        embeddings = self._load_or_compute_embeddings(texts).values

        stats_dict = {}
        if 'mean' in self.statistics:
            stats_dict[f"{self.column_name}_mean"] = np.mean(embeddings, axis=1)
        if 'std' in self.statistics:
            stats_dict[f"{self.column_name}_std"] = np.std(embeddings, axis=1)
        if 'min' in self.statistics:
            stats_dict[f"{self.column_name}_min"] = np.min(embeddings, axis=1)
        if 'max' in self.statistics:
            stats_dict[f"{self.column_name}_max"] = np.max(embeddings, axis=1)

        stats_df = pd.DataFrame(stats_dict, index=df.index)
        return df.drop(columns=[self.column_name]).join(stats_df)

    def get_params(self) -> dict:
        return {
            'Statistics': {
                'computed_metrics': self.statistics,
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

    processor = TextProcessorStatistics(
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
