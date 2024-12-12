from collections import Counter
from typing import List, Optional, Set

import pandas as pd

from src.data.feature_processors.base_processor import BaseProcessor
from src.data.utils import normalize_to_snake_name


class ListProcessor(BaseProcessor):
    """A processor for handling and transforming list features (e.g., skills) into one-hot encoded features.

    Attributes:
        column_name (str): The name of the column to process.
        threshold (Optional[float]): Cumulative coverage threshold for including skills in one-hot encoding.
            Skills with frequency below this threshold are grouped into the "others" category.
        unique_values (Optional[List[str]]): List of unique values to include in one-hot encoding,
            determined during fitting.
    """

    def __init__(self, column_name: str, threshold: float = 0.8):
        """Initializes the ListFeatureProcessor with a column name and optional frequency threshold.

        Args:
            column_name (str): The name of the column to process.
            threshold (Optional[float]): Cumulative coverage threshold for including skills in one-hot encoding.
                Skills with frequency below this threshold are grouped into the "others" category.
        """
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError('threshold must be a float between 0 and 1, inclusive.')
        self.threshold = threshold
        self.unique_values: Optional[Set[str]] = None
        self.cumulative_ratio: Optional[List[float]] = None
        super().__init__(column_name)

    def _fit(self, df: pd.DataFrame):
        """Fits the processor by determining the unique values to include in encoding, based on the frequency.

        Args:
            df (pd.DataFrame): The input DataFrame to fit on.

        Raises:
            ValueError: If the column is not found in the DataFrame.
        """
        # Flatten the list of lists in the specified column
        all_values = [item for sublist in df[self.column_name].dropna() for item in sublist]
        value_counts = Counter(all_values)

        # Convert to a DataFrame for processing
        df_value = pd.DataFrame(value_counts.items(), columns=['value', 'frequency'])
        df_value = df_value.sort_values(by='frequency', ascending=False).reset_index(drop=True)

        # Calculate cumulative ratio
        total_frequency = df_value['frequency'].sum()
        df_value['ratio'] = df_value['frequency'] / total_frequency
        df_value['cumulative_ratio'] = df_value['ratio'].cumsum()

        # Select skills based on threshold
        threshold_index = df_value[df_value['cumulative_ratio'] >= self.threshold].index[0]
        selected_values = df_value.loc[:threshold_index, 'value']
        self.unique_values = set(selected_values)
        self.ratio = df_value.loc[:threshold_index, 'ratio'].tolist()
        self.cumulative_ratio = df_value.loc[:threshold_index, 'cumulative_ratio'].tolist()

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data by one-hot encoding the list feature and adding an "others" category if applicable.

        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame with one-hot encoded features.
        """
        # Dictionary to store all new columns
        one_hot_columns = {}

        # Create binary columns for each unique value
        for value in self.unique_values:
            one_hot_columns[f"{self.column_name}_{normalize_to_snake_name(value)}"] = df[
                self.column_name
            ].apply(
                lambda x: int(value in x) if isinstance(x, list) else 0,
            )

        # Handle "others" category as count if min_frequency is provided
        if self.threshold:
            one_hot_columns[f"{self.column_name}_others"] = df[self.column_name].apply(
                lambda x: (
                    sum(1 for item in x if item not in self.unique_values)
                    if isinstance(x, list)
                    else 0
                ),
            )

        # Combine all one-hot columns into a new DataFrame
        encoded_df = pd.DataFrame(one_hot_columns, index=df.index)

        return encoded_df

    def get_params(self) -> dict:
        """Retrieves the parameters of the processor.

        Returns:
            dict: A dictionary containing the column name, min frequency, and unique values.
        """
        return {
            'column_name': self.column_name,
            'threshold': self.threshold,
            'unique_values': self.unique_values,
            'ratio': self.ratio,
            'cumulative_ratio': self.cumulative_ratio,
        }


if __name__ == '__main__':
    data = pd.DataFrame(
        {
            'skills': [
                ['Python', 'SQL', 'Pandas'],
                ['Python', 'Java', 'C++'],
                ['Python', 'R', 'Statistics'],
                ['SQL', 'Pandas'],
                ['Java', 'C++'],
                ['Python'],
            ],
        },
    )

    processor = ListProcessor(column_name='skills', threshold=0.8)
    processor.fit(data)
    transformed_data = processor.transform(data)
    print(transformed_data)
    print('\nProcessor Parameters:')
    print(processor.get_params())
