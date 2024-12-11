from collections import Counter
from typing import List, Optional

import pandas as pd

from src.data.feature_processors.base_processor import BaseProcessor
from src.data.utils import normalize_skill_name


class ListFeatureProcessor(BaseProcessor):
    """A processor for handling and transforming list features (e.g., skills) into one-hot encoded features.

    Attributes:
        column_name (str): The name of the column to process.
        min_frequency (Optional[int]): Minimum frequency for including a skill as a separate feature.
            Skills with frequency below this threshold are grouped into the "others" category.
        unique_values (Optional[List[str]]): List of unique values to include in one-hot encoding,
            determined during fitting.
    """

    def __init__(self, column_name: str, min_frequency: Optional[int] = None):
        """Initializes the ListFeatureProcessor with a column name and optional frequency threshold.

        Args:
            column_name (str): The name of the column to process.
            min_frequency (Optional[int]): Minimum frequency for including a skill as a separate feature.
                Skills with frequency below this threshold are grouped into the "others" category.
        """
        self.min_frequency = min_frequency
        self.unique_values: Optional[List[str]] = None
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

        # Filter unique values based on the min_frequency threshold
        if self.min_frequency:
            self.unique_values = [
                value for value, count in value_counts.items() if count >= self.min_frequency
            ]
        else:
            self.unique_values = list(value_counts.keys())

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data by one-hot encoding the list feature and adding an "others" category if applicable.

        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame with one-hot encoded features.
        """
        # Initialize one-hot encoded DataFrame
        encoded_df = pd.DataFrame(index=df.index)

        for value in self.unique_values:
            # Create a binary column for each unique value
            encoded_df[f"{self.column_name}_{normalize_skill_name(value)}"] = df[
                self.column_name
            ].apply(
                lambda x: int(value in x) if x else 0,
            )

        # Handle "others" category if min_frequency is provided
        if self.min_frequency:
            encoded_df[f"{self.column_name}_others"] = df[self.column_name].apply(
                lambda x: (
                    sum(1 for item in x if item not in self.unique_values)
                    if isinstance(x, list)
                    else 0
                ),
            )

        return encoded_df

    def get_params(self) -> dict:
        """Retrieves the parameters of the processor.

        Returns:
            dict: A dictionary containing the column name, min frequency, and unique values.
        """
        return {
            'column_name': self.column_name,
            'min_frequency': self.min_frequency,
            'unique_values': self.unique_values,
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

    processor = ListFeatureProcessor(column_name='skills')
    processor.fit(data)
    transformed_data = processor.transform(data)

    print(transformed_data)

    processor = ListFeatureProcessor(column_name='skills', min_frequency=3)
    processor.fit(data)
    transformed_data = processor.transform(data)
    print(transformed_data)

    print('\nProcessor Parameters:')
    print(processor.get_params())
