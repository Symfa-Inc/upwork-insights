from typing import List, Optional, Set

import pandas as pd

from src.data.feature_processors.base_processor import BaseProcessor
from src.data.utils import normalize_to_snake_name


class OneHotProcessor(BaseProcessor):
    """A processor for one-hot encoding a categorical column with an optional cumulative coverage threshold.

    Attributes:
        column_name (str): The name of the column to process.
        threshold (Optional[float]): Cumulative coverage threshold for including categories.
            If not provided, all categories are included.
        selected_categories (Optional[List[str]]): List of categories to include in one-hot encoding,
            determined during fitting.
    """

    def __init__(self, column_name: str, threshold: Optional[float] = None):
        """Initializes the CategoryOneHotProcessor with a column name and optional threshold.

        Args:
            column_name (str): The name of the column to process.
            threshold (Optional[float]): Cumulative coverage threshold for including categories.
                If None, all categories are included.
        """
        super().__init__(column_name)
        self.threshold = threshold
        self.selected_categories: Optional[Set[str]] = None
        self.cumulative_proportions: Optional[List[float]] = None

    def _fit(self, df: pd.DataFrame):
        """Fits the processor by determining the categories to include, based on the cumulative coverage of the dataset.

        Args:
            df (pd.DataFrame): The input DataFrame to fit on.

        Raises:
            ValueError: If the column is not found in the DataFrame.
            ValueError: If the threshold is not between 0 and 1.
        """
        # Calculate category frequencies
        value_counts = df[self.column_name].value_counts(normalize=True)

        # Sort categories by frequency
        sorted_categories = value_counts.items()
        cumulative_coverage = 0
        selected_categories = []
        self.cumulative_proportions = []

        for category, proportion in sorted_categories:
            if self.threshold is not None:
                cumulative_coverage += proportion
                if cumulative_coverage >= self.threshold:
                    break
                self.cumulative_proportions.append(proportion)
            selected_categories.append(category)

        # Save the selected categories
        self.selected_categories = set(
            selected_categories if self.threshold is not None else set(value_counts.keys()),
        )

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data by one-hot encoding the categorical column.

        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame with one-hot encoded features.
        """
        # One-hot encode the selected categories
        one_hot_columns = {}
        for category in self.selected_categories:
            one_hot_columns[f"{self.column_name}_{normalize_to_snake_name(category)}"] = (
                df[self.column_name] == category
            ).astype(int)

        # Handle "others" category if threshold is provided
        if self.threshold is not None:
            one_hot_columns[f"{self.column_name}_others"] = (
                ~df[self.column_name].isin(self.selected_categories)
            ).astype(int)

        # Combine all one-hot columns into a new DataFrame
        encoded_df = pd.DataFrame(one_hot_columns, index=df.index)

        # Concatenate the original DataFrame (excluding the processed column) with the new one-hot columns
        df = pd.concat([df.drop(columns=[self.column_name]), encoded_df], axis=1)

        return df

    def get_params(self) -> dict:
        """Retrieves the parameters of the processor.

        Returns:
            dict: A dictionary containing the column name, threshold, and selected categories.
        """
        return {
            'column_name': self.column_name,
            'threshold': self.threshold,
            'selected_categories': self.selected_categories,
            'cumulative_proportions': self.cumulative_proportions,
        }


if __name__ == '__main__':
    data = pd.DataFrame(
        {
            'category': ['A', 'B', 'A', 'C', 'A', 'B', 'C', 'C', 'C', 'A', 'D', 'E', 'F', 'A'],
        },
    )

    # Initialize processor with a threshold
    processor = OneHotProcessor(column_name='category', threshold=0.7)

    # Fit the processor
    processor.fit(data)

    # Transform the data
    transformed_data = processor.transform(data)

    print('Transformed DataFrame:')
    print(transformed_data)

    print('\nProcessor Parameters:')
    print(processor.get_params())
