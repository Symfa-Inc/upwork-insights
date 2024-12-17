from typing import Any, Dict, Optional

import pandas as pd

from src.data.feature_processors.base_processor import BaseProcessor


class OrdinalProcessor(BaseProcessor):
    """A processor for handling and transforming ordinal categorical columns.

    This class provides functionality to map ordinal categorical values
    to integers based on a user-defined or automatically generated mapping.

    Attributes:
        column_name (str): The name of the column to process.
        mapping (Optional[Dict[Any, int]]): A mapping of ordinal values to integers.
    """

    mapping: Optional[Dict[Any, int]]

    def __init__(self, column_name: str, mapping: Optional[Dict[Any, int]] = None):
        """Initializes the OrdinalProcessor with a column name and optional mapping.

        Args:
            column_name (str): The name of the column to process.
            mapping (Optional[Dict]): Optional mapping of ordinal values to integers.
        """
        self.mapping = mapping
        super().__init__(column_name)

    def _fit(self, df: pd.DataFrame):
        """Fits the processor by generating a mapping of ordinal values to integers if a mapping is not provided.

        Args:
            df (pd.DataFrame): The input DataFrame to fit on.

        Raises:
            ValueError: If the column is not found in the DataFrame.
        """
        if self.mapping is None:
            unique_values = df[self.column_name].dropna().unique()  # Exclude missing values
            self.mapping = {value: i for i, value in enumerate(unique_values)}
        self.mode_value = df[self.column_name].mode().iloc[0]

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data by mapping the ordinal values to integers using the defined or automatic mapping.

        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.

        Raises:
            ValueError: If the processor has not been fitted or the column is missing.
        """
        # Map values and fill missing values with the mode
        df[self.column_name] = (
            df[self.column_name]
            .fillna(self.mode_value)  # Fill missing with mode
            .map(self.mapping)  # Map ordinal values
            .astype(int)
        )

        return df

    def get_params(self) -> dict:
        return {'mapping': self.mapping}


if __name__ == '__main__':
    # Sample DataFrame
    data = pd.DataFrame(
        {
            'ordinal_col': ['low', 'medium', 'high', 'low', 'high', None],
        },
    )

    # Initialize processor
    processor = OrdinalProcessor(column_name='ordinal_col')

    # Fit and transform
    processed_data = processor.process(data)

    print(processed_data)
