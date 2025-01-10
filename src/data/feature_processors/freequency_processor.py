from typing import Dict, Optional

import pandas as pd

from src.data.feature_processors.base_processor import BaseProcessor


class FrequencyProcessor(BaseProcessor):
    """A processor for encoding a categorical feature using relative frequencies.

    Attributes:
        column_name (str): The name of the column to process.
        frequency_mapping (Optional[dict]): A mapping from original values to their relative frequencies,
            determined during fitting.
    """

    def __init__(self, column_name: str):
        """Initializes the FrequencyProcessor with a column name.

        Args:
            column_name (str): The name of the column to process.
        """
        super().__init__(column_name)
        self.frequency_mapping: Optional[Dict[str, float]] = None

    def _fit(self, df: pd.DataFrame):
        """Fits the processor by calculating the relative frequency for each unique value in the column.

        Args:
            df (pd.DataFrame): The input DataFrame to fit on.

        Raises:
            ValueError: If the column is not found in the DataFrame.
        """
        # Calculate relative frequencies
        value_counts = df[self.column_name].value_counts(normalize=True, dropna=False)
        self.frequency_mapping = value_counts.to_dict()

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data by encoding the column using relative frequencies.

        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame with the frequency-encoded column.
        """
        # Map relative frequencies to the column
        df[self.column_name] = df[self.column_name].map(self.frequency_mapping).fillna(0)
        # README. I kept fillna(0) for cases where we encountered a new class in the transformation step.
        # This is not needed for data preparation, but for possible model inference.
        # Our other processors work reliably with out-of-sample classes, so this should work as well.
        # For now it will map na -> df.isna().sum()/len(df)
        # out of sample new class -> 0

        return df

    def get_params(self) -> dict:
        """Retrieves the frequency mapping of the processor.

        Returns:
            dict: A dictionary containing the mapping from original values to their relative frequencies.
        """
        return {
            'column_name': self.column_name,
            'frequency_mapping': self.frequency_mapping,
        }


if __name__ == '__main__':
    # Sample DataFrame
    data = pd.DataFrame(
        {
            '_category': ['A', 'B', 'A', 'C', 'A', 'B', 'C', 'C', 'C', 'A', 'D', 'A'],
            'category': ['A', 'B', 'A', 'C', 'A', 'B', 'C', 'C', 'C', 'A', 'D', 'A'],
        },
    )

    # Initialize and fit the processor
    processor = FrequencyProcessor(column_name='category')
    processor.fit(data)

    # Transform the data
    transformed_data = processor.transform(data)

    print('Transformed DataFrame:')
    print(transformed_data)

    print('\nProcessor Parameters:')
    print(processor.get_params())
