from typing import Optional

import pandas as pd

from src.data.feature_processors.base_processor import BaseProcessor


class BooleanProcessor(BaseProcessor):
    """A processor for handling and processing a column of boolean type data.

    This class provides functionality to clean, fill missing values,
    and ensure consistency for a specified boolean column.

    Manual checks have shown that for all boolean columns in the dataset,
    there are only 5 or 6 missing (None) values in total. Therefore,
    the most convenient and efficient way to handle them is by replacing
    missing values with the most frequent value (True or False).

    Inherits from:
        BaseProcessor
    """

    most_common: Optional[bool]

    def __init__(self, column_name: str):
        """Initializes the BooleanProcessor with the column name.

        Args:
            column_name (str): The name of the column to be processed.
        """
        self.most_common = None
        super().__init__(column_name)  # Initialize the base class

    def _fit(self, df: pd.DataFrame):
        """Fits the processor to the data by determining the most frequent value in the specified column.

        Args:
            df (pd.DataFrame): The input DataFrame to fit on.
        """
        self.most_common = df[self.column_name].mean() >= 0.5

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data by replacing missing values with the most frequent value and converting the column to int type.

        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.

        Raises:
            ValueError: If `fit` has not been called and `most_common` is not set.
        """
        df[self.column_name] = (
            df[self.column_name].infer_objects(copy=False).fillna(self.most_common)
        )
        df[self.column_name] = df[self.column_name].astype(int)
        return df

    def get_params(self) -> dict:
        return {'most_common': self.most_common}


if __name__ == '__main__':
    # Example usage
    import numpy as np

    data = pd.DataFrame(
        {
            '_BooleanColumn': [True, np.nan, False, True, None, True, None, np.nan, False, True],
            'BooleanColumn': [True, np.nan, False, True, None, True, None, np.nan, False, True],
        },
    )
    processor = BooleanProcessor('BooleanColumn')
    processed_data = processor.process(df=data)
    print(processed_data)
