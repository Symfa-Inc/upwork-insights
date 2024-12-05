import pandas as pd

from src.data.feature_processors.base_feature_processor import BaseFeatureProcessor


class BooleanFeatureProcessor(BaseFeatureProcessor):
    """A processor for handling and processing a column of boolean type data.

    This class provides functionality to clean, fill missing values,
    and ensure consistency for a specified boolean column.

    Manual checks have shown that for all boolean columns in the dataset,
    there are only 5 or 6 missing (None) values in total. Therefore,
    the most convenient and efficient way to handle it is by replacing
    missing values with the most frequent value (True or False).

    Inherits from:
        BaseFeatureProcessor
    """

    def __init__(self, column_name: str):
        """Initializes the BooleanFeatureProcessor with the column name.

        Args:
            column_name (str): The name of the column to be processed.
        """
        super().__init__(column_name)  # Initialize the base class

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Method that processes boolean column.

        Processes the specified boolean column by replacing missing values
        with the most common value (True or False) and converting the column
        to a boolean type.

        Args:
            data (pd.DataFrame): The input DataFrame containing the column to process.

        Returns:
            pd.DataFrame: The DataFrame with the processed column where:
                - Missing values are filled with the most common value.
                - The column is converted to boolean type.
        """
        # Determine the most common value (True or False)
        most_common = data[self.column_name] >= 0.5

        data[self.column_name].fillna(most_common)
        data[self.column_name] = data[self.column_name].astype(int)
        return data
