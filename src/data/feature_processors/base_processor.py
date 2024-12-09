from abc import ABC, abstractmethod

import pandas as pd


class BaseProcessor(ABC):
    """An abstract base class for feature processors, defining a structure for all feature processing classes.

    Attributes:
        column_name (str): The name of the column to be processed.
    """

    def __init__(self, column_name: str):
        """Initializes the processor with the column name.

        Args:
            column_name (str): The name of the column to process.
        """
        self.column_name = column_name

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """Abstract method to fit the processor to the data.

        This method should calculate and store any parameters needed for the transformation.

        Args:
            df (pd.DataFrame): The input DataFrame to fit on.
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to transform the data based on the fitted processor.

        This method should apply the transformation to the data.

        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        pass

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes the data by first fitting the processor and then transforming the data.

        Args:
            df (pd.DataFrame): The input DataFrame to process.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        self.fit(df)
        return self.transform(df)
