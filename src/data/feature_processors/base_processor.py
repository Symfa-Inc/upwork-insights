from abc import ABC, abstractmethod

import pandas as pd
from sklearn.exceptions import NotFittedError


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
        self._is_fit = False

    @property
    def is_fit(self) -> bool:
        """Checks if the processor has been fit.

        Returns:
            bool: True if the processor is fit, False otherwise.
        """
        return self._is_fit

    def fit(self, df: pd.DataFrame) -> None:
        """Fits the processor to the data and marks it as fit.

        Args:
            df (pd.DataFrame): The input DataFrame to fit on.
        """
        self._fit(df)
        self._is_fit = True

    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> None:
        """Abstract method to fit the processor to the data.

        This method should calculate and store any parameters needed for the transformation.

        Args:
            df (pd.DataFrame): The input DataFrame to fit on.
        """
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data based on the fitted processor.

        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.

        Raises:
            NotFittedError: If the processor is not fit before calling this method.
            ValueError: If the specified column is not found in the DataFrame.
        """
        if not self.is_fit:
            raise NotFittedError(
                f"The processor '{self.__class__.__name__}' is not fit yet. Call `fit` first.",
            )
        if self.column_name not in df.columns:
            raise ValueError(f"Column '{self.column_name}' not found in the DataFrame.")
        return self._transform(df)

    @abstractmethod
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
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

    @abstractmethod
    def get_params(self) -> dict:
        """Abstract method to retrieve the parameters or settings of the processor.

        Returns:
            dict: A dictionary containing the processor parameters.
        """
        pass
