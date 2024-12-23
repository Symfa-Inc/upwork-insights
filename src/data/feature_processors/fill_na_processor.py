import pandas as pd

from src.data.feature_processors.base_processor import BaseProcessor


class FillNaProcessor(BaseProcessor):
    """A processor to fill NaN values in a column with the mode (most frequent value).

    Attributes:
        column_name (str): The name of the column to process.
        mode (Any): The mode (most frequent value) of the column, determined during fitting.
    """

    def __init__(self, column_name: str):
        """Initializes the FillNaProcessor.

        Args:
            column_name (str): The name of the column to process.
        """
        super().__init__(column_name)
        self.mode = None  # Mode will be determined during fitting

    def _fit(self, df: pd.DataFrame):
        """Fits the processor by calculating the mode of the specified column.

        Args:
            df (pd.DataFrame): The input DataFrame to fit on.
        """
        # Calculate the mode (most frequent value) of the column
        self.mode = (
            df[self.column_name].mode().iloc[0] if not df[self.column_name].mode().empty else None
        )

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data by filling NaN values with the mode.

        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame with NaN values filled.
        """
        df[self.column_name] = df[self.column_name].fillna(self.mode)
        return df

    def get_params(self) -> dict:
        """Retrieves the parameters of the processor.

        Returns:
            dict: A dictionary containing the column name and mode.
        """
        return {
            'column_name': self.column_name,
            'mode': self.mode,
        }
