import pandas as pd

from src.data.feature_processors.base_processor import BaseProcessor


class DeleteProcessor(BaseProcessor):
    """A processor that removes the specified column from the DataFrame.

    Attributes:
        column_name (str): The name of the column to be deleted.
    """

    def _fit(self, df: pd.DataFrame) -> None:
        """No fitting is required for this processor."""
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deletes the specified column from the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with the column removed.
        """
        df = df.drop(columns=[self.column_name])
        return df

    def get_params(self) -> dict:
        """Retrieves the parameters of the processor."""
        return {'column_name': self.column_name}
