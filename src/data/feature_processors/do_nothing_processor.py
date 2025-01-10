import pandas as pd

from src.data.feature_processors.base_processor import BaseProcessor


class DoNothingProcessor(BaseProcessor):
    """A processor that performs no transformation on the specified column.

    Attributes:
        column_name (str): The name of the column to process.
    """

    def _fit(self, df: pd.DataFrame) -> None:
        """No fitting is required for this processor."""
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns the input DataFrame unchanged."""
        return df

    def get_params(self) -> dict:
        """Retrieves the parameters of the processor."""
        return {'column_name': self.column_name}
