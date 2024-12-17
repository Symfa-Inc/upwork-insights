from typing import Optional, Type, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.data.feature_processors.base_processor import BaseProcessor
from src.data.utils import extract_fitted_attributes


class NumericProcessor(BaseProcessor):
    """A processor for scaling numeric features with logic for handling NaN values.

    Attributes:
        column_name (str): The name of the column to process.
        scaler_class (Optional[Type[TransformerMixin]]): The scaler class to use (e.g., StandardScaler, RobustScaler).
        scaler (Optional[TransformerMixin]): The instantiated and fitted scaler.
    """

    def __init__(
        self,
        column_name: str,
        scaler_class: Type[Union[StandardScaler, MinMaxScaler, RobustScaler]] = RobustScaler,
    ):
        """Initializes the NumericProcessor with a column name and optional scaler class.

        Args:
            column_name (str): The name of the column to process.
            scaler_class (Type[Union[StandardScaler, MinMaxScaler, RobustScaler]]): The scaler class to use for scaling.
        """
        self.scaler_class = scaler_class
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]] = None
        super().__init__(column_name)

    def _fit(self, df: pd.DataFrame):
        """Fits the processor by initializing the scaler and learning the parameters.

        Args:
            df (pd.DataFrame): The input DataFrame to fit on.

        Raises:
            ValueError: If the specified column is not found in the DataFrame.
        """
        kwargs = {'with_centering': False} if self.scaler_class == RobustScaler else {}

        self.scaler = self.scaler_class(**kwargs)
        temp_df = df[[self.column_name]].dropna()

        # Fit the scaler
        self.scaler.fit(temp_df)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data using the fitted scaler.

        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame with the scaled feature.
        """
        # Extract the column to scale
        temp_df = df[[self.column_name]]

        # Scale the data (NaNs remain untouched during scaling)
        temp_df = self.scaler.transform(temp_df)

        # Replace NaNs in the scaled data with -1
        if self.scaler_class in {RobustScaler, MinMaxScaler}:
            temp_df = np.where(np.isnan(temp_df), -1, temp_df)
        else:
            temp_df = np.where(np.isnan(temp_df), 0, temp_df)

        # Update the original DataFrame with the scaled column
        df[self.column_name] = temp_df

        return df

    def get_params(self) -> dict:
        """Retrieves the parameters of the processor.

        Returns:
            dict: A dictionary containing the column name, scaler parameters, and scaler type.
        """
        return {
            'column_name': self.column_name,
            'scaler_type': type(self.scaler).__name__,
            'scaler_params': extract_fitted_attributes(self.scaler),
        }


if __name__ == '__main__':
    # Sample DataFrame
    data = pd.DataFrame(
        {
            '_numeric_column': [1, 2, 3, 4, np.nan, 6, 100, 200],
            'numeric_column_1': [1, 2, 3, 4, np.nan, 6, 100, 200],
            'numeric_column_2': [1, 2, 3, 4, np.nan, 6, 100, 200],
        },
    )

    # Processor with RobustScaler
    robust_processor = NumericProcessor(column_name='numeric_column_1')
    transformed_data_robust = robust_processor.process(data)
    print('Transformed Data with RobustScaler:')
    print(transformed_data_robust)

    # Processor with StandardScaler
    standard_processor = NumericProcessor(
        column_name='numeric_column_2',
        scaler_class=StandardScaler,
    )
    transformed_data_standard = standard_processor.process(data)
    print('\nTransformed Data with StandardScaler:')
    print(transformed_data_standard)

    # Processor Parameters
    print('\nProcessor Parameters (RobustScaler):')
    print(robust_processor.get_params())

    print('\nProcessor Parameters (StandardScaler):')
    print(standard_processor.get_params())
