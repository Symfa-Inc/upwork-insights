from abc import ABC, abstractmethod


class BaseFeatureProcessor(ABC):
    """An abstract base class for feature processors, defining a structure for all feature processing classes.

    Attributes:
        column_name (str): The name of the column to be processed.
    """

    def __init__(self, column_name):
        """Initializes the processor with the column name.

        Args:
            column_name (str): The name of the column to process.
        """
        self.column_name = column_name

    @abstractmethod
    def process(self, data):
        """Abstract method for processing data. This must be implemented by all subclasses.

        Args:
            data (pd.DataFrame): The input DataFrame to process.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        pass
