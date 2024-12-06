from typing import List

import pandas as pd

from src.data.feature_processors.base_processor import BaseProcessor


class FeatureProcessingPipeline:
    """A class to define a modular pipeline for feature processing.

    Attributes:
        processors (list): A list of processor objects, each inheriting from `BaseProcessor`.
    """

    def __init__(self, processors: List[BaseProcessor] = None):
        """Initializes the pipeline with a list of processors.

        Args:
            processors (List[BaseProcessor], optional): A list of processor instances to initialize the pipeline.
        """
        self.processors: List[BaseProcessor] = processors if processors else []

    def add_processor(self, processor: BaseProcessor) -> None:
        """Adds a processor to the pipeline.

        Args:
            processor (BaseProcessor): A processor instance to add to the pipeline.
        """
        self.processors.append(processor)

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes the pipeline by applying all processors sequentially to the input df.

        Args:
            df: The initial df to be processed.

        Returns:
            The processed df after all processors have been applied.
        """
        for processor in self.processors:
            df = processor.process(df)
        return df
