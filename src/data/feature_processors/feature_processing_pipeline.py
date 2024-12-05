from typing import List

from src.data.feature_processors.base_feature_processor import BaseFeatureProcessor


class FeatureProcessingPipeline:
    """A class to define a modular pipeline for feature processing.

    Attributes:
        processors (list): A list of processor objects, each inheriting from `BaseFeatureProcessor`.
    """

    def __init__(self, processors: List[BaseFeatureProcessor] = None):
        """Initializes the pipeline with a list of processors.

        Args:
            processors (List[BaseFeatureProcessor], optional): A list of processor instances to initialize the pipeline.
        """
        self.processors: List[BaseFeatureProcessor] = processors if processors else []

    def add_processor(self, processor: BaseFeatureProcessor) -> None:
        """Adds a processor to the pipeline.

        Args:
            processor (BaseFeatureProcessor): A processor instance to add to the pipeline.
        """
        self.processors.append(processor)

    def execute(self, data):
        """Executes the pipeline by applying all processors sequentially to the input data.

        Args:
            data: The initial data to be processed.

        Returns:
            The processed data after all processors have been applied.
        """
        for processor in self.processors:
            data = processor.process(data)
        return data
