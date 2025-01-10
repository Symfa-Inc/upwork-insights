import json
import logging
import pickle
from typing import List

import numpy as np
import pandas as pd

from src.data.feature_processors.base_processor import BaseProcessor
from src.data.feature_processors.boolean_processor import BooleanProcessor
from src.data.feature_processors.ordinal_processor import OrdinalProcessor

logger = logging.getLogger(__name__)


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

        This method also filters columns using only those which are actually processed.

        Args:
            df: The initial df to be processed.

        Returns:
            The processed df after all processors have been applied.
        """
        columns = [processor.column_name for processor in self.processors]
        logger.info(f"Columns used: {columns}")
        df = df[columns]

        total_processors = len(self.processors)
        for i, processor in enumerate(self.processors, start=1):
            # Log the current processor and progress
            logger.info(
                f"Stage {i}/{total_processors}: "
                f"Executing '{processor.__class__.__name__}' for column '{processor.column_name}'",
            )
            # Apply the processor
            df = processor.process(df)
            # Log completion of the current stage
            logger.info(f"Stage {i}/{total_processors} completed successfully.")

        logger.info('Pipeline execution completed successfully.')
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for processor in self.processors:
            df = processor.transform(df)
        return df

    def generate_report(self) -> str:
        """Generates a JSON report of all processors and their parameters.

        Returns:
            str: A JSON string representation of the pipeline configuration.
        """

        def serialize(obj):
            """Converts non-serializable objects into JSON-compatible formats."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert ndarray to list
            elif isinstance(obj, set):
                return list(obj)  # Convert set to list
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)  # Convert numpy floats to Python float
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)  # Convert numpy ints to Python int
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)  # Convert numpy bools or native bool to Python bool
            elif obj is None:
                return None
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}  # Recursively serialize dictionary
            elif isinstance(obj, list):
                return [serialize(v) for v in obj]  # Recursively serialize list
            else:
                return obj  # Return as-is if already serializable

        report = []
        for i, processor in enumerate(self.processors, start=1):
            processor_info = {
                'Processor': i,
                'column_name': processor.column_name,
                'processor_class': processor.__class__.__name__,
                'parameters': serialize(processor.get_params()),  # Serialize parameters
            }
            report.append(processor_info)

        return json.dumps(report, indent=4)

    def save_pipeline(self, path: str) -> None:
        """Saves the FeatureProcessingPipeline object to a file using pickle.

        Args:
            path (str): The file path where the pipeline object will be saved.

        Raises:
            IOError: If the file cannot be written.
        """
        try:
            with open(path, 'wb') as file:
                pickle.dump(self, file)
            logger.info(f"Pipeline saved successfully to {path}.")
        except Exception as e:
            logger.error(f"Failed to save pipeline to {path}: {e}")
            raise

    def save_report(self, path: str) -> None:
        """Saves the pipeline report to a file.

        Args:
            path (str): The file path where the report will be saved.

        Raises:
            IOError: If the file cannot be written.
        """
        try:
            report = self.generate_report()
            with open(path, 'w') as file:
                file.write(report)
            logger.info(f"Pipeline report saved successfully to {path}.")
        except Exception as e:
            logger.error(f"Failed to save pipeline report to {path}: {e}")
            raise


if __name__ == '__main__':
    # Example usage
    boolean_processor = BooleanProcessor(column_name='boolean_col')
    ordinal_processor = OrdinalProcessor(column_name='ordinal_col')
    pipeline = FeatureProcessingPipeline()
    pipeline.add_processor(boolean_processor)
    pipeline.add_processor(ordinal_processor)

    # Sample DataFrame
    data = pd.DataFrame(
        {
            'boolean_col': [True, False, True, None],
            'ordinal_col': ['low', 'medium', 'high', None],
        },
    )

    processed_data = pipeline.execute(data)
    print(processed_data)
