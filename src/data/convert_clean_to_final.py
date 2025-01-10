import logging
import os

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src import PROJECT_DIR
from src.data.feature_processors import FeatureProcessingPipeline
from src.data.pipeline_stages import STAGES
from src.data.utils import get_csv_converters

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def set_up_pipeline() -> FeatureProcessingPipeline:
    return FeatureProcessingPipeline(
        [processor(column, **kwargs) for column, processor, kwargs in STAGES],
    )


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='convert_clean_to_final',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))
    data_path = str(os.path.join(PROJECT_DIR, cfg.data_path))

    # Read the dataset
    df = pd.read_csv(data_path, converters=get_csv_converters())  # .sample(1000)  # noqa: F841

    # Load pipeline
    pipeline = set_up_pipeline()

    # Transform data with the pipeline
    df = pipeline.execute(df)

    # Save the final dataset
    os.makedirs(save_dir, exist_ok=True)
    data_save_path = os.path.join(save_dir, 'final.parquet')  # noqa: F841
    df.to_parquet(data_save_path, index=False)

    pipeline_save_path = os.path.join(save_dir, 'pipeline.pkl')  # noqa: F841
    pipeline.save_pipeline(pipeline_save_path)

    report_save_path = os.path.join(save_dir, 'report.json')
    pipeline.save_report(report_save_path)

    log.info('Complete')


if __name__ == '__main__':
    main()
