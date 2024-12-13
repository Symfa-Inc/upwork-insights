import logging
import os

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src import PROJECT_DIR
from src.data.utils import get_csv_converters

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


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
    df = pd.read_csv(data_path, converters=get_csv_converters())  # noqa: F841

    # Save the clean dataset
    os.makedirs(save_dir, exist_ok=True)
    data_save_path = os.path.join(save_dir, 'final.csv')  # noqa: F841
    df.to_csv(data_save_path, index=False)

    log.info('Complete')


if __name__ == '__main__':
    main()
