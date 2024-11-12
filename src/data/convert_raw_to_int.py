import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src import PROJECT_DIR
from src.data.utils import get_file_list

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='convert_raw_to_int',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    data_dir = str(os.path.join(PROJECT_DIR, cfg.data_dir))
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))

    # Get list of CSVs
    csv_list = get_file_list(  # noqa: F841
        src_dirs=data_dir,
        ext_list='.csv',
    )

    # TODO (Misha): Iterate through CSVs and convert to interim

    # Save interim dataset
    log.info(f'Saving interim dataset to {save_dir}')
    os.makedirs(save_dir, exist_ok=True)

    log.info('Complete')


if __name__ == '__main__':
    main()
