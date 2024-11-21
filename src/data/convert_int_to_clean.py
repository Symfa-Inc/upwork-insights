import logging
import os

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src import PROJECT_DIR
from src.data.country_processor import CountryProcessor
from src.data.utils import get_csv_converters

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def convert_countries(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize country data in the DataFrame by mapping raw country names to standardized ISO3 codes.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'COMPANY_COUNTRY' column.

    Returns:
        pd.DataFrame: DataFrame with an additional 'GEO_COUNTRY' column for standardized country names.
    """
    country_processor = CountryProcessor()

    # Filter rows with non-null COMPANY_COUNTRY values
    df = df.loc[df['COMPANY_COUNTRY'].notnull()].copy()

    # Replace specific country names with standardized values
    country_replacements = {
        'Turkey': 'TUR',
        'Palestinian Territories': 'PSE',
        'Macedonia': 'MKD',
        'Curacao': 'CUW',
        'United States Virgin Islands': 'VIR',
        'Russia': 'RUS',
    }
    df['COMPANY_COUNTRY'] = df['COMPANY_COUNTRY'].replace(country_replacements)

    # Get unique country values
    unique_countries = df['COMPANY_COUNTRY'].unique()
    country_mapping = {
        country: country_processor.get_similar(country) for country in unique_countries
    }

    # Map the processed countries to the DataFrame
    df['GEO_COUNTRY'] = df['COMPANY_COUNTRY'].map(country_mapping)

    # Log a sample for debugging
    logging.info(df[['GEO_COUNTRY', 'COMPANY_COUNTRY']].sample(10, random_state=42))

    return df


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='convert_int_to_clean',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))
    data_path = str(os.path.join(PROJECT_DIR, cfg.data_path))

    # Read the dataset
    df = pd.read_csv(data_path, converters=get_csv_converters())  # noqa: F841

    # TODO: Process the dataset
    df = convert_countries(df)

    # Save the clean dataset
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'clean.csv')  # noqa: F841
    df.to_csv(save_path, index=False)

    log.info('Complete')


if __name__ == '__main__':
    main()
