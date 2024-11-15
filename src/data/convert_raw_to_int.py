import logging
import os
from typing import Dict, Optional

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src import PROJECT_DIR

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def log_table_columns(
    file_paths: Dict,
) -> None:
    """Logs column names for each CSV file in file_paths."""
    for name, path in file_paths.items():
        try:
            df = pd.read_csv(path, nrows=0)
            log.info(f'Columns in {name}: {df.columns.tolist()}')
        except Exception as e:
            log.error(f'Error loading {name}: {e}')
    log.info('')


def load_and_clean_csv(filename: str, index_col: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV file, clean it by removing NaN or duplicate index values, and return a DataFrame.

    This function reads a single CSV file, optionally sets an index column, and cleans the data by
    dropping rows with NaN values in the index and duplicate entries. When duplicates exist, the most
    recent entry is retained based on the `PARSED` column.

    Args:
        filename (str): Path to the CSV file to be read.
        index_col (Optional[str]): Column name to use as the index. Defaults to None.

    Returns:
        pd.DataFrame: A cleaned Pandas DataFrame with duplicates removed and the index set.
    """
    df = pd.read_csv(filename, index_col=index_col)

    # Drop rows with NaN in the index
    df = df[~df.index.isnull()]
    # Remove duplicate index entries, keeping the first occurrence
    df = df.sort_values(by='PARSED').drop_duplicates(keep='last')

    return df


def load_and_merge_datasets(cfg: DictConfig) -> pd.DataFrame:
    """Loads and merges datasets as per the configuration."""
    jobs_df = load_and_clean_csv(cfg.files.JOBS, index_col='ID')
    work_history_df = load_and_clean_csv(cfg.files.WORK_HISTORY, index_col='JOBID')
    companies_df = load_and_clean_csv(cfg.files.COMPANIES, index_col='UID')
    additional_skills_df = pd.read_csv(cfg.files.ADDITIONALSKILLS)
    job_tags_df = pd.read_csv(cfg.files.JOB_TAGS)
    occupations_df = pd.read_csv(cfg.files.OCCUPATIONS)
    segmentation_data_df = pd.read_csv(cfg.files.SEGMENTATIONDATA)
    ontology_skills_df = pd.read_csv(cfg.files.ONTOLOGY_SKILLS)
    job_ontology_skills_df = pd.read_csv(cfg.files.JOB_ONTOLOGY_SKILLS)
    job_skills_with_labels = pd.merge(
        job_ontology_skills_df,
        ontology_skills_df,
        left_on='SKILLID',
        right_on='ID',
        how='inner',
    )
    # Merging datasets step-by-step
    merged_df = merge_jobs_with_work_history(jobs_df, work_history_df)
    merged_df = merge_with_companies(merged_df, companies_df)
    merged_df = merge_with_grouped_data(
        merged_df=merged_df,
        df=additional_skills_df,
        key='JOBID',
        value_col='PREFLABEL',
        new_col_name='ADDITIONAL_SKILLS',
    )
    merged_df = merge_with_grouped_data(
        merged_df=merged_df,
        df=job_tags_df,
        key='JOBID',
        value_col='TAG',
        new_col_name='TAGS',
    )
    merged_df = merge_with_grouped_data(
        merged_df=merged_df,
        df=occupations_df,
        key='JOBID',
        value_col='PREFLABEL',
        new_col_name='OCCUPATIONS',
    )
    merged_df = merge_with_grouped_data(
        merged_df=merged_df,
        df=job_skills_with_labels,
        key='JOBID',
        value_col='PREFLABEL',
        new_col_name='SKILLS',
    )
    merged_df = merge_segmentation_data(merged_df, segmentation_data_df)
    return merged_df


def merge_jobs_with_work_history(
    jobs_df: pd.DataFrame,
    work_history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merges JOBS and WORK_HISTORY tables."""
    merged_df = pd.merge(
        jobs_df,
        work_history_df.add_prefix('WH_'),
        left_on='ID',
        right_index=True,
        how='inner',
    )
    log.info(f'Initial JOBS dataset length: {len(jobs_df)}')
    log.info(f'Initial WORK_HISTORY dataset length: {len(work_history_df)}')
    log.info(f'Length after merging with WORK_HISTORY: {len(merged_df)}')
    return merged_df


def merge_with_companies(
    merged_df: pd.DataFrame,
    companies_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merges with COMPANIES table."""
    merged_df = pd.merge(
        merged_df,
        companies_df.add_prefix('COMPANY_'),
        left_on='COMPANYUID',
        right_index=True,
        how='left',
    )
    log.info(f'Initial COMPANIES dataset length: {len(companies_df)}')
    log.info(f'Length after merging with COMPANIES: {len(merged_df)}')
    return merged_df


def merge_with_grouped_data(
    merged_df: pd.DataFrame,
    df: pd.DataFrame,
    key: str,
    value_col: str,
    new_col_name: str,
) -> pd.DataFrame:
    """Merges grouped data by key, aggregating value_col into lists."""
    grouped = (
        df.groupby(key)[value_col]
        .apply(list)
        .reset_index()
        .rename(columns={value_col: new_col_name})
    )
    merged_df = pd.merge(merged_df, grouped, left_index=True, right_on=key, how='left').drop(
        columns=[key],
    )
    log.info(f'Length after adding {new_col_name}: {len(merged_df)}')
    return merged_df


def merge_segmentation_data(
    merged_df: pd.DataFrame,
    segmentation_data_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merges SEGMENTATIONDATA with grouped lists of tuples."""
    grouped = (
        segmentation_data_df.groupby('JOBID')[['NAME', 'VALUE', 'LABEL', 'TYPE', 'SORTORDER']]
        .apply(
            lambda x: list(x.itertuples(index=False, name=None)),
        )
        .reset_index()
        .rename(columns={0: 'SEGMENTATIONDATA'})
    )
    merged_df = pd.merge(merged_df, grouped, left_index=True, right_on='JOBID', how='left').drop(
        columns=['JOBID'],
    )
    log.info(f'Length after adding SEGMENTATIONDATA: {len(merged_df)}')
    return merged_df


def merge_ontology_skills(
    merged_df: pd.DataFrame,
    job_ontology_skills_df: pd.DataFrame,
    ontology_skills_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merges job skills with ontology skills by SKILLID."""
    job_skills_with_labels = pd.merge(
        job_ontology_skills_df,
        ontology_skills_df[['ID', 'PREFLABEL']],
        left_on='SKILLID',
        right_on='ID',
        how='inner',
    )
    grouped = (
        job_skills_with_labels.groupby('JOBID')['PREFLABEL']
        .apply(list)
        .reset_index()
        .rename(
            columns={'PREFLABEL': 'ONTOLOGY_SKILLS'},
        )
    )
    merged_df = pd.merge(merged_df, grouped, left_on='ID', right_on='JOBID', how='left').drop(
        columns=['JOBID'],
    )
    log.info(f'Length after adding ONTOLOGY_SKILLS: {len(merged_df)}')
    return merged_df


def drop_unnecessary_columns(df: pd.DataFrame) -> None:
    """Drops unnecessary columns from the final dataframe."""
    columns_to_drop = [
        'UID',
        'OPENINGUID',
        'RELEVANCEENCODED',
        'SOURCINGTIMESTAMP',
        'FIXEDPRICEENGAGEMENTDURATIONCTIME',
        'FIXEDPRICEENGAGEMENTDURATIONMTIME',
        'OCCUPATIONONTOLOGYID',
        'OCCUPATIONPRIMARYBROADERUID',
        'OCCUPATIONPRIMARYBROADER',
        'CATEGORYURLSLUG',
        'CATEGORYGROUPURLSLUG',
        'COMPANYUID',
        'WH_COMPANYUID',
        'WH_COMPANYID',
        'COMPANY_RID',
        'COMPANY_LOGOURL',
    ]
    df.drop(columns=columns_to_drop, inplace=True)
    log.info(f'Final dataset columns: {len(df.columns)}')
    log.info(f'Final dataset length: {len(df)}')


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='convert_raw_to_int',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))
    cfg.files = {key: os.path.join(PROJECT_DIR, value) for key, value in cfg.files.items()}
    os.makedirs(save_dir, exist_ok=True)

    # Load and log table columns
    log_table_columns(file_paths=cfg.files)

    # Load, merge, and process datasets
    df = load_and_merge_datasets(cfg)

    # Drop redundant columns
    drop_unnecessary_columns(df)

    # Save the final processed dataset
    save_path = os.path.join(save_dir, 'jobs.csv')
    df.to_csv(save_path, index=False)

    log.info('Complete')


if __name__ == '__main__':
    main()
