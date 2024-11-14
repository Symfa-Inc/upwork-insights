import logging
import os

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src import PROJECT_DIR

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
    # data_dir = str(os.path.join(PROJECT_DIR, cfg.data_dir))
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))

    # TODO (Misha): Iterate through CSVs and convert to interim
    # Print columns for all tables
    for name, path in cfg.data.files.items():
        try:
            df = pd.read_csv(path, nrows=0)
            logging.info(f"Columns in {name}:")
            logging.info(df.columns.tolist())
            logging.info('\n' + '=' * 40 + '\n')
        except Exception as e:
            logging.info(f"Error loading {name} at {path}: {e}")

    #
    # Merge JOBS and WORK_HISTORY
    jobs_df = pd.read_csv(cfg.data.files.JOBS)
    work_history_df = pd.read_csv(cfg.data.files.WORK_HISTORY)
    # Log the initial length of both tables
    logging.info(f"Initial length of JOBS table: {len(jobs_df)}")
    logging.info(f"Initial length of WORK_HISTORY table: {len(work_history_df)}")
    # Merge the tables on JOBS.ID = WORK_HISTORY.JOBID, with intersection only
    merged_df = pd.merge(
        jobs_df,
        work_history_df.add_prefix('WH_'),  # Prefix WORK_HISTORY columns with "WH_"
        left_on='ID',
        right_on='WH_JOBID',
        how='inner',  # Only keep rows present in both tables
    )
    # Log the length of the merged table
    logging.info(f"Length of merged table (intersection): {len(merged_df)}")

    #
    # Extend resulting table with company info
    companies_df = pd.read_csv(cfg.data.files.COMPANIES)
    # Log the initial length of the COMPANIES table
    logging.info(f"Initial length of COMPANIES table: {len(companies_df)}")
    # Merge with the COMPANIES table on JOBS.COMPANYUID = COMPANIES.UID
    merged_df = pd.merge(
        merged_df,
        companies_df.add_prefix('COMPANY_'),  # Prefix COMPANIES columns with "company_"
        left_on='COMPANYUID',
        right_on='COMPANY_UID',
        how='left',  # Use left join to retain all rows in merged_df
    )
    # Log the length of the merged dataset
    logging.info(f"Length of dataset after merging with COMPANIES: {len(merged_df)}")

    #
    # Load the ADDITIONALSKILLS table
    additional_skills_df = pd.read_csv(cfg.data.files.ADDITIONALSKILLS)
    # Log the initial length of the ADDITIONALSKILLS table
    logging.info(f"Initial length of ADDITIONALSKILLS table: {len(additional_skills_df)}")
    # Create a list of PREFLABEL values for each JOBID
    additional_skills_grouped = (
        additional_skills_df.groupby('JOBID')['PREFLABEL']
        .apply(list)
        .reset_index()
        .rename(columns={'PREFLABEL': 'ADDITIONAL_SKILLS'})
    )
    # Merge the grouped additional skills into the main result
    merged_df = pd.merge(
        merged_df,
        additional_skills_grouped,
        left_on='ID',  # Assuming JOBS.ID in the merged result is equivalent to ADDITIONALSKILLS.JOBID
        right_on='JOBID',
        how='left',  # Left join to retain all jobs in the main result
    )
    # Drop the redundant JOBID column from the merge
    merged_df.drop(columns=['JOBID'], inplace=True)
    # Log the length of the final dataset
    logging.info(f"Length of dataset after adding additional skills: {len(merged_df)}")

    #
    # Load the JOB_TAGS table
    job_tags_df = pd.read_csv(cfg.data.files.JOB_TAGS)
    # Log the initial length of the JOB_TAGS table
    logging.info(f"Initial length of JOB_TAGS table: {len(job_tags_df)}")
    # Group tags by JOBID and aggregate TAG values into lists
    tags_grouped = (
        job_tags_df.groupby('JOBID')['TAG']
        .apply(list)
        .reset_index()
        .rename(columns={'TAG': 'TAGS'})
    )
    # Merge the grouped tags into the main result
    merged_df = pd.merge(
        merged_df,
        tags_grouped,
        left_on='ID',  # Assuming JOBS.ID in the merged result is equivalent to JOB_TAGS.JOBID
        right_on='JOBID',
        how='left',  # Left join to retain all jobs in the main result
    )
    # Drop the redundant JOBID column from the merge
    merged_df.drop(columns=['JOBID'], inplace=True)
    # Log the length of the final dataset after adding tags
    logging.info(f"Length of dataset after adding tags: {len(merged_df)}")

    #
    # Load the OCCUPATIONS table
    occupations_df = pd.read_csv(cfg.data.files.OCCUPATIONS)
    # Log the initial length of the OCCUPATIONS table
    logging.info(f"Initial length of OCCUPATIONS table: {len(occupations_df)}")
    # Group occupations by JOBID and aggregate PREFLABEL values into lists
    occupations_grouped = (
        occupations_df.groupby('JOBID')['PREFLABEL']
        .apply(list)
        .reset_index()
        .rename(columns={'PREFLABEL': 'OCCUPATIONS'})
    )
    # Merge the grouped occupations into the main result
    merged_df = pd.merge(
        merged_df,
        occupations_grouped,
        left_on='ID',  # Assuming JOBS.ID in the merged result is equivalent to OCCUPATIONS.JOBID
        right_on='JOBID',
        how='left',  # Left join to retain all jobs in the main result
    )
    # Drop the redundant JOBID column from the merge
    merged_df.drop(columns=['JOBID'], inplace=True)
    # Log the length of the final dataset after adding occupations
    logging.info(f"Length of dataset after adding occupations: {len(merged_df)}")

    #
    # Load the SEGMENTATIONDATA table
    segmentation_data_df = pd.read_csv(cfg.data.files.SEGMENTATIONDATA)
    # Log the initial length of the SEGMENTATIONDATA table
    logging.info(f"Initial length of SEGMENTATIONDATA table: {len(segmentation_data_df)}")
    # Create a list of tuples for each JOBID, containing specified columns
    segmentation_data_grouped = (
        segmentation_data_df.groupby('JOBID')[['NAME', 'VALUE', 'LABEL', 'TYPE', 'SORTORDER']]
        .apply(
            lambda x: list(x.itertuples(index=False, name=None)),
        )  # Convert each group to a list of tuples
        .reset_index()
        .rename(columns={0: 'SEGMENTATIONDATA'})
    )
    # Merge the grouped segmentation data into the main result
    merged_df = pd.merge(
        merged_df,
        segmentation_data_grouped,
        left_on='ID',  # Assuming JOBS.ID in the merged result is equivalent to SEGMENTATIONDATA.JOBID
        right_on='JOBID',
        how='left',  # Left join to retain all jobs in the main result
    )
    # Drop the redundant JOBID column from the merge
    merged_df.drop(columns=['JOBID'], inplace=True)
    # Log the length of the final dataset after adding segmentation data
    logging.info(f"Length of dataset after adding segmentation data: {len(merged_df)}")

    #
    # Load the ONTOLOGY_SKILLS and JOB_ONTOLOGY_SKILLS tables
    ontology_skills_df = pd.read_csv(cfg.data.files.ONTOLOGY_SKILLS)
    job_ontology_skills_df = pd.read_csv(cfg.data.files.JOB_ONTOLOGY_SKILLS)
    # Log the initial lengths of the ONTOLOGY_SKILLS and JOB_ONTOLOGY_SKILLS tables
    logging.info(f"Initial length of ONTOLOGY_SKILLS table: {len(ontology_skills_df)}")
    logging.info(f"Initial length of JOB_ONTOLOGY_SKILLS table: {len(job_ontology_skills_df)}")
    # Merge job_ontology_skills_df with ontology_skills_df on SKILLID and ID to get PREFLABEL for each skill
    job_skills_with_labels = pd.merge(
        job_ontology_skills_df,
        ontology_skills_df[['ID', 'PREFLABEL']],  # Only keep necessary columns
        left_on='SKILLID',
        right_on='ID',
        how='inner',
    )
    # Group PREFLABEL values by JOBID into lists
    ontology_skills_grouped = (
        job_skills_with_labels.groupby('JOBID')['PREFLABEL']
        .apply(list)
        .reset_index()
        .rename(columns={'PREFLABEL': 'ONTOLOGY_SKILLS'})
    )
    # Merge the grouped ontology skills into the main result
    final_df = pd.merge(
        merged_df,
        ontology_skills_grouped,
        left_on='ID',  # Assuming JOBS.ID in the merged result is equivalent to JOB_ONTOLOGY_SKILLS.JOBID
        right_on='JOBID',
        how='left',  # Left join to retain all jobs in the main result
    )
    # Drop the redundant JOBID column from the merge
    final_df.drop(columns=['JOBID'], inplace=True)
    # Log the length of the final dataset after adding ontology skills
    logging.info(f"Length of dataset after adding ontology skills: {len(final_df)}")

    #
    # Drop unnecessary columns
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
        'WH_JOBID',
        'WH_JOBUID',
        'COMPANY_RID',
        'COMPANY_LOGOURL',
    ]
    final_df.drop(
        columns=columns_to_drop,
        inplace=True,
    )

    logging.info('\n' + '=' * 40 + '\n')
    logging.info('Columns in final dataset:')
    logging.info(final_df.columns.tolist())
    logging.info('\n' + '=' * 40 + '\n')

    # Save interim dataset
    log.info(f'Saving interim dataset to {save_dir}')
    os.makedirs(save_dir, exist_ok=True)
    final_df.to_csv(save_dir + '/finished_jobs.csv')

    log.info('Complete')


if __name__ == '__main__':
    main()
