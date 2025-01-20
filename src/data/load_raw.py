import logging
import os
from contextlib import contextmanager

import hydra
import snowflake.connector
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src import PROJECT_DIR

load_dotenv()

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@contextmanager
def manage_tmp_stage(cursor, stage_name: str):
    try:
        # Setup phase: Create the temporary stage
        log.info(f"Creating temporary stage {stage_name} in Snowflake...")
        cursor.execute(f"CREATE OR REPLACE STAGE {stage_name}")
        yield  # Transfer control to the `with` block
    except Exception as e:
        # Handle exceptions if needed
        log.error(f"An error occurred: {e}")
        raise  # Re-raise the exception
    finally:
        # Cleanup phase: Ensure the stage is dropped
        log.info(f"Removing stage {stage_name}...")
        cursor.execute(f"DROP STAGE IF EXISTS {stage_name}")
        log.info(f"Stage {stage_name} removed successfully.")


def save_tables_into_stage(cursor, stage_name: str):
    cursor.execute('SHOW TABLES IN SCHEMA UPWORK_JOBS_V2')
    tables = cursor.fetchall()
    table_names = [table[1] for table in tables]  # Use correct index for table name

    for table_name in table_names:
        log.info(f"Saving table {table_name} into stage...")
        cursor.execute(
            f"""
            COPY INTO @{stage_name}/{table_name}.csv.gz
            FROM {table_name}
            FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY = '"')
            SINGLE=TRUE MAX_FILE_SIZE=536870912 HEADER=TRUE;
        """,
        )


def load_tables_from_stage(cursor, stage_name: str, saving_path: str):
    if not saving_path.startswith('file://'):
        saving_path = f"file://{os.path.abspath(saving_path)}"

    log.info('Downloading files from stage into local directory...')
    cursor.execute(f"LIST @{stage_name}")
    files_in_stage = cursor.fetchall()

    for file_info in files_in_stage:
        file_path = file_info[0]  # Adjust index to fetch file name
        log.info(f"Downloading: {file_path}")
        cursor.execute(f"GET @{file_path} '{saving_path}'")

    log.info(f"All files downloaded in {saving_path}")


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='load_raw',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    save_dir = os.path.join(PROJECT_DIR, cfg.save_dir)
    stage_name = cfg.snowflake_tmp_stage
    os.makedirs(save_dir, exist_ok=True)

    # Establish Snowflake connection
    conn = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA'),
        role=os.getenv('SNOWFLAKE_ROLE'),
    )
    cursor = conn.cursor()

    try:
        with manage_tmp_stage(cursor, stage_name):
            save_tables_into_stage(cursor, stage_name)
            load_tables_from_stage(cursor, stage_name, save_dir)
    finally:
        cursor.close()
        conn.close()

    log.info('Complete')


if __name__ == '__main__':
    main()
