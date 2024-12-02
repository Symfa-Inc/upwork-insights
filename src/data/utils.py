import ast
import os
from pathlib import Path
from typing import List, Union

import pandas as pd

DATASET_COLUMN_MAPPING = {
    'ID': 'id',
    'TITLE': 'title',
    'DESCRIPTION': 'description',
    'CONNECTPRICE': 'connect_price',
    'JOBTYPE': 'job_type',
    'CONTRACTORTIER': 'contractor_tier',
    'PUBLISHTIME': 'publsih_time',
    'PERSONSTOHIRE': 'persons_to_hire',
    'PREMIUM': 'is_premium',
    'OCCUPATIONPREFLABEL': 'occupation',
    'OPENINGCOVERLETTERREQUIRED': 'is_cover_letter_required',
    'OPENINGACCESS': 'opening_access',
    'OPENINGFREELANCERMILESTONESALLOWED': 'is_milestones_allowed',
    'OPENINGVISIBILITY': 'opening_visibility',
    'CATEGORYNAME': 'category',
    'CATEGORYGROUPNAME': 'category_group',
    'TYPE': 'type',
    'BROWSER': 'browser',
    'DEVICE': 'device',
    'ENGLISHSKILL': 'english_skill',
    'ENGLISHPROFICIENCY': 'english_proficiency',
    'FREELANCERTYPE': 'freelancer_type',
    'RISINGTALENT': 'is_rising_talent',
    'EARNINGS': 'earnings',
    'LOCALMARKET': 'is_local_market',
    'WH_STATUS': 'wh_status',
    'WH_TOTALHOURS': 'wh_total_hours',
    'WH_FEEDBACKSCORE': 'wh_feedback_score',
    'WH_FEEDBACKTOCLIENTSCORE': 'wh_feedback_to_client_score',
    'WH_TOTALCHARGE': 'wh_total_charge',
    'WH_RATEAMOUNT': 'wh_hourly_rate',
    'WH_DURATION': 'wh_duration',
    'COMPANY_NAME': 'company_name',
    'COMPANY_DESCRIPTION': 'company_description',
    'COMPANY_SUMMARY': 'company_summary',
    'COMPANY_SIZE': 'company_size',
    'COMPANY_INDUSTRY': 'company_industry',
    'COMPANY_VISIBLE': 'company_visible',
    'COMPANY_JOBSPOSTEDCOUNT': 'company_jobs_posted_count',
    'COMPANY_JOBSFILLEDCOUNT': 'company_jobs_filled_count',
    'COMPANY_FEEDBACKCOUNT': 'company_feedback_count',
    'COMPANY_HOURSCOUNT': 'company_hours_count',
    'COMPANY_TOTALCHARGESAMOUNT': 'company_total_charges_amount',
    'COMPANY_SCORE': 'company_feedback_score',
    'COMPANY_AVGHOURLYJOBSRATEAMOUNT': 'company_avg_hourly_rate',
    'COMPANY_CSSTIER': 'company_css_tier',
    'COMPANY_HIRE_RATE': 'company_hire_rate',
    'COMPANY_EXPERIENCE': 'company_experience',
    'SEGMENTATION_DATA_VALUE': 'segmentation_data_value',
    'SEGMENTATION_DATA_LABEL': 'segmentation_data_label',
    'OCCUPATION': 'occupation',
    'ADDITIONAL_SKILLS': 'additional_skills',
    'TAGS': 'tags',
    'SKILLS': 'skills',
    'ENGAGEMENTDURATIONLABEL': 'engagement_duration',
    'ENGAGEMENTTYPE': 'engagement_type',
    'BUDGET_MIN': 'budget_min',
    'BUDGET_MAX': 'budget_max',
    'GEO_COUNTRY_NAME': 'geo_country_name',
    'GEO_COUNTRY_POPULATION': 'geo_country_population',
    'GEO_COUNTRY_GDPPC': 'geo_country_gdppc',
    'GEO_CITY_NAME': 'geo_city_name',
    'GEO_CITY_POPULATION': 'geo_city_population',
    'GEO_CITY_AGGLOMERATION': 'geo_city_agglomeration',
}


def safe_literal_eval(val):
    """Safely evaluate a string to a Python literal, handling empty strings."""
    if pd.isnull(val) or val.strip() == '':
        return []  # Treat empty strings or NaN as empty lists
    try:
        return ast.literal_eval(val)
    except (SyntaxError, ValueError):
        return None  # Replace malformed entries with None


def get_csv_converters() -> dict:
    """Returns a dictionary of converters for specific columns when reading CSV files.

    The converters are used to process specific columns in the CSV during loading with `pd.read_csv`.
    For example, columns containing list-like strings can be deserialized into Python lists.

    Returns:
        dict: A dictionary where keys are column names and values are functions
              to process the column data during CSV reading.

    Example:
        >>> converters = get_csv_converters()
        >>> df = pd.read_csv("example.csv", converters=converters)
    """
    converters = {
        'SKILLS': safe_literal_eval,
        'TAGS': safe_literal_eval,
        'ADDITIONAL_SKILLS': safe_literal_eval,
    }
    return converters


def get_file_list(
    src_dirs: Union[List[str], str],
    ext_list: Union[List[str], str],
    filename_template: str = '',
) -> List[str]:
    """Get a list of files in the specified directory with specific extensions.

    Args:
        src_dirs: directory(s) with files inside
        ext_list: extension(s) used for a search
        filename_template: include files with this template
    Returns:
        all_files: a list of file paths
    """
    all_files = []
    src_dirs = [src_dirs] if isinstance(src_dirs, str) else src_dirs
    ext_list = [ext_list] if isinstance(ext_list, str) else ext_list
    for src_dir in src_dirs:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                file_ext = Path(file).suffix
                file_ext = file_ext.lower()
                if file_ext in ext_list and filename_template in file:
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
    all_files.sort()
    return all_files


if __name__ == '__main__':
    get_file_list(
        src_dirs='data/raw',
        ext_list='.csv',
    )
