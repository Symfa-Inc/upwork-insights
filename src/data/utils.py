import ast
import logging
import os
import pickle
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
log = logging.getLogger()

DATASET_COLUMN_MAPPING = {
    # General information
    'ID': 'id',
    'TITLE': 'title',
    'DESCRIPTION': 'description',
    'PUBLISHTIME': 'publsih_time',
    'PREMIUM': 'is_premium',
    'CONNECTPRICE': 'connect_price',
    'JOBTYPE': 'job_type',
    'CONTRACTORTIER': 'contractor_tier',
    'PERSONSTOHIRE': 'persons_to_hire',
    # Job details
    'OCCUPATION': 'occupation',
    'OPENINGCOVERLETTERREQUIRED': 'is_cover_letter_required',
    'OPENINGACCESS': 'opening_access',
    'OPENINGFREELANCERMILESTONESALLOWED': 'is_milestones_allowed',
    'OPENINGVISIBILITY': 'opening_visibility',
    'CATEGORYNAME': 'category',
    'CATEGORYGROUPNAME': 'category_group',
    'TYPE': 'job_posting_type',
    # Device and browser
    'BROWSER': 'browser',
    'DEVICE': 'device',
    # Skills and tags
    'TAGS': 'tags',
    'SKILLS': 'skills',
    'ADDITIONAL_SKILLS': 'additional_skills',
    # Language and proficiency
    'ENGLISHSKILL': 'english_skill',
    'ENGLISHPROFICIENCY': 'english_proficiency',
    # Freelancer details
    'FREELANCERTYPE': 'freelancer_type',
    'RISINGTALENT': 'is_rising_talent',
    'EARNINGS': 'earnings',
    # Geo-related fields
    'GEO_COUNTRY_NAME': 'geo_country_name',
    'GEO_COUNTRY_POPULATION': 'geo_country_population',
    'GEO_COUNTRY_GDPPC': 'geo_country_gdppc',
    'GEO_CITY_NAME': 'geo_city_name',
    'GEO_CITY_POPULATION': 'geo_city_population',
    'GEO_CITY_AGGLOMERATION': 'geo_city_agglomeration',
    'LOCALMARKET': 'is_local_market',
    # Work history
    'WH_STATUS': 'wh_status',
    'WH_TOTALHOURS': 'wh_total_hours',
    'WH_FEEDBACKSCORE': 'wh_feedback_score',
    'WH_FEEDBACKTOCLIENTSCORE': 'wh_feedback_to_client_score',
    'WH_TOTALCHARGE': 'wh_total_charge',
    'WH_RATEAMOUNT': 'wh_hourly_rate',
    'WH_DURATION': 'wh_duration',
    # Company details
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
    # Segmentation
    'SEGMENTATION_DATA_VALUE': 'segmentation_data_value',
    'SEGMENTATION_DATA_LABEL': 'segmentation_data_label',
    # Engagement details
    'ENGAGEMENTDURATIONLABEL': 'engagement_duration',
    'ENGAGEMENTTYPE': 'engagement_type',
    'BUDGET_MIN': 'budget_min',
    'BUDGET_MAX': 'budget_max',
}

# The list of features to be deleted is described at https://symfa.fibery.io/RnD/Description-360
COLUMNS_TO_REMOVE = [
    'COMPANY_CONTRACTDATE',
    'COMPANY_CITY',
    'COMPANY_COUNTRY',
    'HOURLYENGAGEMENTTYPE',
    'OPENINGDURATION',
    'OPENINGHOURLYBUDGETMIN',
    'OPENINGHOURLYBUDGETMAX',
    'OPENINGWORKLOAD',
    'OPENINGHOURLYBUDGETMIN',
    'OPENINGHOURLYBUDGETMAX',
    'OPENINGAMOUNT',
    'FIXEDPRICEAMOUNT',
    'OCCUPATIONPREFLABEL',
    'HOURLYBUDGETMIN',
    'HOURLYBUDGETMAX',
    'APPLIED',
    'OPENINGFREELANCERSTOHIRE',
    'OPENINGCONTRACTORTIER',
    'OPENINGTYPE',
    'OPENINGHOURLYBUDGETTYPE',
    'WH_RESPONSE_FOR_FREELANCER_FEEDBACK',
    'WH_RESPONSE_FOR_CLIENT_FEEDBACK',
    'WH_FEEDBACKTOCLIENTCOMMENT',
    'WH_FEEDBACKCOMMENT',
    'WH_STARTDATE',
    'WH_ENDDATE',
    'ISSTSVECTORSEARCHRESULT',
    'CREATETIME',
    'ENTERPRISEJOB',
    'TOTALAPPLICANTS',
    'OCCUPATIONENTITYSTATUS',
    'ISPREMIUM',
    'ISTOPRATED',
    'OPENINGENGAGEMENTTYPE',
    'OPENINGTYPE',
    'OPENINGAMOUNT',
    'OPENINGCONTRACTORTIER',
    'OPENINGFREELANCERSTOHIRE',
    'OPENINGHIDDEN',
    'OPENINGSITESOURCE',
    'OPENINGKEEPOPENONHIRE',
    'OPENINGAUTOREVIEWSTATUS',
    'OPENINGWORKLOAD',
    'OPENINGDURATION',
    'SITESOURCE',
    'TOTALTIMEJOBPOSTFLOWAIV2',
    'TOTALTIMESPENTONREVIEWPAGEAIV2',
    'STARTTIMEJOBPOSTFLOWAIV2',
    'SOURCINGUPDATECOUNT',
    'SOURCINGUPDATEFORBIDDEN',
    'JOBSUCCESSSCORE',
    'LOCATIONCHECKREQUIRED',
    'COMPANYUID',
    'PARSED',
    'WH_PARSED',
    'COMPANY_URL',
    'COMPANY_ISCOMPANYVISIBLEINPROFILE',
    'COMPANY_ISEDCREPLICATED',
    'COMPANY_STATE',
    'COMPANY_COUNTRYTIMEZONE',
    'COMPANY_OFFSETFROMUTCMILLIS',
    'COMPANY_JOBSOPENCOUNT',
    'COMPANY_TOTALASSIGNMENTS',
    'COMPANY_ACTIVEASSIGNMENTSCOUNT',
    'COMPANY_TOTALJOBSWITHHIRES',
    'COMPANY_ISPAYMENTMETHODVERIFIED',
    'COMPANY_PARSED',
    'SEGMENTATION_DATA_NAME',
    'SEGMENTATION_DATA_TYPE',
    'SEGMENTATION_DATA_SORTORDER',
]


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
        converters = get_csv_converters()
        df = pd.read_csv("example.csv", converters=converters)
    """
    converters = {
        'SKILLS': safe_literal_eval,
        'skills': safe_literal_eval,
        'TAGS': safe_literal_eval,
        'tags': safe_literal_eval,
        'ADDITIONAL_SKILLS': safe_literal_eval,
        'additional_skills': safe_literal_eval,
    }
    return converters


def extract_fitted_attributes(obj) -> dict:
    """Extracts all fitted attributes (those ending with '_') from a scikit-learn object.

    Args:
        obj: A scikit-learn transformer or estimator.

    Returns:
        dict: A dictionary of fitted attributes and their values.
    """
    return {
        attr: (
            getattr(obj, attr).tolist()
            if isinstance(getattr(obj, attr), np.ndarray)
            else getattr(obj, attr)
        )
        for attr in dir(obj)
        if attr.replace('__', '').endswith('_') and not attr.startswith('_')
    }


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


def normalize_to_snake_name(skill: str):
    return skill.replace(' ', '_').lower()


def get_embeddings(
    texts: List[str],
    batch_size: int = 1000,
    model: str = 'text-embedding-3-large',
) -> np.ndarray:
    """Generates embeddings for a list of strings using OpenAI's API.

    Args:
        texts (List[str]): A list of strings to generate embeddings for.
        batch_size (int): The size of the batches for API requests. Defaults to 1000.
        model (str): The OpenAI model to use for generating embeddings. Defaults to 'text-embedding-3-large'.

    Returns:
        np.ndarray: A numpy array of embeddings.
    """
    cleaned_texts = [text if isinstance(text, str) and text != '' else 'Missing' for text in texts]
    embeddings = []

    openai_client = OpenAI()
    for i in range(0, len(cleaned_texts), batch_size):
        batch = cleaned_texts[i : i + batch_size]
        response = openai_client.embeddings.create(input=batch, model=model)
        embeddings.extend([res.embedding for res in response.data])
    return np.array(embeddings)


def save_model_to_pickle(model, model_path: str):
    """Save a model to a pickle file."""
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        log.info(f"Model saved to {model_path}")
    except Exception as e:
        log.error(f"Failed to save model to {model_path}: {e}")


def load_model_from_pickle(model_path: str):
    """Load a model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        log.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        log.error(f"Failed to load model from {model_path}: {e}")
        return None


if __name__ == '__main__':
    get_file_list(
        src_dirs='data/raw',
        ext_list='.csv',
    )
