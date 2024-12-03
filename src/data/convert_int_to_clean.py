import asyncio
import logging
import os
from typing import Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from openai import AsyncOpenAI

from src import PROJECT_DIR
from src.data.city_processor import CityProcessor
from src.data.country_processor import CountryProcessor
from src.data.location_normalizer import LocationNormalizer
from src.data.utils import COLUMNS_TO_REMOVE, DATASET_COLUMN_MAPPING, get_csv_converters

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def clean_country_names(
    df: pd.DataFrame,
    country_processor: CountryProcessor,
    country_col: str = 'COMPANY_COUNTRY',
    new_country_col: str = 'GEO_COUNTRY_NAME',
) -> pd.DataFrame:
    """Standardize country names in the DataFrame by mapping raw country names to standardized ISO3 codes.

    This function processes a DataFrame containing raw country data, replacing specific country names
    with standardized values, and uses a `CountryProcessor` to convert raw country names into ISO3 codes.
    The standardized country names are added as a new column.

    Args:
        df (pd.DataFrame): Input DataFrame containing country data.
        country_processor (CountryProcessor): An instance of `CountryProcessor` used for standardizing country names.
        country_col (str): The column name in the DataFrame containing raw country names.
        new_country_col (str): The column name to be added for standardized ISO3 country names.

    Returns:
        pd.DataFrame: A DataFrame with an additional column (`new_country_col`) containing standardized ISO3 country names.
    """
    # Filter rows with non-null COMPANY_COUNTRY values
    df = df.loc[df[country_col].notnull()].copy()
    # Replace specific country names with standardized values
    country_replacements = {
        # Used to clean jobs dataset
        'Palestinian Territories': 'PSE',
        'Macedonia': 'MKD',
        'United States Virgin Islands': 'VIR',
        # Used to clean GDP dataset
        'Czech Republic (Czechia)': 'CZE',
        'Côte d\'Ivoire': 'CIV',
        'DR Congo': 'COD',
        'State of Palestine': 'PSE',
        'Congo': 'COG',
        'Timor-Leste': 'TLS',
        'Saint Kitts & Nevis': 'KNA',
        'St. Vincent & Grenadines': 'VCT',
        'Sao Tome & Principe': 'STP',
    }
    df[country_col] = df[country_col].replace(country_replacements)
    # Get unique country values
    unique_countries = df[country_col].unique()
    country_mapping = {
        country: country_processor.get_similar(country) for country in unique_countries
    }
    # Map the processed countries to the DataFrame
    df[new_country_col] = df[country_col].map(country_mapping)
    # Log a sample for debugging
    logging.info(df[[new_country_col, new_country_col]].sample(10, random_state=42))
    return df


def add_country_population(
    df: pd.DataFrame,
    country_processor: CountryProcessor,
    country_col: str = 'GEO_COUNTRY_NAME',
    population_col: str = 'GEO_COUNTRY_POPULATION',
) -> pd.DataFrame:
    """Add a population column to a DataFrame based on country names.

    This function uses a `CountryProcessor` instance to fetch population data for each country
    listed in the specified column of the DataFrame. The population data is added as a new column.

    Args:
        df (pd.DataFrame): Input DataFrame containing country names.
        country_processor (CountryProcessor): An instance of `CountryProcessor` with a method
                                              to retrieve population data by country name.
        country_col (str): The column name in the DataFrame containing country names.
                           Default is 'GEO_COUNTRY_NAME'.
        population_col (str): The column name to be added for country population data.
                              Default is 'GEO_COUNTRY_POPULATION'.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column containing the population
                      data for each country.
    """
    df[population_col] = df[country_col].apply(country_processor.get_population)
    return df


def get_city_names_mapping(
    df: pd.DataFrame,
    city_processor: CityProcessor,
    openai_processor: LocationNormalizer,
    city_col: str = 'COMPANY_CITY',
    country_col: str = 'GEO_COUNTRY_NAME',
) -> dict[str, dict[str, str]]:
    """Generate a mapping of raw city names to standardized city names, grouped by country.

    This function processes a DataFrame containing raw city and country data, standardizing
    city names using `CityProcessorGeoCache` for geographic validation and correction.
    If `CityProcessorGeoCache` yields a low similarity score, the OpenAI API (via `OpenAIProcessor`) is used
    as a fallback to infer the standardized city name. The resulting mapping is structured as:

        {
            "Country1": {
                "Raw City 1": "Standardized City 1",
                "Raw City 2": "Standardized City 2",
                ...
            },
            "Country2": {
                ...
            }
        }

    Args:
        df (pd.DataFrame): Input DataFrame containing raw city and country information.
        city_processor (CityProcessor): An instance of `CityProcessorGeoCache` for validating and standardizing city names.
        openai_processor (LocationNormalizer): An instance of `OpenAIProcessor` for inferring city names
                                            in ambiguous cases.
        city_col (str): The column name in the DataFrame containing raw city names.
        country_col (str): The column name in the DataFrame containing raw country names.

    Returns:
        dict[str, dict[str, str]]: A nested dictionary mapping raw city names to standardized city names,
                                   grouped by their corresponding country.
    """
    # Extract unique combinations of city and country from the DataFrame
    unique_combinations = df[[city_col, country_col]].drop_duplicates().dropna()
    # First loop: Process cities using CityProcessor
    city_mapping = {}
    low_confidence_cities = []
    for _, row in unique_combinations.iterrows():
        city = row[city_col]  # Raw city name
        country = row[country_col]  # Country tag
        # Use CityProcessor to get the standardized city name and similarity score
        cleaned_name, score = city_processor.get_similar(city, country)
        if score and score >= 0.88:
            city_mapping[(city, country)] = cleaned_name
        else:
            low_confidence_cities.append((city, country))
    # Second loop: Queue OpenAI requests for low-confidence cities
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    openai_requests = [
        openai_processor.get_city(city, country) for city, country in low_confidence_cities
    ]
    openai_results = loop.run_until_complete(asyncio.gather(*openai_requests))
    loop.close()
    # Third loop: Process OpenAI responses
    for (city, country), cleaned_name in zip(low_confidence_cities, openai_results):
        if cleaned_name not in ['""', '\'\'', '']:
            city_mapping[(city, country)] = city_processor.normalize(cleaned_name)
            continue

        city_mapping[(city, country)] = None
    # Fourth loop: Create the nested dictionary mapping
    result: dict[str, dict[str, str]] = {}
    for (city, country), cleaned_name in city_mapping.items():
        if country not in result:
            result[country] = {}
        result[country][city] = cleaned_name
    return result


def clean_city_names(
    df: pd.DataFrame,
    city_processor: CityProcessor,
    location_normalizer: LocationNormalizer,
    city_col: str = 'COMPANY_CITY',
    country_col: str = 'GEO_COUNTRY_NAME',
    new_city_col: str = 'GEO_CITY_NAME',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize city names in a DataFrame by mapping raw city names to standardized city names.

    This function uses `CityProcessorGeoCache` for primary city name standardization and validation, and falls back
    to `OpenAIProcessor` for ambiguous cases. The standardized city names are added as a new column
    to the DataFrame. The mapping is performed using the `get_city_names_mapping` function to create a
    nested dictionary of country-to-city mappings, which is then applied across the rows of the DataFrame.

    Additionally, the function prints statistics before and after the transformation:
    - Total count of null and non-null city names.
    - Count of unique city names.

    The function also generates a DataFrame containing the mapping between raw city names and standardized
    city names, grouped by their respective countries. This mapping can be saved or analyzed further.

    Args:
        df (pd.DataFrame): Input DataFrame containing raw city and country data.
        city_processor (CityProcessor): An instance of `CityProcessorGeoCache` to handle city name validation
                                         and standardization.
        location_normalizer (LocationNormalizer): An instance of `OpenAIProcessor` to infer city names
                                            in ambiguous or low-confidence cases.
        city_col (str): The column name in the DataFrame containing raw city names.
        country_col (str): The column name in the DataFrame containing raw country names.
        new_city_col (str): The column name to be added to the DataFrame for standardized city names.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame]:
        - The updated DataFrame with an added column (`new_city_col`) containing standardized city names.
        - A mapping DataFrame with columns `old_name`, `country`, and `new_name` for further analysis.
    """
    # Log statistics before transformation
    total_nulls_before = df[city_col].isnull().sum()
    total_not_nulls_before = df[city_col].notnull().sum()
    unique_cities_before = df[[city_col, country_col]].nunique()
    # Generate the city names mapping
    city_mapping = get_city_names_mapping(
        df,
        city_processor,
        location_normalizer,
    )

    # Define a mapping function
    def map_city(row):
        country = row[country_col]
        city = row[city_col]
        return city_mapping.get(country, {}).get(city, None)

    # Apply the mapping function to create a new column
    df[new_city_col] = df.apply(map_city, axis=1)
    # Log statistics after transformation
    total_nulls_after = df[new_city_col].isnull().sum()
    total_not_nulls_after = df[new_city_col].notnull().sum()
    unique_cities_after = df[new_city_col].nunique()
    logging.info(f"Null city names: {total_nulls_before} -> {total_nulls_after}")
    logging.info(f"Non-null city names: {total_not_nulls_before} -> {total_not_nulls_after}")
    logging.info(f"Unique cities: {unique_cities_before} -> {unique_cities_after}")
    # Flatten the nested dictionary into a list of tuples
    rows = [
        {'old_name': old_name, 'country': country, 'new_name': new_name}
        for country, city_dict in city_mapping.items()
        for old_name, new_name in city_dict.items()
    ]
    # Convert the list of tuples into a DataFrame
    mapping_df = pd.DataFrame(rows, columns=['old_name', 'country', 'new_name'])
    return df, mapping_df


def map_city_agglomerations(
    df: pd.DataFrame,
    openai_processor: LocationNormalizer,
    city_processor: CityProcessor,
    city_col: str = 'GEO_CITY_NAME',
    state_col: str = 'COMPANY_STATE',
    country_col: str = 'GEO_COUNTRY_NAME',
) -> dict[tuple[str, Optional[str], str], str]:
    """Create a mapping of unique city, state, and country combinations to their agglomerations.

    Args:
        df (pd.DataFrame): Input DataFrame containing city, state, and country data.
        openai_processor (LocationNormalizer): An instance of `LocationNormalizer` used to determine
                                               the agglomeration of a city.
        city_processor (CityProcessor):
        city_col (str): The column name in the DataFrame containing city names. Default is 'GEO_CITY_NAME'.
        state_col (str): The column name in the DataFrame containing state names (optional).
                         Default is 'COMPANY_STATE'.
        country_col (str): The column name in the DataFrame containing country names.
                           Default is 'GEO_COUNTRY_NAME'.

    Returns:
        dict[tuple[str, Optional[str], str], str]: A mapping of (city, state, country) to agglomerations.
    """
    # Extract unique city, state, and country combinations
    unique_combinations = (
        df[[city_col, state_col, country_col]].drop_duplicates().dropna().to_records(index=False)
    )
    unique_combinations = [(row[0], row[1] or None, row[2]) for row in unique_combinations]
    max_retries = 3
    cooldown = 1
    batch_size = 1000

    # Async function to process agglomerations
    async def process_batch(batch):
        """Process a batch of requests concurrently."""

        async def request_with_retries(city, country, state):
            retries = 0
            while retries < max_retries:
                try:
                    res = await openai_processor.get_agglomeration(city, country, state)
                    return city_processor.normalize(res)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        print(f"Failed to process {city}, {country}, {state}: {e}")
                        return None
                    await asyncio.sleep(cooldown)  # Short delay before retry

        tasks = [request_with_retries(city, country, state) for city, state, country in batch]
        results = await asyncio.gather(*tasks)
        return list(zip(batch, results))

    async def process_agglomerations():
        """Process all unique combinations in batches."""
        results = []
        for i in range(0, len(unique_combinations), batch_size):
            batch = unique_combinations[i : i + batch_size]
            batch_results = await process_batch(batch)
            results.extend(batch_results)
            if i + batch_size < len(unique_combinations):
                await asyncio.sleep(cooldown)  # Cooldown between batches
        mapping = {(city, state, country): agg for (city, state, country), agg in results}
        return mapping

    # Run the async function and return the mapping
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    mapping = loop.run_until_complete(process_agglomerations())
    loop.close()
    return mapping


def add_city_agglomeration(
    df: pd.DataFrame,
    openai_processor: LocationNormalizer,
    city_processor: CityProcessor,
    city_col: str = 'GEO_CITY_NAME',
    state_col: str = 'COMPANY_STATE',
    country_col: str = 'GEO_COUNTRY_NAME',
    agglomeration_col: str = 'GEO_CITY_AGGLOMERATION',
) -> pd.DataFrame:
    """Add a column to a DataFrame with agglomeration data for each city.

    This function uses an `OpenAIProcessor` instance to determine if each city in the DataFrame
    belongs to a larger metropolitan area (agglomeration). The agglomeration information is added
    as a new column in the DataFrame. If the city does not belong to an agglomeration, the city name
    itself is returned in the new column.

    Args:
        df (pd.DataFrame): Input DataFrame containing city, state, and country data.
        openai_processor (LocationNormalizer): An instance of `OpenAIProcessor` used to determine
                                            the agglomeration of a city.
        city_col (str): The column name in the DataFrame containing city names.
                        Default is 'GEO_CITY_NAME'.
        state_col (str): The column name in the DataFrame containing state names (optional).
                         Default is 'COMPANY_STATE'.
        country_col (str): The column name in the DataFrame containing country names.
                           Default is 'GEO_COUNTRY_NAME'.
        agglomeration_col (str): The column name to be added for agglomeration data.
                                 Default is 'GEO_CITY_AGGLOMERATION'.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column containing agglomeration
                      data for each city.
    """
    # Create the mapping for unique city, state, and country combinations
    mapping = map_city_agglomerations(
        df,
        openai_processor,
        city_processor,
    )
    # Map the agglomeration data to the DataFrame
    df[agglomeration_col] = df.apply(
        lambda row: mapping.get((row[city_col], row[state_col], row[country_col]), None),
        axis=1,
    )
    return df


def add_city_population(
    df: pd.DataFrame,
    city_processor: CityProcessor,
    city_col: str = 'GEO_CITY_NAME',
    country_col: str = 'GEO_COUNTRY_NAME',
    city_population_col: str = 'GEO_CITY_POPULATION',
) -> pd.DataFrame:
    """Add a population column to a DataFrame based on city and country data.

    This function uses a `CityProcessorGeoCache` instance to retrieve population data for each
    city in the DataFrame, grouped by its respective country. The population data is added
    as a new column to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing city and country information.
        city_processor (CityProcessor): An instance of `CityProcessorGeoCache` used to retrieve
                                                population data based on city and country.
        city_col (str): The column name in the DataFrame containing city names.
                        Default is 'GEO_CITY_NAME'.
        country_col (str): The column name in the DataFrame containing country names.
                           Default is 'GEO_COUNTRY_NAME'.
        city_population_col (str): The column name to be added for city population data.
                                   Default is 'GEO_CITY_POPULATION'.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column containing population data
                      for each city.
    """

    def map_population(row):
        country = row[country_col]
        city = row[city_col]
        return city_processor.get_population(city, country)

    df[city_population_col] = df.apply(map_population, axis=1)
    return df


def clean_postcodes(
    df: pd.DataFrame,
    city_col: str = 'COMPANY_CITY',
) -> pd.DataFrame:
    """Replaces postal codes to city names in a DataFrame based on known postal codes.

    This function replaces postal codes in the specified city column with their corresponding
    city names based on a predefined mapping. It standardizes the city names for rows where
    postal codes are used instead of actual city names.

    Args:
        df (pd.DataFrame): Input DataFrame containing a column with city names or postal codes.
        city_col (str): The name of the column in the DataFrame that contains city names
                        or postal codes (default is 'COMPANY_CITY').

    Returns:
        pd.DataFrame: A DataFrame with postal codes replaced by their corresponding city names
                      in the specified column.
    """
    postcodes = {
        '1020': 'Vienna',
        '6340': 'Baar',
        '10249': 'Berlin',
        '86928': 'Hofstetten',
        '95447': 'Bayreuth',
        '83022': 'Rosenheim',
        '93320': 'Les Pavillons-sous-Bois',
        '67930': 'Beinheim',
        '97233': 'Schoelcher',
        '11000': 'Belgrade',
        '11137': 'Stockholm',
        '10024': 'New York City',
        '33602': 'Tampa',
        '89119': 'Las Vegas',
    }
    df[city_col] = df[city_col].replace(postcodes)
    return df


def load_gdp_data() -> dict[str, float]:
    """Load GDP per capita data and return it as a dictionary.

    This function reads a CSV file containing GDP per capita data for various countries and combines it with
    manually added GDP values for certain countries not present in the dataset. The GDP data is mapped by
    ISO3 country codes.

    Returns:
        dict[str, float]: A dictionary where the keys are ISO3 country codes and the values are GDP per capita
                          (float) for the year 2023.

    Notes:
        - Manually added GDP data includes:
            - Taiwan (TWN): 72485.0
            - Jersey (JEY): 57092.0
            - United States Minor Outlying Islands (UMI): 46381.0
            - Guadeloupe (GLP): 23695.0
            - Cook Islands (COK): 21994.0
        - GDP data source: https://databank.worldbank.org/reports.aspx?source=2&series=NY.GDP.MKTP.CD&country#
        - 'TWN', 'JEY', 'UMI', 'GLP', 'COK' - Source: wikipedia
    """
    gdp_df = pd.read_csv('data/gdp_data.csv', usecols=['Country Code', 'GDPPC'])
    result = {row['Country Code']: row['GDPPC'] for _, row in gdp_df.iterrows()}
    return result


def add_gdp_data(
    df,
    country_col: str = 'GEO_COUNTRY_NAME',
    new_gdp_col: str = 'GEO_COUNTRY_GDPPC',
) -> pd.DataFrame:
    """Add GDP per capita data to a DataFrame based on country names or ISO3 codes.

    This function maps GDP per capita data (loaded from `load_gdp_data`) to a specified country column in the
    DataFrame and adds a new column containing the GDP per capita values.

    Args:
        df (pd.DataFrame): The input DataFrame containing country data.
        country_col (str, optional): Name of the column in the DataFrame containing country names or ISO3 codes.
                                     Defaults to 'GEO_COUNTRY_NAME'.
        new_gdp_col (str, optional): Name of the new column to store GDP per capita values. Defaults to
                                     'GEO_COUNTRY_GDPPC'.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column for GDP per capita values.
    """
    gdp_mapping = load_gdp_data()
    df[new_gdp_col] = df[country_col].map(gdp_mapping)
    return df


def calculate_experience(
    df: pd.DataFrame,
    join_date_col: str = 'COMPANY_CONTRACTDATE',
    experience_col: str = 'COMPANY_EXPERIENCE',
) -> pd.DataFrame:
    """Calculate experience in months based on the join date column and a fixed reference date.

    This function calculates the experience as the number of months between the date in the `join_date_col`
    column and December 1, 2024. The calculated experience is stored in a new column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the join date column.
        join_date_col (str): Name of the column containing join dates. Defaults to 'CONTRACTOR_DATE'.
        experience_col (str): Name of the column to store the calculated experience in months.
                              Defaults to 'COMPANY_EXPERIENCE'.

    Returns:
        pd.DataFrame: The input DataFrame with the added column for experience in months.
    """
    # Define the reference date (December 1, 2024)
    reference_date = pd.Timestamp('2024-12-01')
    # Convert the join date column to datetime
    df[join_date_col] = pd.to_datetime(df[join_date_col], errors='coerce')
    # Calculate the experience in months
    df[experience_col] = df[join_date_col].apply(
        lambda x: (
            (reference_date.year - x.year) * 12 + (reference_date.month - x.month)
            if pd.notnull(x)
            else None
        ),
    )
    return df


def calculate_wh_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate work history duration in hours.

    Calculate a new column 'WH_DURATION' based on the difference in hours between
    'WH_STARTDATE' and 'WH_ENDDATE'. If 'WH_ENDDATE' is null, 'WH_DURATION' will remain null.

    Args:
        df (pd.DataFrame): Input dataframe containing 'WH_STARTDATE' and 'WH_ENDDATE' columns.

    Returns:
        pd.DataFrame: The updated dataframe with a new column 'WH_DURATION'.
    """
    # Ensure columns are datetime
    df['WH_STARTDATE'] = pd.to_datetime(df['WH_STARTDATE'])
    df['WH_ENDDATE'] = pd.to_datetime(df['WH_ENDDATE'])
    # Calculate duration in hours
    df['WH_DURATION'] = df.apply(
        lambda row: (
            (row['WH_ENDDATE'] - row['WH_STARTDATE']).total_seconds() / 3600
            if pd.notnull(row['WH_ENDDATE'])
            else None
        ),
        axis=1,
    )
    return df


def clean_duration_label(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and consolidate engagement duration labels.

    Add a new column 'ENGAGEMENTDURATIONLABEL' to the dataframe based on the logic:
    - If JOBTYPE is 'FIXED', take the value from FIXEDPRICEENGAGEMENTDURATIONLABEL.
    - If JOBTYPE is 'HOURLY', take the value from HOURLYENGAGEMENTDURATIONLABEL.

    Args:
        df (pd.DataFrame): Input dataframe containing JOBTYPE, HOURLYENGAGEMENTDURATIONLABEL,
                           and FIXEDPRICEENGAGEMENTDURATIONLABEL columns.

    Returns:
        pd.DataFrame: The updated dataframe with a new column ENGAGEMENTDURATIONLABEL.
    """
    df['ENGAGEMENTDURATIONLABEL'] = df['OPENINGDURATION']
    return df


def update_engagement_type(df: pd.DataFrame) -> pd.DataFrame:
    """Update engagement type based on job type.

    Update the column HOURLYENGAGEMENTTYPE with a new column ENGAGEMENTTYPE:
    - If JOBTYPE is 'FIXED', set ENGAGEMENTTYPE to 'project'.
    - If JOBTYPE is 'HOURLY', set ENGAGEMENTTYPE to the value of HOURLYENGAGEMENTTYPE.

    Args:
        df (pd.DataFrame): Input dataframe containing JOBTYPE and HOURLYENGAGEMENTTYPE.

    Returns:
        pd.DataFrame: The updated dataframe with a new column ENGAGEMENTTYPE.
    """
    df['ENGAGEMENTTYPE'] = df.apply(
        lambda row: 'PROJECT' if row['JOBTYPE'] == 'FIXED' else row['HOURLYENGAGEMENTTYPE'],
        axis=1,
    )
    return df


def create_budget_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create budget min and max columns.

    Create two new columns 'BUDGET_MIN' and 'BUDGET_MAX':
    - If JOBTYPE is 'HOURLY', set values from 'HOURLYBUDGETMIN' and 'HOURLYBUDGETMAX'.
    - If JOBTYPE is 'FIXED', set both columns to the value from 'FIXEDPRICEAMOUNT'.

    Args:
        df (pd.DataFrame): Input dataframe containing 'JOBTYPE', 'FIXEDPRICEAMOUNT',
                           'HOURLYBUDGETMIN', and 'HOURLYBUDGETMAX'.

    Returns:
        pd.DataFrame: The updated dataframe with new columns 'Budget Min' and 'Budget Max'.
    """
    df['BUDGET_MIN'] = df.apply(
        lambda row: (
            row['FIXEDPRICEAMOUNT'] if row['JOBTYPE'] == 'FIXED' else row['HOURLYBUDGETMIN']
        ),
        axis=1,
    )
    df['BUDGET_MAX'] = df.apply(
        lambda row: (
            row['FIXEDPRICEAMOUNT'] if row['JOBTYPE'] == 'FIXED' else row['HOURLYBUDGETMAX']
        ),
        axis=1,
    )
    return df


def calculate_hire_rate(
    df: pd.DataFrame,
    jobs_posted_count: str = 'COMPANY_JOBSPOSTEDCOUNT',
    jobs_filled_count: str = 'COMPANY_JOBSFILLEDCOUNT',
    hire_rate_col: str = 'COMPANY_HIRE_RATE',
) -> pd.DataFrame:
    df[hire_rate_col] = df[jobs_filled_count] / df[jobs_posted_count]
    df[hire_rate_col] = df[hire_rate_col].replace([np.inf, -np.inf], np.nan)
    df[hire_rate_col] = df[hire_rate_col].fillna(0)
    df[hire_rate_col] = df[hire_rate_col].clip(lower=0, upper=1)
    return df


def clean_unnecessary_columns(
    df: pd.DataFrame,
    columns_to_remove: list,
) -> pd.DataFrame:
    """Remove unnecessary columns from a DataFrame."""
    return df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')


def rename_features(
    df: pd.DataFrame,
    column_mapping: dict[str, str],
) -> pd.DataFrame:
    df = df.rename(columns=column_mapping)
    df = df[column_mapping.values()]
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
    country_processor = CountryProcessor()
    city_processor = CityProcessor()
    location_normalizer = LocationNormalizer(AsyncOpenAI())
    df = calculate_hire_rate(df)
    df = calculate_wh_duration(df)
    df = calculate_experience(df)
    df = clean_duration_label(df)
    df = update_engagement_type(df)
    df = create_budget_columns(df)

    # Process the dataset
    df = clean_country_names(df, country_processor)
    df = df[df['GEO_COUNTRY_NAME'].notna()]
    df = add_country_population(df, country_processor)
    df = add_gdp_data(df)

    df = clean_postcodes(df)
    df, df_cities = clean_city_names(
        df,
        city_processor,
        location_normalizer,
    )
    df = df[df['GEO_CITY_NAME'].notna()]
    df = add_city_population(df, city_processor)
    df = add_city_agglomeration(df, location_normalizer, city_processor)

    df['RISINGTALENT'] = df['RISINGTALENT'].fillna(False)
    df['RISINGTALENT'] = df['RISINGTALENT'].astype(bool)

    # Drop unnecessary columns
    df = clean_unnecessary_columns(df, columns_to_remove=COLUMNS_TO_REMOVE)

    # Rename and reorder columns
    df = rename_features(df, DATASET_COLUMN_MAPPING)

    # Save the clean dataset
    os.makedirs(save_dir, exist_ok=True)
    city_save_path = os.path.join(save_dir, 'cities.csv')
    df_cities.to_csv(city_save_path, index=False)
    data_save_path = os.path.join(save_dir, 'clean.csv')  # noqa: F841
    df.to_csv(data_save_path, index=False)

    log.info('Complete')


if __name__ == '__main__':
    main()
