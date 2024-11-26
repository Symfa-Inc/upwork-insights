import logging
import os

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

from src import PROJECT_DIR
from src.data.city_processor import CityProcessor
from src.data.country_processor import CountryProcessor
from src.data.location_normalizer import LocationNormalizer
from src.data.utils import get_csv_converters

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
    # Initialize the result dictionary
    result: dict[str, dict[str, str]] = {}
    # Iterate through each unique city-country pair
    for _, row in unique_combinations.iterrows():
        city = row[city_col]  # Raw city name
        country = row[country_col]  # Raw country name
        # Use CityProcessor to get the standardized city name and similarity score
        cleaned_name, score = city_processor.get_similar(city, country)
        # If the similarity score is below the threshold, use OpenAI to infer the city
        if not score or score < 0.88:
            cleaned_name = openai_processor.get_city(city, country)
        # If no valid city name is found, set it to None
        if not cleaned_name:
            cleaned_name = None
        # Ensure the country key exists in the result dictionary
        if country not in result:
            result[country] = {}
        # Map the raw city name to the cleaned name
        result[country][city] = cleaned_name
    # Return the nested mapping of countries to their city mappings
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
        city_processor=city_processor,
        openai_processor=location_normalizer,
        city_col=city_col,
        country_col=country_col,
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


def add_city_agglomeration(
    df: pd.DataFrame,
    openai_processor: LocationNormalizer,
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

    def map_agglomeration(row):
        country = row[country_col]
        state = row[state_col] if row[state_col] else None
        city = row[city_col]
        return openai_processor.get_agglomeration(city, country, state)

    # Apply the mapping function to create a new column
    df[agglomeration_col] = df.apply(map_agglomeration, axis=1)
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
    """
    gdp_df = pd.read_csv('data/gdp_data.csv', usecols=['Country Code', '2023 [YR2023]'])
    # these countries are not present in dataset so I add manually. Source: wikipedia
    result = {
        'TWN': 72485.0,
        'JEY': 57092.0,
        'UMI': 46381.0,
        'GLP': 23695.0,
        'COK': 21994.0,
    }
    result.update({row['Country Code']: row['2023 [YR2023]'] for _, row in gdp_df.iterrows()})
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
    location_normalizer = LocationNormalizer(OpenAI())

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
    df = add_city_agglomeration(df, location_normalizer)
    df_cities.to_csv(os.path.join(save_dir, 'cities.csv'), index=False)

    df = calculate_experience(df)

    # Drop unnecessary columns

    # Save the clean dataset
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'clean.csv')  # noqa: F841
    df.to_csv(save_path, index=False)

    log.info('Complete')


if __name__ == '__main__':
    main()
