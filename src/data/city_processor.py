from typing import Optional, Tuple

import pandas as pd
import unidecode
from geonamescache import GeonamesCache
from Levenshtein import ratio

from src.data.country_processor import CountryProcessor


class CityProcessor:
    """Class for processing city names."""

    def __init__(
        self,
        database_path: str = 'data/geodata.csv',
    ) -> None:
        data = pd.read_csv(database_path, usecols=['city', 'city_ascii', 'iso3', 'lat', 'lng'])
        self.database = [
            (
                self._normalize(row['city']),
                self._normalize(row['city_ascii']),
                self._normalize(row['iso3']),
            )
            for _, row in data.dropna(subset=['city', 'city_ascii']).iterrows()
        ]

    @staticmethod
    def _normalize(name: str) -> str:
        return unidecode.unidecode(name.lower().strip())

    def get_similar(
        self,
        name: str,
        country: str,
        threshold: float = 0.75,
    ) -> Tuple[Optional[str], Optional[float]]:
        """Find the best match for a city name and country, returning the city name, latitude, and longitude.

        Args:
            name (str): The city name to search for.
            country (str): The country ISO3 code to filter by.
            threshold (float): The minimum similarity score for a match.

        Returns:
            Optional[Tuple[str, float, float]]: The matched city name, latitude, and longitude, or None if no match is found.
        """
        normalized_name = self._normalize(name)
        normalized_country = self._normalize(country)

        # Filter database by country
        filtered_db = [
            (city, city_ascii, iso3)
            for city, city_ascii, iso3 in self.database
            if normalized_country == iso3
        ]

        # Calculate similarity scores
        similarities = [
            (city, max(ratio(normalized_name, city), ratio(normalized_name, city_ascii)))
            for city, city_ascii, iso3 in filtered_db
        ]

        # Filter results based on the threshold
        filtered_results = [(city, score) for city, score in similarities if score >= threshold]

        # Sort results by similarity score in descending order and pick the best match
        if filtered_results:
            best_match = sorted(filtered_results, key=lambda x: x[1], reverse=True)[0]
            return best_match[0].title(), best_match[1]
        return None, None


class CityProcessorGeoCache:
    """Class for processing city names using GeonamesCache."""

    def __init__(self, min_city_population: int = 500) -> None:
        self.gc = GeonamesCache(min_city_population=min_city_population)
        self.country_processor = CountryProcessor()
        self.city_data = self.gc.get_cities()
        self.cities_database = self._build_database()
        self.population_database = self._build_population_database()

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize a string for comparison."""
        return unidecode.unidecode(str(name).lower().strip())

    def _convert_iso2_to_iso3(self, iso2: str) -> Optional[str]:
        """Convert ISO2 country code to ISO3."""
        return self.country_processor.iso2_to_iso3[iso2]

    def _build_population_database(self) -> dict[str, dict[str, int]]:
        """Build the population database grouped by ISO3 country codes and cities."""
        population_data: dict[str, dict[str, int]] = {}

        for _, city in self.city_data.items():
            # Convert country code to ISO3
            iso3_country = self._convert_iso2_to_iso3(city['countrycode'])
            if not iso3_country:
                continue
            # Normalize original city name
            original_name = unidecode.unidecode(city['name'])
            population = city['population']
            # Check if country in the database
            if iso3_country not in population_data:
                population_data[iso3_country] = {}
            population_data[iso3_country][original_name] = population
        return population_data

    def _build_database(self) -> dict[str, list]:
        """Build the city database grouped by ISO3 country codes."""
        grouped_data: dict[str, list] = {}

        for _, city in self.city_data.items():
            # Convert country code to ISO3
            iso3_country = self._convert_iso2_to_iso3(city['countrycode'])
            if not iso3_country:
                continue
            # Normalize original city name
            original_name = unidecode.unidecode(city['name'])
            # Expand with alternative names if available
            alternative_names = city.get('alternatenames', [])

            # This is awful crutch but I dont know how to fix it
            if city['geonameid'] in [5082331]:
                alternative_names = list(filter(lambda x: x not in ['New York'], alternative_names))
            # End of crutch

            all_names = [
                (self._normalize(name), original_name)
                for name in alternative_names + [original_name]
            ]
            # Add to grouped data
            if iso3_country not in grouped_data:
                grouped_data[iso3_country] = []
            grouped_data[iso3_country].extend(all_names)
        return grouped_data

    def get_similar(
        self,
        name: str,
        country: str,
        threshold: float = 0.75,
    ) -> Tuple[Optional[str], Optional[float]]:
        """Find the best match for a city name and country, returning the city name.

        Args:
            name (str): The city name to search for.
            country (str): The country ISO3 code to filter by.
            threshold (float): The minimum similarity score for a match.

        Returns:
            Optional[str]: The matched city name, or None if no match is found.
        """
        normalized_name = self._normalize(name)
        # Retrieve cities for the specified country
        city_list = self.cities_database.get(country, [])
        # Calculate similarity scores
        similarities = [
            (original_name, ratio(normalized_name, alt_name))
            for alt_name, original_name in city_list
        ]
        # Filter results based on the threshold
        filtered_results = [
            (original_name, score) for original_name, score in similarities if score > threshold
        ]
        # Sort results by similarity score in descending order and pick the best match
        if filtered_results:
            recalculated_scores = [
                (original_name, score_alt, ratio(normalized_name, self._normalize(original_name)))
                for original_name, score_alt in filtered_results
            ]
            # Calculate a combined score using a weighted approach
            coef = 0.75
            combined_scores = [
                (
                    original_name,
                    coef * score_alt + (1 - coef) * score_original,
                )  # Adjust weights as needed
                for original_name, score_alt, score_original in recalculated_scores
            ]
            best_match = sorted(combined_scores, key=lambda x: x[1], reverse=True)[0]
            # Sort recalculated results by similarity score in descending order
            return best_match[0].title(), best_match[1]
        return None, None

    def get_population(
        self,
        city_name: str,
        country: str,
    ) -> Optional[int]:
        """Find population for city name and country, returning the population.

        This function assumes search by cleaned city names

        Args:
            city_name (str): The city name to search for.
            country (str): The country ISO3 code to filter by.

        Returns:
            Optional[int]: Population of a city, or None if no city is found.
        """
        return self.population_database.get(country, {}).get(city_name, None)


if __name__ == '__main__':
    # Example Usage
    csv_path = 'data/geodata.csv'
    city_processor = CityProcessorGeoCache()
    test_city_names = [
        ('Örhus', 'DNK'),
        ('Århus', 'DNK'),
        ('Aarhus', 'DNK'),
        ('aarhus', 'DNK'),
        ('Aarhus C', 'DNK'),
        ('Aarhus N', 'DNK'),
        ('Aarhus V', 'DNK'),
        ('LA', 'USA'),
        ('NY', 'USA'),
        ('Киев', 'UKR'),
        ('newyork', 'USA'),
        ('New York ', 'USA'),
        ('United States ', 'USA'),
    ]
    for src_city_name, src_country_name in test_city_names:
        result = city_processor.get_similar(src_city_name, src_country_name, threshold=0.8)
        if result[0] is not None:
            print(f'{src_city_name} -> {result[0]} with a score of {result[1]:.2f}')
        else:
            print(f'{src_city_name} -> No match found')
