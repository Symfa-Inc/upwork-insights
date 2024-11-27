from typing import Dict, Optional, Tuple

import unidecode
from geonamescache import GeonamesCache
from Levenshtein import ratio

from src.data.country_processor import CountryProcessor


class CityProcessor:
    """Class for processing city names using GeonamesCache."""

    def __init__(self, min_city_population: int = 500) -> None:
        self.gc = GeonamesCache(min_city_population=min_city_population)
        self.country_processor = CountryProcessor()
        self.city_data = self.gc.get_cities()
        self.cities_database, self.population_database = self._build_databases()

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize a string for comparison."""
        return unidecode.unidecode(name.lower().strip())

    @staticmethod
    def normalize(name: str) -> str:
        """Normalize city name for output."""
        return CityProcessor._normalize(name).title()

    def _convert_iso2_to_iso3(self, iso2: str) -> Optional[str]:
        """Convert ISO2 country code to ISO3."""
        return self.country_processor.iso2_to_iso3.get(iso2)

    def _build_databases(self) -> Tuple[Dict[str, list], Dict[str, Dict[str, int]]]:
        """Build city and population databases grouped by ISO3 country codes."""
        cities_database: Dict[str, list] = {}
        population_database: Dict[str, Dict[str, int]] = {}

        for city in self.city_data.values():
            iso3_country = self._convert_iso2_to_iso3(city['countrycode'])
            if not iso3_country:
                continue

            city_name = unidecode.unidecode(city['name'])
            population = city['population']
            alternative_names = city.get('alternatenames', [])

            # Ensure problematic entries are handled cleanly
            if city['geonameid'] == 5082331:
                alternative_names = [name for name in alternative_names if name != 'New York']

            all_names = [
                (self._normalize(name), city_name) for name in alternative_names + [city_name]
            ]

            # Populate city database
            cities_database.setdefault(iso3_country, []).extend(all_names)

            # Populate population database
            if iso3_country not in population_database:
                population_database[iso3_country] = {}
            population_database[iso3_country][city_name] = population

        return cities_database, population_database

    def get_similar(
        self,
        name: str,
        country: str,
        threshold: float = 0.80,
    ) -> Tuple[Optional[str], Optional[float]]:
        """Find the best match for a city name and country, returning the city name.

        Args:
            name (str): The city name to search for.
            country (str): The country ISO3 code to filter by.
            threshold (float): The minimum similarity score for a match.

        Returns:
            Tuple[Optional[str], Optional[float]]: The matched city name and its similarity score.
        """
        normalized_name = self._normalize(name)
        city_list = self.cities_database.get(country, [])
        if not city_list:
            return None, None

        # Calculate similarity scores and filter by threshold
        matches = [
            (original_name, ratio(normalized_name, alt_name))
            for alt_name, original_name in city_list
            if ratio(normalized_name, alt_name) > threshold
        ]

        if not matches:
            return None, None

        # Sort matches by descending similarity and return the best match
        best_match = max(matches, key=lambda x: x[1])
        return best_match[0].title(), best_match[1]

    def get_population(
        self,
        city_name: str,
        country: str,
    ) -> Optional[int]:
        """Find population for a city in a specific country.

        Args:
            city_name (str): The city name to search for.
            country (str): The country ISO3 code.

        Returns:
            Optional[int]: Population of the city, or None if not found.
        """
        return self.population_database.get(country, {}).get(city_name)


if __name__ == '__main__':
    # Example Usage
    city_processor = CityProcessor()
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
        ('New York', 'USA'),
        ('New York City', 'USA'),
        ('Киев', 'UKR'),
        ('newyork', 'USA'),
        ('New York ', 'USA'),
        ('United States ', 'USA'),
    ]

    print('City Name -> Best Match -> Similarity Score -> Population')
    for city_name, country_code in test_city_names:
        match, score = city_processor.get_similar(city_name, country_code, threshold=0.8)
        if match:
            population = city_processor.get_population(match, country_code)
            print(f"{city_name} -> {match} -> {score:.2f} -> {population}")
        else:
            print(f"{city_name} -> No match found")
