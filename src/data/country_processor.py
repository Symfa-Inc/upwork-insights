from typing import Optional, Union

import geonamescache
import unidecode


class CountryProcessor:
    """Class for processing country names."""

    def __init__(self):
        # Initialize the Geonamescache instance and extract countries
        self.gc = geonamescache.GeonamesCache()
        self.countries = self.gc.get_countries()
        self.population_database = self._build_population_database()
        # Create mappings for easier lookup
        self.name_to_iso3 = {
            self._normalize(data['name']): data['iso3'] for code, data in self.countries.items()
        }
        self.iso2_to_iso3 = {code: data['iso3'] for code, data in self.countries.items()}

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize the input country name for comparison."""
        return unidecode.unidecode(name.lower().strip())

    def _build_population_database(self) -> dict[str, int]:
        """Creates a map from country tag to population."""
        population_data: dict[str, int] = {}
        for _, country in self.countries.items():
            country_tag = country['iso3']
            population = country['population']
            population_data[country_tag] = population
        return population_data

    def _get_geonamescache_match(self, name: str) -> Union[str, None]:
        """Find the closest match for a country name or ISO code using geonamescache."""
        if not isinstance(name, str):
            return None

        normalized_name = self._normalize(name)

        # Check by country name
        if normalized_name in self.name_to_iso3:
            return self.name_to_iso3[normalized_name]

        # Check by ISO-2 code
        if name.upper() in self.iso2_to_iso3:
            return self.iso2_to_iso3[name.upper()]

        # Check by ISO-3 code
        for data in self.countries.values():
            if name.upper() == data['iso3']:
                return data['iso3']

        return None

    def get_similar(self, name: str) -> Union[str, None]:
        """Get the standardized 3-letter ISO code using geonamescache.

        Args:
            name (str): Input country name or ISO code.

        Returns:
            str: Standardized 3-letter ISO code or None if no match is found.
        """
        return self._get_geonamescache_match(name)

    def get_population(
        self,
        country: str,
    ) -> Optional[int]:
        """Find population for of country, returning the population.

        This function assumes search by cleaned country tag.

        Args:
            country (str): The country ISO3 code to filter by.

        Returns:
            Optional[int]: Population of a city, or None if no city is found.
        """
        return self.population_database.get(country, None)


if __name__ == '__main__':
    # Example Usage
    country_processor = CountryProcessor()
    test_country_names = [
        'Russia',
        'RUS',
        'RU',
        'United States',
        'USA',
        'US',
        'Denmark',
        'DK',
        'DNK',
        'Test Country',
        'TCN',
        'TX',
    ]

    for country_name in test_country_names:
        result = country_processor.get_similar(country_name)
        if result:
            print(f'{country_name} -> {result}')
        else:
            print(f'{country_name} -> No match found')
