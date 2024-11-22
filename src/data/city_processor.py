from typing import Optional, Tuple

import pandas as pd
import unidecode
from Levenshtein import ratio


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
                row['lat'],
                row['lng'],
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
    ) -> Optional[Tuple[str, float, float]]:
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
            (city, city_ascii, iso3, lat, lng)
            for city, city_ascii, iso3, lat, lng in self.database
            if normalized_country == iso3
        ]

        # Calculate similarity scores
        similarities = [
            (city, lat, lng, max(ratio(normalized_name, city), ratio(normalized_name, city_ascii)))
            for city, city_ascii, iso3, lat, lng in filtered_db
        ]

        # Filter results based on the threshold
        filtered_results = [
            (city, lat, lng, score) for city, lat, lng, score in similarities if score >= threshold
        ]

        # Sort results by similarity score in descending order and pick the best match
        if filtered_results:
            best_match = sorted(filtered_results, key=lambda x: x[3], reverse=True)[0]
            return best_match[0].title(), best_match[1], best_match[2]

        return None


if __name__ == '__main__':
    # Example Usage
    csv_path = 'data/geodata.csv'
    city_processor = CityProcessor(database_path=csv_path)
    test_city_names = [
        'Örhus',
        'Århus',
        'Aarhus',
        'aarhus',
        'Aarhus C',
        'Aarhus N',
        'Aarhus V',
        'NY',
        'Киев',
    ]
    test_countries = ['DNK', 'DNK', 'DNK', 'DNK', 'DNK', 'DNK', 'DNK', 'USA', 'UKR']
    for city_name, city_country in zip(test_city_names, test_countries):
        result = city_processor.get_similar(city_name, city_country, threshold=0.75)
        if result:
            target_city_name, *_, target_score = result
            print(f'{city_name} -> {target_city_name} (Score: {target_score:.2f})')
        else:
            print(f'{city_name} -> No match found')
