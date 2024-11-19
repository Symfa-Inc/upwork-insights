from abc import ABC, abstractmethod

import pandas as pd
import unidecode
from Levenshtein import ratio


class LocationProcessor(ABC):
    """Abstract base class for processing location data."""

    def __init__(self, database_path, column_name):
        self.database = self._load_data(database_path, column_name)

    @staticmethod
    def _normalize(name):
        return unidecode.unidecode(name.lower().strip())

    def _load_data(self, database_path, column_name):
        data = pd.read_csv(database_path, usecols=[column_name])
        return [self._normalize(item) for item in data[column_name].dropna().unique()]

    @abstractmethod
    def get_similar(self, name, top_k):
        pass


class CityProcessor(LocationProcessor):
    """Class for processing city names."""

    def get_similar(self, name, top_k=5):
        normalized_name = self._normalize(name)
        similarities = [(city, ratio(normalized_name, city)) for city in self.database]
        results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        return results


class CountryProcessor(LocationProcessor):
    """Class for processing country names."""

    def get_similar(self, name, top_k=5):
        normalized_name = self._normalize(name)
        similarities = [(country, ratio(normalized_name, country)) for country in self.database]
        results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        return results


# Example Usage
csv_path = 'data/geodata.csv'
city_processor = CityProcessor(csv_path, 'city_ascii')
country_processor = CountryProcessor(csv_path, 'country')

test_city_names = ['Örhus', 'Aarhus', 'aarhus', 'Aarhus C', 'Aarhus N', 'Aarhus V']
for city_name in test_city_names:
    print(city_name, city_processor.get_similar(city_name, top_k=3))

test_country_names = ['Dnemark', 'Danmark', 'Denmark', 'Denmark C', 'Denmark N', 'Denmark V']
for country_name in test_country_names:
    print(country_name, country_processor.get_similar(country_name, top_k=3))
