from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas as pd
import unidecode
from Levenshtein import ratio


class LocationProcessor(ABC):
    """Abstract base class for processing location data."""

    def __init__(
        self,
        column_name: str,
        database_path: str = 'data/geodata.csv',
    ) -> None:
        self.database = self._load_data(database_path, column_name)

    @staticmethod
    def _normalize(name: str) -> str:
        return unidecode.unidecode(name.lower().strip())

    def _load_data(
        self,
        database_path: str,
        column_name: str,
    ) -> List[str]:
        data = pd.read_csv(database_path, usecols=[column_name])
        return [self._normalize(item) for item in data[column_name].dropna().unique()]

    @abstractmethod
    def get_similar(
        self,
        name: str,
        top_k: int = 1,
    ):
        pass


class CityProcessor(LocationProcessor):
    """Class for processing city names."""

    def get_similar(
        self,
        name: str,
        top_k: int = 3,
        threshold: float = 0.75,
    ) -> List[Tuple[str, float]]:
        normalized_name = self._normalize(name)
        similarities = [(city, ratio(normalized_name, city)) for city in self.database]
        filtered_results = [item for item in similarities if item[1] >= threshold]
        final_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:top_k]
        return final_results


class CountryProcessor(LocationProcessor):
    """Class for processing country names."""

    def get_similar(
        self,
        name: str,
        top_k: int = 3,
        threshold: float = 0.75,
    ) -> List[Tuple[str, float]]:
        normalized_name = self._normalize(name)
        similarities = [(city, ratio(normalized_name, city)) for city in self.database]
        filtered_results = [item for item in similarities if item[1] >= threshold]
        final_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:top_k]
        return final_results


if __name__ == '__main__':
    # Example Usage
    csv_path = 'data/geodata.csv'

    city_processor = CityProcessor(database_path=csv_path, column_name='city_ascii')
    test_city_names = ['Örhus', 'Århus', 'Aarhus', 'aarhus', 'Aarhus C', 'Aarhus N', 'Aarhus V']
    for city_name in test_city_names:
        result = city_processor.get_similar(city_name, top_k=1, threshold=0.7)
        if result:
            target_country_name, target_score = result[0]
            print(f'{city_name} -> {target_country_name.capitalize()} (Score: {target_score:.2f})')
        else:
            print(f'{city_name} -> No match found')

    country_processor = CountryProcessor(database_path=csv_path, column_name='country')
    test_country_names = ['Dnemark', 'Danmark', 'Denmarc', 'Denmark C', 'Denmark N', 'Denmark V']
    for country_name in test_country_names:
        result = country_processor.get_similar(country_name, top_k=1, threshold=0.7)
        if result:
            target_country_name, target_score = result[0]
            print(
                f'{country_name} -> {target_country_name.capitalize()} (Score: {target_score:.2f})',
            )
        else:
            print(f'{country_name} -> No match found')
