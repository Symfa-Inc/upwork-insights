from typing import List, Tuple

import pandas as pd
import unidecode
from Levenshtein import ratio


class CityProcessor:
    """Class for processing city names."""

    def __init__(
        self,
        database_path: str = 'data/geodata.csv',
    ) -> None:
        data = pd.read_csv(database_path, usecols=['city_ascii'])
        self.database = [self._normalize(item) for item in data['city_ascii'].dropna().unique()]

    @staticmethod
    def _normalize(name: str) -> str:
        return unidecode.unidecode(name.lower().strip())

    def get_similar(
        self,
        name: str,
        threshold: float = 0.75,
    ) -> List[Tuple[str, float]]:
        normalized_name = self._normalize(name)
        similarities = [(city, ratio(normalized_name, city)) for city in self.database]
        filtered_results = [item for item in similarities if item[1] >= threshold]
        final_result = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:1]
        final_result = [(city.title(), score) for city, score in final_result]
        return final_result


if __name__ == '__main__':
    # Example Usage
    csv_path = 'data/geodata.csv'
    city_processor = CityProcessor(database_path=csv_path)
    test_city_names = ['Örhus', 'Århus', 'Aarhus', 'aarhus', 'Aarhus C', 'Aarhus N', 'Aarhus V']
    for city_name in test_city_names:
        result = city_processor.get_similar(city_name, threshold=0.75)
        if result:
            target_city_name, target_score = result[0]
            print(f'{city_name} -> {target_city_name} (Score: {target_score:.2f})')
        else:
            print(f'{city_name} -> No match found')
