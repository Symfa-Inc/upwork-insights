from typing import List, Tuple

import pandas as pd
import unidecode
from Levenshtein import ratio


class CountryProcessor:
    """Class for processing country names."""

    def __init__(self, database_path: str = 'data/geodata.csv') -> None:
        data = pd.read_csv(database_path, usecols=['country', 'iso3'])
        self.database = [
            (
                self._normalize(row['country']),
                self._normalize(row['iso3']),
            )
            for _, row in data.iterrows()
        ]

    @staticmethod
    def _normalize(name: str) -> str:
        return unidecode.unidecode(name.lower().strip())

    def get_similar(self, name: str, threshold: float = 0.75) -> List[Tuple[str, float]]:
        normalized_name = self._normalize(name)
        similarities = []
        for country, iso3 in self.database:
            scores = [
                (country, ratio(normalized_name, country)),
                (country, ratio(normalized_name, iso3)),
            ]
            best_match = max(scores, key=lambda x: x[1])
            similarities.append(best_match)

        filtered_results = [item for item in similarities if item[1] >= threshold]
        final_result = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:1]
        final_result = [(country.title(), score) for country, score in final_result]
        return final_result


if __name__ == '__main__':
    # Example Usage
    csv_path = 'data/geodata.csv'
    country_processor = CountryProcessor(database_path=csv_path)
    test_country_names = ['RUS', 'USA', 'Uni states', 'Dnemark', 'Danmark', 'Denmarc', 'DNK']
    for country_name in test_country_names:
        result = country_processor.get_similar(country_name, threshold=0.75)
        if result:
            target_country_name, target_score = result[0]
            print(f'{country_name} -> {target_country_name} (Score: {target_score:.2f})')
        else:
            print(f'{country_name} -> No match found')
