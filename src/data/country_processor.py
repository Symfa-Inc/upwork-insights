from typing import Optional

import pycountry
import unidecode


class CountryProcessor:
    """Class for processing country names."""

    @staticmethod
    def _normalize(name: str) -> str:
        return unidecode.unidecode(name.lower().strip())

    def _get_pycountry_match(self, name: str) -> Optional[str]:
        """Find the closest match for a country name or ISO code using pycountry."""
        try:
            # Try matching by country name
            country = pycountry.countries.lookup(name)
            # Return the standardized country name
            return country.alpha_3
        except LookupError:
            # Return None if no match is found
            return None

    def get_similar(self, name: str) -> Optional[str]:
        """Get the standardized 3-letter ISO code using pycountry.

        Args:
            name (str): Input country name or ISO code.

        Returns:
            str: Standardized 3-letter ISO code or None if no match is found.
        """
        normalized_name = self._normalize(name)
        # Check pycountry database for a match
        result = self._get_pycountry_match(normalized_name)
        return result


if __name__ == '__main__':
    # Example Usage
    csv_path = 'data/geodata.csv'
    country_processor = CountryProcessor()
    test_country_names = ['RUS', 'USA', 'Uni states', 'Dnemark', 'Danmark', 'Denmarc', 'DNK']
    for country_name in test_country_names:
        result = country_processor.get_similar(country_name)
        if result:
            target_country_name = result
            print(f'{country_name} -> {target_country_name}')
        else:
            print(f'{country_name} -> No match found')
