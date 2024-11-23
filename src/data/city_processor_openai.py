import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


class OpenAIProcessor:
    """A processor for interacting with the OpenAI API to standardize and validate city and geographic information.

    This class provides functionality to:
    - Validate and standardize city names using the OpenAI API.
    - Determine if a city belongs to a larger metropolitan area (agglomeration).
    - Ensure consistent naming conventions in geographic data.

    Attributes:
        client (OpenAI): An instance of the OpenAI client used to communicate with the API.

    Methods:
        get_city(place_name: str, country_tag: str) -> str:
            Validates and standardizes a city name based on its country tag using the OpenAI API.
            Returns the corrected city name if valid, or an empty string if invalid.

        get_agglomeration(place_name: str) -> str:
            Determines if a city belongs to a larger metropolitan area (agglomeration) and returns the name
            of the agglomeration. If the city does not belong to an agglomeration, returns the city name itself.
    """

    def __init__(self, client: OpenAI):
        self.client = client

    def get_city(
        self,
        place_name: str,
        country_tag: str,
    ) -> str:
        """Use OpenAI API to infer the city associated with a given place name and validate against a country tag."""
        prompt = (
            f"Given the place name '{place_name}' and the country tag '{country_tag}', "
            'correct the city name if it is misspelled, and return only the corrected city name. '
            'Ensure that the city is valid and belongs to the specified country. '
            'If the input is incorrect or city does not exist or not is in the country, respond with an empty string. '
            'Do not include any additional information or explanation in your response.'
        )

        try:
            response = self.client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant specialized in geography.',
                    },
                    {
                        'role': 'user',
                        'content': prompt,
                    },
                ],
                temperature=0,
            )
            city = response.choices[0].message.content.strip()
            return city
        except Exception as e:
            return f'An error occurred: {e}'

    def get_agglomeration(
        self,
        place_name: str,
        country_name: str,
        state_name: Optional[str] = None,
    ) -> str:
        """Use OpenAI API to determine if a city belongs to a larger metropolitan area (agglomeration).

        This function takes a city name (`place_name`), along with the country name and optionally the state name,
        and queries the OpenAI API to determine if the city belongs to a larger metropolitan area. If it does,
        the function returns the name of the metropolitan area (with a consistent suffix, such as 'Metropolitan Area').
        If the city does not belong to a metropolitan area, the function returns the city name itself.

        Args:
            place_name (str): The name of the city or place to analyze.
            country_name (str): The name of the country the city is located in.
            state_name (Optional[str]): The name of the state or region the city is located in, if applicable.

        Returns:
            str: The name of the metropolitan area if the city belongs to one, or the city name itself if not.
                 In case of an error, a descriptive error message is returned.
        """
        prompt = (
            f"Given the place name '{place_name}, {(f'{state_name}, ' if state_name else '') + country_name}', "
            f"determine if it belongs to a larger metropolitan area (agglomeration). "
            'If the city is part of an agglomeration, respond only with the name of the agglomeration, '
            "and ensure the response ends with 'Metropolitan Area' for consistency but only for metropolitan areas. "
            'If the city is not part of any agglomeration, respond only with the name of the city itself, do not '
            'include the state or country tags. Do not include any additional information or explanation in your '
            'response.'
        )

        try:
            response = self.client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant specialized in geography.',
                    },
                    {
                        'role': 'user',
                        'content': prompt,
                    },
                ],
                temperature=0,
            )
            agglomeration = response.choices[0].message.content.strip()
            return agglomeration
        except Exception as e:
            return f'An error occurred: {e}'


# Example Usage
if __name__ == '__main__':
    # Example Usage
    place_names = [
        ('New York', 'USA'),
        ('Boston', 'GBR'),
        ('Down Town Dubai', 'ARE'),
        ('keysborough', 'USA'),
        ('Running Springs', 'USA'),
        ('Hollywood', 'USA'),
        ('Woodland Hills', 'USA'),
        ('lakewodo', 'USA'),
        ('Moon Township', 'USA'),
        ('London', 'IND'),
        ('New York', 'ARG'),
        ('nova iorque', 'USA'),
    ]
    client = OpenAI()
    openai_processor = OpenAIProcessor(client)
    for place_name, country in place_names:
        city = openai_processor.get_city(place_name, country)
        print(f'{place_name} -> {city}')

    for place_name, country in place_names:
        agglomeration = openai_processor.get_agglomeration(place_name, country)
        print(f'{place_name} -> {agglomeration}')
