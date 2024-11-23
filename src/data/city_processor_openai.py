import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


def get_city(
    client: OpenAI,
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
        response = client.chat.completions.create(
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
    client = OpenAI(api_key=openai_api_key)
    for place_name, country in place_names:
        city = get_city(client, place_name, country)
        print(f'{place_name} -> {city}')
