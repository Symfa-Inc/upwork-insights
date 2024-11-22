import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


def get_city(
    client: OpenAI,
    place_name: str,
) -> str:
    """Use OpenAI API to infer the city associated with a given place name."""
    prompt = (
        f"Given the place name '{place_name}', respond only with the name of the city it belongs to. "
        'Do not include any additional information or explanation.'
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
        'New York',
        'Boston',
        'Down Town Dubai',
        'keysborough',
        'Hollywood',
        'Woodland Hills',
    ]
    client = OpenAI(api_key=openai_api_key)
    for place_name in place_names:
        city = get_city(client, place_name)
        print(f'{place_name} -> {city}')
