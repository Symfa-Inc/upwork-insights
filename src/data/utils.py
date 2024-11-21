import ast
import os
from pathlib import Path
from typing import List, Union


def get_csv_converters() -> dict:
    """Returns a dictionary of converters for specific columns when reading CSV files.

    The converters are used to process specific columns in the CSV during loading with `pd.read_csv`.
    For example, columns containing list-like strings can be deserialized into Python lists.

    Returns:
        dict: A dictionary where keys are column names and values are functions
              to process the column data during CSV reading.

    Example:
        >>> converters = get_csv_converters()
        >>> df = pd.read_csv("example.csv", converters=converters)
    """
    converters = {
        'SKILLS': ast.literal_eval,
        'TAGS': ast.literal_eval,
        'ADDITIONAL_SKILLS': ast.literal_eval,
    }
    return converters


def get_file_list(
    src_dirs: Union[List[str], str],
    ext_list: Union[List[str], str],
    filename_template: str = '',
) -> List[str]:
    """Get a list of files in the specified directory with specific extensions.

    Args:
        src_dirs: directory(s) with files inside
        ext_list: extension(s) used for a search
        filename_template: include files with this template
    Returns:
        all_files: a list of file paths
    """
    all_files = []
    src_dirs = [src_dirs] if isinstance(src_dirs, str) else src_dirs
    ext_list = [ext_list] if isinstance(ext_list, str) else ext_list
    for src_dir in src_dirs:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                file_ext = Path(file).suffix
                file_ext = file_ext.lower()
                if file_ext in ext_list and filename_template in file:
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
    all_files.sort()
    return all_files


if __name__ == '__main__':
    get_file_list(
        src_dirs='data/raw',
        ext_list='.csv',
    )
