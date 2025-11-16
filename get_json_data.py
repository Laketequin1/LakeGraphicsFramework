import json
from typing import Any

def read_json_data(file_name: str) -> Any:
    """
    Reads JSON data from the file, then returns the data as a Python data structure.

    Parameters:
        file_name (str): The name of the JSON file. E.G. "json_file.json".

    Returns:
        data (Any): The read data converted to a Python data structure.
    """
    with open(str(file_name), "r") as file:
        data = json.load(file)
        return data
    
def write_json_data(file_name: str, data_struct: Any) -> None:
    """
    Formats and saves a python data structure to the JSON file.

    Parameters:
        file_name (str): The name of the JSON file. E.G. "json_file.json".
        data_struct (Any): The python data structure to be saved to the file.
    """
    with open(str(file_name), "w") as file:
        json.dump(data_struct, file, indent=4)