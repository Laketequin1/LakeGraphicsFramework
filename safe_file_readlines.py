def safe_file_readlines(file_path: str):
    """
    Reads lines from the file, raising errors if there is a failure.

    Will raise an exception if the file is not found or is not readable (E.G. Insufficient permissions)

    Parameters:
        file_path (str): The path of the file to read.
    """
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: File not found at '{file_path}'. \n\nError: {e}")
    except IOError as e:
        raise IOError(f"Error: Unable to read file '{file_path}': \n\n Error: {e}")