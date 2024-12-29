def safe_file_readlines(path):
    try:
        with open(path, 'r') as f:
            return f.readlines()
    except FileNotFoundError as e:
        raise Exception(f"Error: File not found at '{path}'. \n\nError: {e}")
    except IOError as e:
        raise Exception(f"Error: Unable to read file '{path}': {e}")