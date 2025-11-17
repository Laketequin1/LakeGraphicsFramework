"""
Validates variable types for the project GraphicsFramework.

Used to validate user parameters being passed into GraphicsFramework functions.
"""

### Imports ###
from typing import NewType, Tuple, Any, NoReturn
from numbers import Real
from collections.abc import Sequence


### Type Hints ###
PositiveInt = NewType('PositiveInt', int)
Coordinate = NewType('Coordinate', tuple[int, int])
Size = NewType('Size', tuple[PositiveInt, PositiveInt])
AnyString = (str, bytes, bytearray)
ColorRGBA = Tuple[float, float, float, float]


### Functions ###
def validate_types(expected_types: list[tuple[str, Any, type]]) -> None | NoReturn:
    """
    Validates a list with the type of a variable against the expected type. The function will raise an error if valitadion statement is invalid, otherwise continues.

    Parameters:
        expected_types (list):
            expected_type (tuple):
                name (str): The name of the variable being validated.
                var (Any): The variable to be validated.
                expected_type (type): The expected type for the variable.        
    """
    for expected_type in expected_types:
        validate_type(*expected_type)
    # Validation success
    
def validate_type(name: str, var: Any, expected_type: type) -> None | NoReturn:
    """
    Validates the type of a variable against the expected type. The function will raise an error if valitadion statement is invalid, otherwise continues.

    Parameters:
        name (str): The name of the variable being validated.
        var (Any): The variable to be validated.
        expected_type (type): The expected type for the variable.
    """
    if expected_type is PositiveInt:
        _validate_positiveint(name, var)
        return # Validation success
    
    if expected_type is Coordinate:
        _validate_coordinate(name, var)
        return # Validation success
    
    if expected_type is Size:
        _validate_size(name, var)
        return # Validation success
    
    if expected_type is ColorRGBA:
        _validate_color_rgba(name, var)
        return # Validation success
    
    if not isinstance(var, expected_type):
        raise TypeError(f"Invalid type for {name}. Expected {expected_type}, got {type(var)}.")
    # Validation success

def _validate_positiveint(name: str, var: Any) -> None | NoReturn:
    """
    [Private]
    Validates that the variable is a positive integer. The function will raise an error if valitadion statement is invalid, otherwise continues.

    Parameters:
        name (str): The name of the variable being validated.
        var (Any): The variable to be validated.
    """
    if not isinstance(var, Real):
        raise TypeError(f"Invalid type for {name}. Expected {Real}, got {type(var)}.")

    if var < 0:
        raise ValueError(f"Invalid value for {name}. Size numbers must be positive.")   

def _validate_coordinate(name: str, var: Any) -> None | NoReturn:
    """
    [Private]
    Validates that the variable is a sequence of two numbers representing coordinates. The function will raise an error if valitadion statement is invalid, otherwise continues.

    Parameters:
        name (str): The name of the variable being validated.
        var (Any): The variable to be validated, expected to be a sequence of two numbers.
    """
    if not isinstance(var, Sequence) or isinstance(var, AnyString):
        raise TypeError(f"Invalid type for Coordinate '{name}'. Expected {Sequence}, got {type(var)}.")
    
    if len(var) != 2: # type: ignore
        raise TypeError(f"Invalid length for Coordinate '{name}'. Coordinate must be two numbers.")
    
    for i in [0, 1]:
        if not isinstance(var[i], Real):
            raise TypeError(f"Invalid type for Coordinate[{i}] '{name}'. Expected {Real}, got {type(var[i])}.") # type: ignore

def _validate_size(name: str, var: Any) -> None | NoReturn:
    """
    [Private]
    Validates that the variable is a sequence of two positive numbers representing size. The function will raise an error if valitadion statement is invalid, otherwise continues.

    Parameters:
        name (str): The name of the variable being validated.
        var (Any): The variable to be validated, expected to be a sequence of two real numbers.
    """
    if not isinstance(var, Sequence) or isinstance(var, AnyString):
        raise TypeError(f"Invalid type for Size '{name}'. Expected {Sequence}, got {type(var)}.")
    
    if len(var) != 2: # type: ignore
        raise TypeError(f"Invalid length for Size '{name}'. Size must be two numbers.")
    
    for i in [0, 1]:
        if not isinstance(var[i], Real):
            raise TypeError(f"Invalid type for Size[{i}] '{name}'. Expected {Real}, got {type(var[i])}.") # type: ignore
    
    for i in [0, 1]:
        if var[i] < 0:
            raise ValueError(f"Invalid value for Size[{i}] '{name}'. Size numbers must be positive, but the value was {var[i]}.")
        
def _validate_color_rgba(name: str, var: Any) -> None | NoReturn:
    """
    [Private]
    Validates that the variable is a sequence of 4 positive floats ranging from 0 to 1 representing nomalised rgba: Red, Green, Blue, Alpha. The function will raise an error if valitadion statement is invalid, otherwise continues.

    Parameters:
        name (str): The name of the variable being validated.
        var (Any): The variable to be validated, expected to be a sequence of 4 real positive floats.
    """
    if not isinstance(var, Sequence) or isinstance(var, AnyString):
        raise TypeError(f"Invalid type for the Color '{name}'. Expected {Sequence}, got {type(var)}.")
    
    if len(var) != 4: # type: ignore
        raise TypeError(f"Invalid length for the Color '{name}'. Color must be contain four channels: Red, Green, Blue, Alpha.")
    
    for i in range(4):
        if not isinstance(var[i], Real):
            raise TypeError(f"Invalid type for the Color[{i}] '{name}'. Expected {Real}, got {type(var[i])}.") # type: ignore
    
    for i in range(4):
        if var[i] < 0 or var[i] > 1:
            raise ValueError(f"Invalid value for the Color[{i}] '{name}'. Color numbers must be between 0 and 1 inclusive, but the value was {var[i]}.")
        

### Example code ###
def main():
    size = (0, 0)
    caption = "Title"
    max_fps = 100
    print_gl_errors = True
    validate_types([('size', size, Size), # type: ignore
                    ('caption', caption, str),
                    ('max_fps', max_fps, int),
                    ('print_gl_errors', print_gl_errors, bool)]) # OK
    
    vsync = False
    validate_type('vsync', vsync, bool) # OK
    
    size = (0, -20) # Set to an invalid size
    validate_types([('size', size, Size), # type: ignore
                    ('caption', caption, str),
                    ('max_fps', max_fps, int),
                    ('print_gl_errors', print_gl_errors, bool)]) # Fails type check, raises error

# Entry point
if __name__ == "__main__":
    main()