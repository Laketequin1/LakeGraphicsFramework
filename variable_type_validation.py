"""
Validates variable types for the project GraphicsFramework.

Used to validate user parameters being passed into the GraphicsFramework functions.
"""

from typing import NewType, Tuple, Any, NoReturn
from numbers import Real
from collections.abc import Sequence

PositiveInt = NewType('PositiveInt', int)
Coordinate = NewType('Coordinate', tuple[int, int])
Size = NewType('Size', tuple[PositiveInt, PositiveInt])
AnyString = (str, bytes, bytearray)

@staticmethod
def validate_types(expected_types: list[tuple[str, Any, type]]) -> None | NoReturn:
    """
    Validates a list with the type of a variable against the expected type.

    Parameters:
        expected_types (list):
            expected_type (tuple):
                name (str): The name of the variable being validated.
                var (Any): The variable to be validated.
                expected_type (type): The expected type for the variable.

    Returns:
        None: If validation passes, the function returns nothing.
        NoReturn: If the function raises an error, it does not return any value.
    """
    for expected_type in expected_types:
        validate_type(*expected_type)
    # Validation success
    
@staticmethod
def validate_type(name: str, var: Any, expected_type: type) -> None | NoReturn:
    """
    Validates the type of a variable against the expected type.

    Parameters:
        name (str): The name of the variable being validated.
        var (Any): The variable to be validated.
        expected_type (type): The expected type for the variable.

    Returns:
        None: If validation passes, the function returns nothing.
        NoReturn: If the function raises an error, it does not return any value.
    """
    if expected_type is PositiveInt:
        _validate_PositiveInt(name, var)
        return # Validation success
    
    if expected_type is Coordinate:
        _validate_Coordinate(name, var)
        return # Validation success
    
    if expected_type is Size:
        _validate_Size(name, var)
        return # Validation success
    
    if not isinstance(var, expected_type):
        raise TypeError(f"Invalid type for {name}. Expected {expected_type}, got {type(var)}.")
    # Validation success

@staticmethod
def _validate_PositiveInt(name: str, var: Any) -> None | NoReturn:
    """
    [Private]
    Validates that the variable is a positive integer.

    Parameters:
        name (str): The name of the variable being validated.
        var (Any): The variable to be validated.

    Returns:
        None: If validation passes, the function returns nothing.
        NoReturn: If the function raises an error, it does not return any value.
    """
    if not isinstance(var, Real):
        raise TypeError(f"Invalid type for {name}. Expected {Real}, got {type(var)}.")

    if var < 0:
        raise ValueError(f"Invalid value for {name}. Size numbers must be positive.")   

@staticmethod
def _validate_Coordinate(name: str, var: Any) -> None | NoReturn:
    """
    [Private]
    Validates that the variable is a sequence of two numbers representing coordinates.

    Parameters:
        name (str): The name of the variable being validated.
        var (Any): The variable to be validated, expected to be a sequence of two numbers.

    Returns:
        None: If validation passes, the function returns nothing.
        NoReturn: If the function raises an error, it does not return any value.
    """
    if not isinstance(var, Sequence) or isinstance(var, AnyString):
        raise TypeError(f"Invalid type for Coordinate '{name}'. Expected {Sequence}, got {type(var)}.")
    
    if len(var) != 2:
        raise TypeError(f"Invalid length for Coordinate '{name}'. Coordinate must be two numbers.")
    
    for i in [0, 1]:
        if not isinstance(var[i], Real):
            raise TypeError(f"Invalid type for Coordinate[{i}] '{name}'. Expected {Real}, got {type(var[i])}.")

@staticmethod
def _validate_Size(name: str, var: Any) -> None | NoReturn:
    """
    [Private]
    Validates that the variable is a sequence of two positive numbers representing size.

    Parameters:
        name (str): The name of the variable being validated.
        var (Any): The variable to be validated, expected to be a sequence of two real numbers.

    Returns:
        None: If validation passes, the function returns nothing.
        NoReturn: If the function raises an error, it does not return any value.
    """
    if not isinstance(var, Sequence) or isinstance(var, AnyString):
        raise TypeError(f"Invalid type for Size '{name}'. Expected {Sequence}, got {type(var)}.")
    
    if len(var) != 2:
        raise TypeError(f"Invalid length for Size '{name}'. Size must be two numbers.")
    
    for i in [0, 1]:
        if not isinstance(var[i], Real):
            raise TypeError(f"Invalid type for Size[{i}] '{name}'. Expected {Real}, got {type(var[i])}.")
    
    for i in [0, 1]:
        if var[i] < 0:
            raise ValueError(f"Invalid value for Size[{i}] '{name}'. Size numbers must be positive.")