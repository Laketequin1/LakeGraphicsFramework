from typing import NewType, Tuple, Any, NoReturn
from numbers import Real
from collections.abc import Sequence

PositiveInt = NewType('PositiveInt', int)
Coordinate = NewType('Coordinate', tuple[int, int])
Size = NewType('Size', tuple[PositiveInt, PositiveInt])
AnyString = (str, bytes, bytearray)

class Validate:
    """
    A class containing static methods to validate variable types and values.
    """
    @staticmethod
    def validate_types(expected_types: list[tuple[str, Any, type]]) -> None | NoReturn:
        """
        Validates a list with the type of a variable against the expected type.

        Args:
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
            Validate.validate_type(*expected_type)
        
    @staticmethod
    def validate_type(name: str, var: Any, expected_type: type) -> None | NoReturn:
        """
        Validates the type of a variable against the expected type.

        Args:
            name (str): The name of the variable being validated.
            var (Any): The variable to be validated.
            expected_type (type): The expected type for the variable.

        Returns:
            None: If validation passes, the function returns nothing.
            NoReturn: If the function raises an error, it does not return any value.
        """
        if expected_type is PositiveInt:
            Validate._validate_PositiveInt()
            return
        
        if expected_type is Coordinate:
            Validate._validate_Coordinate()
            return
        
        if expected_type is Size:
            Validate._validate_Size()
            return
        
        if not isinstance(var, expected_type):
            raise TypeError(f"Invalid type for {name}. Expected {expected_type}, got {type(var)}.")

    @staticmethod
    def _validate_PositiveInt(name: str, var: Any) -> None | NoReturn:
        """
        Validates that the variable is a positive integer.

        Args:
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
    def _validate_Coordinate(var: Any) -> None | NoReturn:
        """
        Validates that the variable is a sequence of two numbers representing coordinates.

        Args:
            var (Any): The variable to be validated, expected to be a sequence of two numbers.

        Returns:
            None: If validation passes, the function returns nothing.
            NoReturn: If the function raises an error, it does not return any value.
        """
        if not isinstance(var, Sequence) or isinstance(var, AnyString):
            raise TypeError(f"Invalid type for Coordinate. Expected {Sequence}, got {type(var)}.")
        
        if len(var) != 2:
            raise TypeError(f"Invalid length for Coordinate. Coordinate must be two numbers.")
        
        for i in [0, 1]:
            if not isinstance(var[i], Real):
                raise TypeError(f"Invalid type for Coordinate[{i}]. Expected {Real}, got {type(var[i])}.")
    
    @staticmethod
    def _validate_Size(var: Any) -> None | NoReturn:
        """
        Validates that the variable is a sequence of two positive numbers representing size.

        Args:
            var (Any): The variable to be validated, expected to be a sequence of two real numbers.

        Returns:
            None: If validation passes, the function returns nothing.
            NoReturn: If the function raises an error, it does not return any value.
        """
        if not isinstance(var, Sequence) or isinstance(var, AnyString):
            raise TypeError(f"Invalid type for Size. Expected {Sequence}, got {type(var)}.")
        
        if len(var) != 2:
            raise TypeError(f"Invalid length for Size. Size must be two numbers.")
        
        for i in [0, 1]:
            if not isinstance(var[i], Real):
                raise TypeError(f"Invalid type for Size[{i}]. Expected {Real}, got {type(var[i])}.")
        
        for i in [0, 1]:
            if var[i] < 0:
                raise ValueError(f"Invalid value for Size[{i}]. Size numbers must be positive.")


class Window:
    def __init__(self, size: Size, caption: str = "", fullscreen: bool = False, vsync: bool = True, max_fps: int = 0):
        Validate.validate_types([('size', size, Size),
                                 ('caption', caption, str),
                                 ('fullscreen', fullscreen, bool),
                                 ('vsync', vsync, bool),
                                 ('max_fps', max_fps, int)])
        
        self.size = size
        self.caption = caption
        self.fullscreen = fullscreen
        self.vsync = vsync
        self.max_fps = max_fps

        



    

class GraphicsEngine:
    pass

def main():
    window = Window((2, 5))

if __name__ == "__main__":
    main()