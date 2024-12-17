from typing import NewType, Tuple, Any
from collections.abc import Iterable

PositiveInt = NewType('PositiveInt', int)
Coordinate = NewType('Coordinate', tuple[int, int])
Size = NewType('Size', tuple[PositiveInt, PositiveInt])

help(NewType)