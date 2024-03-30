""" TODO: Module docstring
"""

from decimal import Decimal
import math
from typing import List
from typing import Self
from typing import Tuple
from typing import Union

from astrolib.constants import NANOSECONDS_PER_SECOND
from astrolib.constants import SECONDS_PER_HOUR
from astrolib.constants import SECONDS_PER_MINUTE
from astrolib.constants import SECONDS_PER_SOLAR_DAY
from astrolib.matrix import Matrix

_DECIMAL_NANOSECONDS_PER_SECOND = Decimal(str(NANOSECONDS_PER_SECOND))


class TimeSpan:
    """Class represents a time structure supporting nanosecond precision."""

    @staticmethod
    def undefined() -> Self:
        """Factory method to create an undefined TimeSpan."""
        return TimeSpan(None, None)

    @staticmethod
    def zero() -> Self:
        """Factory method to create a zero TimeSpan."""
        return TimeSpan(0, 0)

    @staticmethod
    def from_seconds(seconds: float) -> Self:
        """Factory method to create a TimeSpan from a number of seconds."""
        return TimeSpan(*_decompose_decimal_seconds(seconds))

    @staticmethod
    def from_minutes(minutes: float) -> Self:
        """Factory method to create a TimeSpan from a number of minutes."""
        return TimeSpan(*_decompose_decimal_seconds(minutes * SECONDS_PER_MINUTE))

    @staticmethod
    def from_hours(minutes: float) -> Self:
        """Factory method to create a TimeSpan from a number of hours."""
        return TimeSpan(*_decompose_decimal_seconds(minutes * SECONDS_PER_HOUR))

    @staticmethod
    def from_days(days: float) -> Self:
        """Factory method to create a TimeSpan from a number of mean solar days."""
        return TimeSpan(*_decompose_decimal_seconds(days * SECONDS_PER_SOLAR_DAY))

    def __init__(self, whole_seconds: int, nano_seconds: int) -> None:
        def normalize_time(ws: int, ns: int) -> Tuple[int, int]:
            """Function for normalizing whole vs sub-second digits."""
            ws += math.copysign(1, ns) * 1
            ns -= math.copysign(1, ns) * NANOSECONDS_PER_SECOND
            return ws, ns

        self._whole_seconds = None
        self._nano_seconds = None
        if (whole_seconds is not None) and (nano_seconds is not None):
            while abs(nano_seconds) >= NANOSECONDS_PER_SECOND:
                whole_seconds, nano_seconds = normalize_time(
                    whole_seconds, nano_seconds
                )
            if math.copysign(1, whole_seconds) != math.copysign(1, nano_seconds):
                whole_seconds, nano_seconds = normalize_time(
                    whole_seconds, nano_seconds
                )
            self._whole_seconds = int(whole_seconds)
            self._nano_seconds = int(nano_seconds)

    def __str__(self) -> str:
        return (
            f"[whole_seconds = {self._whole_seconds}, nano_seconds = {self._nano_seconds}]"
            if self.is_defined()
            else "Undefined"
        )

    def __repr__(self) -> str:
        return f"[{self._whole_seconds}, {self._nano_seconds}"

    def __hash__(self) -> int:
        return hash((self._whole_seconds, self._nano_seconds))

    def __eq__(self, other: Self) -> bool:
        if not isinstance(other, TimeSpan):
            return False
        if self._whole_seconds != other._whole_seconds:
            return False
        if self._nano_seconds != other._nano_seconds:
            return False
        return True

    def __lt__(self, other: Self) -> bool:
        if not isinstance(other, TimeSpan):
            return False
        if self._whole_seconds > other._whole_seconds:
            return False
        if self._whole_seconds == other._whole_seconds:
            if self._nano_seconds >= other._nano_seconds:
                return False
        return True

    def __le__(self, other: Self) -> bool:
        if not isinstance(other, TimeSpan):
            return False
        if self._whole_seconds > other._whole_seconds:
            return False
        if self._whole_seconds == other._whole_seconds:
            if self._nano_seconds > other._nano_seconds:
                return False
        return True

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, TimeSpan):
            return NotImplemented
        return TimeSpan(
            self._whole_seconds + other._whole_seconds,
            self._nano_seconds + other._nano_seconds,
        )

    def __radd__(self, other: Self) -> Self:
        return self.__add__(other)

    def __sub__(self, other: Self) -> Self:
        if not isinstance(other, TimeSpan):
            return NotImplemented
        return TimeSpan(
            self._whole_seconds - other._whole_seconds,
            self._nano_seconds - other._nano_seconds,
        )

    def __rsub__(self, other: Self) -> Self:
        return -1.0 * self.__sub__(other)

    def __mul__(self, other: Union[float, int]) -> Self:
        if not isinstance(other, (float, int)):
            return NotImplemented
        ws, ns = _decompose_decimal_seconds(other * self._whole_seconds)
        return TimeSpan(ws, math.floor(other * self._nano_seconds) + ns)

    def __rmul__(self, other: Union[float, int]) -> Self:
        if not isinstance(other, (float, int)):
            return NotImplemented
        return self.__mul__(other)

    def __abs__(self) -> Self:
        ws = abs(self._whole_seconds) if self._whole_seconds is not None else None
        ns = abs(self._nano_seconds) if self._nano_seconds is not None else None
        return TimeSpan(ws, ns)

    def __neg__(self) -> Self:
        ws = -1 * self._whole_seconds if self._whole_seconds is not None else None
        ns = -1 * self._nano_seconds if self._nano_seconds is not None else None
        return TimeSpan(ws, ns)

    def is_defined(self) -> bool:
        """Returns a boolean indicator of whether or not the calling TimeSpan is defined.

        Returns:
            bool: Boolean indicator of whether or not the calling TimeSpan is defined.
        """
        return (self._whole_seconds is not None) and (self._nano_seconds is not None)

    def to_seconds(self) -> float:
        """Returns the calling TimeSpan's value converted to seconds. This conversion could
        potentially not preserve the calling TimeSpan's precision.
        """
        return self._whole_seconds + (self._nano_seconds / NANOSECONDS_PER_SECOND)

    def to_minutes(self) -> float:
        """Returns the calling TimeSpan's value converted to minutes. This conversion could
        potentially not preserve the calling TimeSpan's precision.
        """
        return self.to_seconds() / SECONDS_PER_MINUTE

    def to_hours(self) -> float:
        """Returns the calling TimeSpan's value converted to hours. This conversion could
        potentially not preserve the calling TimeSpan's precision.
        """
        return self.to_seconds() / SECONDS_PER_HOUR

    def to_days(self) -> float:
        """Returns the calling TimeSpan's value converted to mean solar days. This conversion could
        potentially not preserve the calling TimeSpan's precision.
        """
        return self.to_seconds() / SECONDS_PER_SOLAR_DAY


class ElementSetBase:
    """Class represents a set of generic state vector elements, e.g. a set of Keplerian orbital
    elements, a set of Cartesian orbital elements, a set of Euler angles and the corresponding
    sequence, etc.
    """

    def __init__(self, elements: List[Matrix]) -> None:
        self._elements = Matrix.from_column_matrices(elements)

    @property
    def num_elements(self) -> int:
        """TODO: Property docstring"""
        return self._elements.num_rows

    def to_column_matrix(self) -> Matrix:
        """TODO: Method docstring"""
        return self._elements

    def from_column_matrix(self, value: Matrix) -> "ElementSetBase":
        """TODO: Method docstring"""
        if not isinstance(value, Matrix) or value.num_cols != 1:
            raise ValueError("Input value must be a column matrix.")
        if value.num_rows != self.num_elements:
            raise ValueError(
                f"Input column matrix must have {self.num_elements} elements."
            )
        self._elements = value


def _decompose_decimal_seconds(seconds: float) -> Tuple[int, int]:
    """TODO: Function docstring"""
    decimal_sec = Decimal(str(seconds))
    return int(decimal_sec), int((decimal_sec % 1) * _DECIMAL_NANOSECONDS_PER_SECOND)
