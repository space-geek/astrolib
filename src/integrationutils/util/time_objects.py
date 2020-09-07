from decimal import Decimal
from math import copysign
from math import floor
from typing import Tuple

from integrationutils.util.constants import NANOSECONDS_PER_SECOND


_DECIMAL_NANOSECONDS_PER_SECOND = Decimal(str(NANOSECONDS_PER_SECOND))

class TimeSpan:
    """Class represents a time structure supporting nanosecond precision."""

    @classmethod
    def undefined(cls):
        """Factor method to create an undefined TimeSpan."""
        return cls(None,None)

    @classmethod
    def zero(cls):
        """Factor method to create a zero TimeSpan."""
        return cls(0,0)

    @classmethod
    def from_seconds(cls, seconds: float):
        """Factory method to create a TimeSpan from seconds."""
        return cls(*_decompose_decimal_seconds(seconds))

    def __init__(self, whole_seconds: int, nano_seconds: int):
        def normalize_time(ws: int, ns: int) -> Tuple[int,int]:
            """Function for normalizing whole vs sub-second digits."""
            ws += (copysign(1,ns) * 1)
            ns -= (copysign(1,ns) * NANOSECONDS_PER_SECOND)
            return ws, ns
        self.whole_seconds = None
        self.nano_seconds = None
        if (whole_seconds is not None) and (nano_seconds is not None):
            while abs(nano_seconds) >= NANOSECONDS_PER_SECOND:
                whole_seconds, nano_seconds = normalize_time(whole_seconds, nano_seconds)
            if copysign(1,whole_seconds) != copysign(1,nano_seconds):
                whole_seconds, nano_seconds = normalize_time(whole_seconds, nano_seconds)
            self.whole_seconds = int(whole_seconds)
            self.nano_seconds = int(nano_seconds)

    def __str__(self):
        return f'[whole_seconds = {self.whole_seconds}, nano_seconds = {self.nano_seconds}]'

    def __eq__(self, other) -> bool:
        if not isinstance(other, TimeSpan):
            return False
        if self.whole_seconds != other.whole_seconds:
            return False
        if self.nano_seconds != other.nano_seconds:
            return False
        return True

    def __lt__(self, other) -> bool:
        if not isinstance(other, TimeSpan):
            return False
        if self.whole_seconds > other.whole_seconds:
            return False
        if self.whole_seconds == other.whole_seconds:
            if self.nano_seconds >= other.nano_seconds:
                return False
        return True

    def __le__(self, other) -> bool:
        if not isinstance(other, TimeSpan):
            return False
        if self.whole_seconds > other.whole_seconds:
            return False
        if self.whole_seconds == other.whole_seconds:
            if self.nano_seconds > other.nano_seconds:
                return False
        return True

    def __add__(self, other):
        if not isinstance(other, TimeSpan):
            return NotImplemented
        return TimeSpan(self.whole_seconds + other.whole_seconds,
                        self.nano_seconds + other.nano_seconds)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, TimeSpan):
            return NotImplemented
        return TimeSpan(self.whole_seconds - other.whole_seconds,
                        self.nano_seconds - other.nano_seconds)

    def __rsub__(self, other):
        return -1.0 * self.__sub__(other)

    def __mul__(self, other):
        if not isinstance(other, (float, int)):
            return NotImplemented
        ws, ns = _decompose_decimal_seconds(other * self.whole_seconds)
        return TimeSpan(ws, floor(other * self.nano_seconds) + ns)

    def __rmul__(self, other):
        if not isinstance(other, (float, int)):
            return NotImplemented
        return self.__mul__(other)

    def __abs__(self):
        return TimeSpan(abs(self.whole_seconds), abs(self.nano_seconds))

    def to_seconds(self) -> float:
        """Returns the calling TimeSpan's value converted to seconds. This conversion could
        potentially not preserve the calling TimeSpan's precision.
        """
        return self.whole_seconds + (self.nano_seconds / NANOSECONDS_PER_SECOND)

def _decompose_decimal_seconds(seconds: float) -> Tuple[int, int]:
    decimal_sec = Decimal(str(seconds))
    return int(decimal_sec), int((decimal_sec % 1) * _DECIMAL_NANOSECONDS_PER_SECOND)
