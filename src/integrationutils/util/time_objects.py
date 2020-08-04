from decimal import Decimal

from integrationutils.util.constants import NANOSECONDS_PER_SECOND

_DECIMAL_NANOSECONDS_PER_SECOND = Decimal(str(NANOSECONDS_PER_SECOND))

class TimeSpan:
    """Class represents a time structure supporting nanosecond precision."""

    def __init__(self, whole_seconds: int = 0, nano_seconds: int = 0):
        if whole_seconds < 0.0:
            raise ValueError('Input number of whole seconds must be positive.')
        if nano_seconds < 0.0:
            raise ValueError('Input number of nanoseconds must be positive.')
        while nano_seconds >= NANOSECONDS_PER_SECOND:
            whole_seconds += 1
            nano_seconds -= NANOSECONDS_PER_SECOND
        self.whole_seconds = whole_seconds
        self.nano_seconds = nano_seconds

    def __str__(self):
        return f'[whole_seconds = {self.whole_seconds}, nano_seconds = {self.nano_seconds}]'

    def to_seconds(self) -> float:
        """Returns the calling TimeSpan's value converted to seconds. This conversion could
        potentially not preserve the calling TimeSpan's precision.
        """
        return self.whole_seconds + (self.nano_seconds / NANOSECONDS_PER_SECOND)

    @classmethod
    def from_seconds(cls, seconds: float):
        """Factory method to create a TimeSpan from seconds."""
        if seconds < 0.0:
            raise ValueError('Input number of seconds must be positive.')
        decimal_sec = Decimal(str(seconds))
        return cls(int(decimal_sec), int((decimal_sec % 1) * _DECIMAL_NANOSECONDS_PER_SECOND))



