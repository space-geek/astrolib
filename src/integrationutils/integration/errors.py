

class NumericalIntegrationError(Exception):
    """ Base exception type for errors encountered in numerical integration."""

class MinimumStepSizeExceededError(NumericalIntegrationError):
    """ Exception raised when integration step exceeds minimum step size.

    Attributes:
        h (float)       Attempted integration step size, in seconds.
        h_min (float)   Minimum integration step size, in seconds.

    """

    def __init__(self, h_sec: float, h_min_sec: float):
        self.message = f"Minimum integration step size exceeded. A {h_sec} second step was attempted with a minimum integration step size of {h_min_sec} seconds."
        self.h = h_sec
        self.h_min = h_min_sec
        super().__init__(self.message)
