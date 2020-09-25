from typing import Dict
from typing import List

from astrolib.base_objects import Matrix
from astrolib.base_objects import TimeSpan
from astrolib.state_vector import ElementSetBase


class ForceModelBase():
    """ Class represents..."""

    def __init__(self):
        pass

    def compute_derivatives(self, t: TimeSpan, X: ElementSetBase) -> Matrix:
        raise NotImplementedError()

    def compute_partials(self, t: TimeSpan, X: ElementSetBase) -> Matrix:
        raise NotImplementedError()


class DynamicsModel():
    """Class represents..."""

    def __init__(self):
        self.models: Dict[str, List[ForceModelBase]] = dict()

    def evaluate(self) -> Matrix:
        pass