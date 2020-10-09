from typing import Dict
from typing import List
from typing import Tuple
from typing import Type
from typing import Union

from astrolib.base_objects import Matrix
from astrolib.base_objects import TimeSpan
from astrolib.state_vector import ElementSetBase
from astrolib.state_vector import StateVector


class ForceModelBase():
    """ Class represents..."""

    def __init__(self, required_element_types: Union[Type, Tuple[Type,...]]):
        self._required_element_types = required_element_types

    def compute_derivatives(self, t: TimeSpan, X: Union[ElementSetBase, List[ElementSetBase]]) -> Matrix:
        raise NotImplementedError()

    def compute_partials(self, t: TimeSpan, X: ElementSetBase) -> Matrix:
        raise NotImplementedError()


class DynamicsModel():
    """Class represents..."""

    def __init__(self):
        self._force_models: Dict[str, List[ForceModelBase]] = dict()

    def evaluate(self, state: StateVector) -> Matrix:
        
        for force_model in self._force_models:
            