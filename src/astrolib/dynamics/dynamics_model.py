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

    def __init__(self):
        pass


class AccelerationModel(ForceModelBase):

    def __init__(self):
        super().__init__()

    def compute_acceleration(self, t: TimeSpan, X: ElementSetBase) -> Vec3d:
        raise NotImplementedError()

    def compute_partials(self, t: TimeSpan, X: ElementSetBase) -> Matrix:
        raise NotImplementedError()


class TorqueModel(ForceModelBase):

    def __init__(self):
        super().__init__()

    def compute_torque(self, t: TimeSpan, X: ElementSetBase) -> Matrix:
        raise NotImplementedError()

    def compute_partials(self, t: TimeSpan, X: ElementSetBase) -> Matrix:
        raise NotImplementedError()


class DynamicsModel():
    """Class represents..."""

    def __init__(self):
        self._force_models: Dict[str, List[ForceModelBase]] = dict()

    def evaluate(self, state: StateVector) -> Matrix:
        accel_models = [x for x in self._force_models if isinstance(x, AccelerationModel)]
        torque_models = [x for x in self._force_models if isinstance(x, TorqueModel)]
        derivatives = []
        for element_set in state.elements:
            