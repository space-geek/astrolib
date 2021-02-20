""" TODO: Module docstring
"""
from typing import List
from typing import Set
from typing import Type

from astrolib import Matrix
from astrolib.state_vector import StateVector


class ForceModelBase():
    """ TODO: Class docstring
    """

    supported_state_vector_types: Set[Type] = None

    def compute_acceleration(self, state: StateVector) -> Matrix:
        raise NotImplementedError

    def compute_partials(self, state: StateVector) -> Matrix:
        raise NotImplementedError


class DynamicsModel():
    """ TODO: Class docstring
    """

    def __init__(self, force_models: List[ForceModelBase] = list()):
        self.force_models = force_models

    def evaluate(self, state: StateVector) -> Matrix:
        forces = [x for x in self.force_models if type(state) in x.supported_state_vector_types]
        accel = Matrix.zeros(3,1)
        for force in forces:
            accel += force.compute_acceleration(state)
        return Matrix.from_column_matrices([state.rates, accel])
            