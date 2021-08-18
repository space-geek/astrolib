""" TODO: Module docstring
"""
from typing import Callable
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Type

from astrolib import Matrix
from astrolib import TimeSpan
from astrolib.dynamics.dynamics_model import DynamicsModel
from astrolib.integration.euler import integrate as integrate_euler
from astrolib.integration.rk4 import integrate as integrate_rk4
from astrolib.integration.rk45 import integrate as integrate_rk45
from astrolib.state_vector import StateVector
from astrolib.util.constants import EARTH_MU


class PropagatorBase():
    """Base class for all propagators."""

    def __init__(self):
        pass

    def get_state(self, epoch: TimeSpan, *args, **kwargs) -> StateVector:
        raise NotImplementedError

    def get_states(self, epochs: List[TimeSpan], *args, **kwargs) -> 'Ephemeris':
        raise NotImplementedError


class Ephemeris(PropagatorBase):

    @classmethod
    def empty(cls) -> 'Ephemeris':
        """ Factory method to create an empty Ephemeris.
        """
        return Ephemeris([])

    def __init__(self, states: List[StateVector]):
        super().__init__()
        self._states = {x.epoch : x for x in states}

    def __str__(self) -> str:
        return f"[Ephemeris with:\n\tStart Epoch:\t\t{self.start_epoch.to_seconds()} seconds" \
               f"\n\tEnd Epoch:\t\t{self.end_epoch.to_seconds()} seconds" \
               f"\n\tDuration:\t\t{self.duration.to_seconds()} seconds" \
               f"\n\tNumber of States:\t{self.num_states}]"

    def __iter__(self) -> Iterator[StateVector]:
        for state in sorted(self._states.values(), key=lambda x: x.epoch):
            yield state

    @property
    def num_states(self) -> int:
        return len(self._states)

    @property
    def start_epoch(self) -> TimeSpan:
        return min(self._states.keys()) if self._states else TimeSpan.undefined()

    @property
    def end_epoch(self) -> TimeSpan:
        return max(self._states.keys()) if self._states else TimeSpan.undefined()

    @property
    def duration(self) -> TimeSpan:
        return (self.end_epoch - self.start_epoch) \
            if (self.start_epoch.is_defined() and self.end_epoch.is_defined()) else TimeSpan.zero()

    @property
    def epochs(self) -> Iterator[TimeSpan]:
        for state in self:
            yield state.epoch

    def add_state(self, state: StateVector):
        self._states[state.epoch] = state

    def add_states(self, states: List[StateVector]):
        for state in states:
            self.add_state(state)

    def clear(self):
        self._states.clear()

    #pylint: disable=arguments-differ
    def get_state(self, epoch: TimeSpan) -> StateVector:
        state = None
        if epoch in self._states:
            state = self._states[epoch]
        else:
            #TODO Implement interpolation of state based on time
            raise NotImplementedError
        return state

    def get_states(self, epochs: List[TimeSpan]) -> 'Ephemeris':
        return Ephemeris(states=[self.get_state(epoch) for epoch in epochs])


class Integrator(PropagatorBase):

    def __init__(self):
        super().__init__()
        self.dynamics_model = DynamicsModel()

    #pylint: disable=arguments-differ
    def get_state(self, epoch: TimeSpan, X_0: StateVector, *args, **kwargs) -> StateVector:
        raise NotImplementedError

    #pylint: disable=arguments-differ
    def get_states(self, epochs: List[TimeSpan], X_0: StateVector, *args, **kwargs) -> Ephemeris:
        sorted_epochs = sorted(epochs)
        states = [X_0]
        for epoch in sorted_epochs:
            states.append(self.get_state(epoch, states[-1]))
        return Ephemeris(states=states)

    def propagate_to(self, epoch: TimeSpan, X_0: StateVector) -> StateVector:
        return self.get_state(epoch, X_0)

    def _evaluate_dynamics(self, epoch: TimeSpan, X: Matrix, state_vector_type: Type) -> Matrix:
        return self.dynamics_model.evaluate(state_vector_type.from_column_matrix(epoch, X))


class FixedStepIntegrator(Integrator):

    def __init__(self,
                 integration_func: Callable[[TimeSpan, Matrix, TimeSpan, Callable[[TimeSpan, Matrix], Matrix]],
                                            Tuple[TimeSpan, Matrix, TimeSpan]],
                 step_size: TimeSpan):
        super().__init__()
        self._integration_func = integration_func
        self.step_size = step_size

    #pylint: disable=arguments-differ
    def get_state(self, epoch: TimeSpan, X_0: StateVector) -> StateVector:
        self._check_time_step_validity(X_0.epoch, epoch)
        state_vector_type = type(X_0)
        t_n = X_0.epoch
        x_n = X_0.to_column_matrix()
        while t_n != epoch:
            t_n, x_n, _ = self._integration_func(t_n, x_n, self.step_size,
                                                 lambda t, x: self._evaluate_dynamics(t, x, state_vector_type))
        return state_vector_type.from_column_matrix(epoch, x_n)

    #pylint: disable=arguments-differ
    #pylint: disable=useless-super-delegation
    def get_states(self, epochs: List[TimeSpan], X_0: StateVector) -> Ephemeris:
        return super().get_states(epochs, X_0)

    def _check_time_step_validity(self, t_0: TimeSpan, t_f: TimeSpan) -> bool:
        if not ((t_f - t_0).to_seconds() % self.step_size.to_seconds()) == 0.0:
            raise ValueError("Unable to propagate to desired epoch using fixed-step " \
                             "integrator with selected step size.")


class Euler(FixedStepIntegrator):

    def __init__(self, step_size: TimeSpan):
        super().__init__(integrate_euler, step_size)


class RK4(FixedStepIntegrator):

    def __init__(self, step_size: TimeSpan):
        super().__init__(integrate_rk4, step_size)


class VariableStepIntegrator(Integrator):

    def __init__(self,
                 integration_func: Callable[[TimeSpan, Matrix, TimeSpan, Callable[[TimeSpan, Matrix], Matrix]],
                                            Tuple[TimeSpan, Matrix, TimeSpan]],
                 relative_error_tolerance: float = 1.0e-12
                 ):
        super().__init__()
        self._integration_func = integration_func
        self.rel_tol = relative_error_tolerance

    #pylint: disable=arguments-differ
    def get_state(self, epoch: TimeSpan, X_0: StateVector) -> StateVector:
        state_vector_type = type(X_0)
        t_n = X_0.epoch
        x_n = X_0.to_column_matrix()
        while t_n != epoch:
            step_size = epoch - t_n
            t_n, x_n, _ = self._integration_func(t_n, x_n, step_size,
                                                 lambda t, x: self._evaluate_dynamics(t, x, state_vector_type),
                                                 rel_tol=self.rel_tol)
        return state_vector_type.from_column_matrix(epoch, x_n)

    #pylint: disable=arguments-differ
    #pylint: disable=useless-super-delegation
    def get_states(self, epochs: List[TimeSpan], X_0: StateVector) -> Ephemeris:
        return super().get_states(epochs, X_0)


class RK45(VariableStepIntegrator):

    def __init__(self):
        super().__init__(integrate_rk45)


class TwoBody(PropagatorBase):

    def __init__(self, mu: float = EARTH_MU):
        super().__init__()
        self._mu = mu

    #pylint: disable=arguments-differ
    def get_state(self, epoch: TimeSpan, X_0: StateVector) -> StateVector:
        raise NotImplementedError

    #pylint: disable=arguments-differ
    def get_states(self, epochs: List[TimeSpan], X_0: StateVector):# -> Ephemeris:
        sorted_epochs = sorted(epochs, lambda x: x.epoch)
        states = [X_0]
        for epoch in sorted_epochs:
            states.append(self.get_state(epoch, states[-1]))
        return Ephemeris(states=states)
