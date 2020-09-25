from typing import List

from astrolib.base_objects import TimeSpan
from astrolib.state_vector import StateVector


class PropagatorBase():
    """Base class for all propagators."""

    def __init__(self):
        pass

    def get_state(self, epoch: TimeSpan, X_0: StateVector) -> StateVector:
        raise NotImplementedError()

    def get_states(self, epochs: List[TimeSpan], X_0: StateVector):# -> Ephemeris:
        raise NotImplementedError()


class Ephemeris(PropagatorBase):
    
    def __init__(self, states: List[StateVector] = []):
        super().__init__()
        self._states = {x.epoch : x for x in states}
    
    def __str__(self) -> str:
        return f"to be implemented"

    def __iter__(self) -> StateVector:
        for state in sorted(self._states.values(), lambda x: x.epoch):
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
        return (self.end_epoch - self.start_epoch) if (self.start_epoch.is_defined() and self.end_epoch.is_defined()) else TimeSpan.zero()

    def add_state(self, state: StateVector):
        self._states[state.epoch] = state

    def add_states(self, states: List[StateVector]):
        for state in states:
            self.add_state(state)

    def clear(self):
        self._states.clear()

    def get_state(self, epoch: TimeSpan) -> StateVector:
        if epoch in self._states:
            return self._states[epoch]
        else:
            #TODO Implement interpolation of state base on time
            raise NotImplementedError()

    def get_states(self, epochs: List[TimeSpan]):# -> Ephemeris:
        states = [self.get_state(epoch) for epoch in epochs]
        return Ephemeris(states=states)


class Integrator(PropagatorBase):
    
    def __init__(self):
        super().__init__()

    def get_state(self, epoch: TimeSpan, X_0: StateVector) -> StateVector:
        raise NotImplementedError()

    def get_states(self, epochs: List[TimeSpan], X_0: StateVector) -> Ephemeris:
        raise NotImplementedError()


class Euler(Integrator):
    
    def __init__(self):
        super().__init__()


class RK4(Integrator):
    
    def __init__(self):
        super().__init__()


class RK45(Integrator):
    
    def __init__(self):
        super().__init__()
    