from astrolib.base_objects import TimeSpan

class PropagatorBase():
    """Base class for all propagators."""

    def __init__(self):
        self.step_size = TimeSpan.from_seconds(60.0)

    def step(self, step_size: TimeSpan = None):
        raise NotImplementedError()