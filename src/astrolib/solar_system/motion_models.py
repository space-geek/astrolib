from astrolib.base_objects import Matrix
from astrolib.base_objects import TimeSpan
from astrolib.base_objects import Vector3
from astrolib.orbit_elements import CartesianElements


class CelestialObjectMotionModel:

    def __init__(self):
        pass

    def get_position_at_epoch(self, t: TimeSpan) -> Vector3:
        raise NotImplementedError

    def get_posvel_at_epoch(self, t: TimeSpan) -> CartesianElements:
        raise NotImplementedError

class OriginFixedMotionModel(CelestialObjectMotionModel):

    def __init__(self):
        super().__init__()

    def get_position_at_epoch(self, _):
        return Vector3.zeros()

    def get_posvel_at_epoch(self, _) -> CartesianElements:
        return CartesianElements(position=Vector3.zeros(), velocity=Vector3.zeros())
