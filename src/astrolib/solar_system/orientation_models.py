from astrolib.attitude_elements import Quaternion
from astrolib.base_objects import Matrix
from astrolib.base_objects import TimeSpan
from astrolib.base_objects import Vec3d


class CelestialObjectOrientationModel:

    def __init__(self):
        pass

    def get_orientation_at_epoch(self, t: TimeSpan) -> Quaternion:
        raise NotImplementedError()

    def get_angular_velocity_at_epoch(self, t: TimeSpan) -> Vec3d:
        raise NotImplementedError()

class InertiallyFixedOrientationModel(CelestialObjectOrientationModel):

    def __init__(self):
        super().__init__()

    def get_orientation_at_epoch(self, t: TimeSpan) -> Quaternion:
        return Quaternion.identity()

    def get_angular_velocity_at_epoch(self, t: TimeSpan) -> Vec3d:
        return Vec3d.zeros()
