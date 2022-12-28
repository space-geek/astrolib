from astrolib.attitude_elements import Quaternion
from astrolib.base_objects import Matrix
from astrolib.base_objects import TimeSpan
from astrolib.base_objects import Vector3



class CelestialObjectOrientationModel:

    def __init__(self):
        pass

    def get_orientation_at_epoch(self, epoch: TimeSpan) -> Quaternion:
        raise NotImplementedError

    def get_angular_velocity_at_epoch(self, epoch: TimeSpan) -> Vector3:
        raise NotImplementedError


class InertiallyFixedOrientationModel(CelestialObjectOrientationModel):

    def __init__(self):
        super().__init__()

    def get_orientation_at_epoch(self, _) -> Quaternion:
        return Quaternion.identity()

    def get_angular_velocity_at_epoch(self, _) -> Vector3:
        return Vector3.zeros()
