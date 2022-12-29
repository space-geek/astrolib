from astrolib.attitude_elements import Quaternion
from astrolib.base_objects import Matrix
from astrolib.base_objects import TimeSpan
from astrolib.base_objects import Vector3


class CelestialObjectOrientationModel:
    """TODO: Class docstring"""

    def __init__(self):
        pass

    def get_orientation_at_epoch(self, t: TimeSpan) -> Quaternion:
        """TODO: Method docstring"""
        raise NotImplementedError

    def get_angular_velocity_at_epoch(self, t: TimeSpan) -> Vector3:
        """TODO: Method docstring"""
        raise NotImplementedError


class InertiallyFixedOrientationModel(CelestialObjectOrientationModel):
    """TODO: Class docstring"""

    def __init__(self):
        super().__init__()

    def get_orientation_at_epoch(self, t: TimeSpan) -> Quaternion:
        """TODO: Method docstring"""
        return Quaternion.identity()

    def get_angular_velocity_at_epoch(self, t: TimeSpan) -> Vector3:
        """TODO: Method docstring"""
        return Vector3.zeros()
