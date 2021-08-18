from astrolib.attitude_elements import Quaternion
from astrolib.base_objects import TimeSpan
from astrolib.base_objects import Vector3
from astrolib.solar_system.motion_models import CelestialObjectMotionModel
from astrolib.solar_system.orientation_models import CelestialObjectOrientationModel

class CelestialObject:

    def __init__(self, motion_model: CelestialObjectMotionModel, orientation_model: CelestialObjectOrientationModel):
        self._motion_model = motion_model
        self._orientation_model = orientation_model

    def get_position_at_epoch(self, epoch: TimeSpan) -> Vector3:
        return self._motion_model.get_position_at_epoch(epoch)

    def get_velocity_at_epoch(self, epoch: TimeSpan) -> Vector3:
        return self._motion_model.get_velocity_at_epoch(epoch)

    def get_orientation_at_epoch(self, epoch: TimeSpan) -> Quaternion:
        return self._orientation_model.get_orientation_at_epoch(epoch)
