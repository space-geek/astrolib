from integrationutils.attitude_elements import Quaternion
from integrationutils.base_objects import TimeSpan
from integrationutils.base_objects import Vec3d
from integrationutils.solar_system.motion_models import CelestialObjectMotionModel
from integrationutils.solar_system.orientation_models import CelestialObjectOrientationModel

class CelestialObject:

    def __init__(self, motion_model: CelestialObjectMotionModel, orientation_model: CelestialObjectOrientationModel):
        self._motion_model = motion_model
        self._orientation_model = orientation_model

    def get_position_at_epoch(self, epoch: TimeSpan) -> Vec3d:
        return self._motion_model.get_position_at_epoch(epoch)

    def get_velocity_at_epoch(self, epoch: TimeSpan) -> Vec3d:
        return self._motion_model.get_velocity_at_epoch(epoch)

    def get_orientation_at_epoch(self, epoch: TimeSpan) -> Quaternion:
        return self._orientation_model.get_orientation_at_epoch(epoch)
