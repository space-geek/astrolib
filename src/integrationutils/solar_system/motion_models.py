from integrationutils.base_objects import Matrix
from integrationutils.base_objects import TimeSpan
from integrationutils.base_objects import Vec3d
from integrationutils.orbit_elements import CartesianElements


class CelestialObjectMotionModel:

    def __init__(self):
        pass

    def get_posvel_at_epoch(self, t: TimeSpan) -> CartesianElements:
        raise NotImplementedError()

class OriginFixedMotionModel(CelestialObjectMotionModel):

    def __init__(self):
        super().__init__()

    def get_posvel_at_epoch(self, _) -> CartesianElements:
        return CartesianElements(position=Vec3d.zeros(), velocity=Vec3d.zeros())
