from astrolib.base_objects import Matrix
from astrolib.base_objects import TimeSpan
from astrolib.base_objects import Vec3d
from astrolib.solar_system.motion_models import CelestialObjectMotionModel
from astrolib.dynamics.dynamics_base import AccelerationModel
from astrolib.orbit_elements import CartesianElements


class PointMassGravityModel(AccelerationModel):

    def __init__(self, mu: float, motion_model: CelestialObjectMotionModel):
        super().__init__()
        self.mu = mu
        self.motion_model = motion_model

    def compute_acceleration(self, t: TimeSpan, X: CartesianElements) -> Vec3d:
        rel_pos = X.position - self.motion_model.get_position_at_epoch(t)
        return -(self.mu / (rel_pos.norm()**3)) * rel_pos

    def compute_partials(self, t: TimeSpan, X: Matrix) -> Matrix:
        raise NotImplementedError()


class SphericalHarmonicGravityModel(AccelerationModel):

    def __init__(self, mu: float, motion_model: CelestialObjectMotionModel, coefficients: Matrix):
        super().__init__()
        self._point_mass_component = PointMassGravityModel(mu, motion_model)
        self._coeffs = coefficients

    def compute_acceleration(self, t: TimeSpan, X: CartesianElements) -> Matrix:
        return self._point_mass_component.get_acceleration_at_epoch(t, X) + Vec3d.zeros()

    def compute_partials(self, t: TimeSpan, X: Matrix) -> Matrix:
        raise NotImplementedError()
