from astrolib.base_objects import Matrix
from astrolib.base_objects import TimeSpan
from astrolib.base_objects import Vec3d
from astrolib.solar_system.motion_models import CelestialObjectMotionModel
from astrolib.dynamics.dynamics_base import ForceModelBase
from astrolib.orbit_elements import CartesianElements


class PointMassGravityModel(ForceModelBase):

    def __init__(self, mu: float, motion_model: CelestialObjectMotionModel):
        super().__init__()
        self.mu = mu
        self.motion_model = motion_model

    def compute_derivatives(self, t: TimeSpan, X: CartesianElements) -> Matrix:
        return Matrix.from_column_matrices([X.velocity, self.get_acceleration_at_epoch(t, X)])

    def compute_partials(self, t: TimeSpan, X: Matrix) -> Matrix:
        return Matrix.zeros(X.num_rows)

    def get_acceleration_at_epoch(self, t: TimeSpan, X: CartesianElements) -> Vec3d:
        rel_pos = X.position - self.motion_model.get_position_at_epoch(t)
        return -(self.mu / (rel_pos.norm()**3)) * rel_pos


class SphericalHarmonicGravityModel(ForceModelBase):

    def __init__(self, mu: float, motion_model: CelestialObjectMotionModel, coefficients: Matrix):
        super().__init__()
        self._point_mass_component = PointMassGravityModel(mu, motion_model)
        self._coeffs = coefficients

    def get_derivatives(self, t: TimeSpan, X: CartesianElements) -> Matrix:
        return Matrix.from_column_matrices([X.velocity, self.get_acceleration_at_epoch(t, X)])

    def get_partials(self, t: TimeSpan, X: Matrix) -> Matrix:
        raise NotImplementedError()

    def get_acceleration_at_epoch(self, t: TimeSpan, X: CartesianElements) -> Vec3d:
        return self._point_mass_component.get_acceleration_at_epoch(t, X) + Vec3d.zeros()
