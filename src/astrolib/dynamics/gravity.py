""" TODO: Module docstring
"""
from astrolib import Matrix
from astrolib import TimeSpan
from astrolib import Vector3
from astrolib.dynamics import ForceModelBase
from astrolib.solar_system.motion_models import CelestialObjectMotionModel
from astrolib.state_vector import CartesianStateVector


class PointMassGravityModel(ForceModelBase):

    supported_state_vector_types = set([CartesianStateVector])

    def __init__(self, mu: float, motion_model: CelestialObjectMotionModel):
        super().__init__()
        self.mu = mu
        self.motion_model = motion_model

    def compute_acceleration(self, state: CartesianStateVector) -> Matrix:
        rel_pos = state.elements.position - self.motion_model.get_position_at_epoch(state.epoch)
        return -(self.mu / (rel_pos.norm()**3)) * rel_pos

    def compute_partials(self, state: CartesianStateVector) -> Matrix:
        raise NotImplementedError


class SphericalHarmonicGravityModel(ForceModelBase):

    supported_state_vector_types = (CartesianStateVector)

    def __init__(self, mu: float, motion_model: CelestialObjectMotionModel, coefficients: Matrix):
        super().__init__()
        self._point_mass_component = PointMassGravityModel(mu, motion_model)
        self._coeffs = coefficients

    def compute_acceleration(self, state: CartesianStateVector) -> Matrix:
        #TODO Implement spherical harmonic acceleration model
        return self._point_mass_component.compute_acceleration(state) + Vector3.zeros()

    def compute_partials(self, t: TimeSpan, X: Matrix) -> Matrix:
        raise NotImplementedError
