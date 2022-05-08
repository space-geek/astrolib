""" TODO Module docstring
"""

from astrolib import Vector3
from astrolib.orbit_elements import OrbitElementSet
from astrolib.constants import EARTH_MU


def angular_momentum_vector(elements: OrbitElementSet) -> Vector3:
    #TODO inject orbit element conversion to cartesian elements
    return elements.position.cross(elements.velocity)

def eccentricity_vector(elements: OrbitElementSet, mu: float = EARTH_MU) -> Vector3:
    #TODO inject orbit element conversion to cartesian elements
    return (1 / mu) * elements.velocity.cross(angular_momentum_vector(elements)) - elements.position.normalized()

def specific_energy(elements: OrbitElementSet, mu: float = EARTH_MU) -> float:
    #TODO inject orbit element conversion to cartesian elements
    return (1 / 2) * pow(elements.velocity.norm(), 2) - mu * (1 / elements.position.norm())
