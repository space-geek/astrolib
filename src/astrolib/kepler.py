""" TODO Module docstring
"""

from astrolib import Vector3
from astrolib.orbit_elements import OrbitElementSet
from astrolib.constants import EARTH_MU


def angular_momentum_vector(elements: OrbitElementSet) -> Vector3:
    """TODO: Function docstring"""
    # TODO inject orbit element conversion to cartesian elements
    return elements.position.cross(elements.velocity)


def eccentricity_vector(elements: OrbitElementSet, mu: float = EARTH_MU) -> Vector3:
    """TODO: Function docstring"""
    # TODO inject orbit element conversion to cartesian elements
    hvec: Vector3 = angular_momentum_vector(elements)
    return (1 / mu) * elements.velocity.cross(hvec) - elements.position.normalized()


def specific_energy(elements: OrbitElementSet, mu: float = EARTH_MU) -> float:
    """TODO: Function docstring"""
    # TODO inject orbit element conversion to cartesian elements
    rmag = elements.position.norm()
    vmag = elements.velocity.norm()
    return (1 / 2) * pow(vmag, 2) - mu * (1 / rmag)
