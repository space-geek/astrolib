""" Sandbox file for script snippet tests.
"""
from astrolib import Vector3
from astrolib.orbit_elements import CartesianElements
from astrolib.base.frames import CoordinateFrame
from astrolib.solar_system import Earth
from astrolib.solar_system import EarthSystemBarycenter



if __name__ == "__main__":

    state = CartesianElements(position=Vector3(42164, 0, 0), velocity=Vector3(0, 3.074, 0))
    state = CartesianElements(position=..., velocity=...) # uses origin=Earth, frame=CoordinateFrame.GCRF
    state = CartesianElements(position=..., velocity=..., origin=Earth, frame=CoordinateFrame.J2000)

    eci_state = CartesianElements(..., frame=CoordinateFrame.J2000).rotate_to(CoordinateFrame.GCRF)
    ecef_state = eci_state.rotate_to(CoordinateFrame.ECEF) # performs GCRF -> ECEF rotation (no translation)
    barycentric_inertial_state = ecef_state.rotate_to(CoordinateFrame.ICRF).relative_to(EarthSystemBarycenter)
    diff_between_eci_states = barycentric_inertial_state.relative_to(eci_state)
