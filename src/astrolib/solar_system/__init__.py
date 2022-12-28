from astrolib.solar_system.celestial_objects import CelestialObject
from astrolib.solar_system.motion_models import OriginFixedMotionModel
from astrolib.solar_system.orientation_models import InertiallyFixedOrientationModel

Sun = CelestialObject(OriginFixedMotionModel(), InertiallyFixedOrientationModel())
Mercury = CelestialObject(OriginFixedMotionModel(), InertiallyFixedOrientationModel())
Venus = CelestialObject(OriginFixedMotionModel(), InertiallyFixedOrientationModel())
Earth = CelestialObject(OriginFixedMotionModel(), InertiallyFixedOrientationModel())
Mars = CelestialObject(OriginFixedMotionModel(), InertiallyFixedOrientationModel())
Jupiter = CelestialObject(OriginFixedMotionModel(), InertiallyFixedOrientationModel())
Saturn = CelestialObject(OriginFixedMotionModel(), InertiallyFixedOrientationModel())
Neptune = CelestialObject(OriginFixedMotionModel(), InertiallyFixedOrientationModel())
Uranus = CelestialObject(OriginFixedMotionModel(), InertiallyFixedOrientationModel())
Pluto = CelestialObject(OriginFixedMotionModel(), InertiallyFixedOrientationModel())

EarthSystemBarycenter = CelestialObject(OriginFixedMotionModel(), InertiallyFixedOrientationModel())
SolarSystemBarycenter = CelestialObject(OriginFixedMotionModel(), InertiallyFixedOrientationModel())
