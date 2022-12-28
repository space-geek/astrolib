""" TODO: Module docstring
"""
import enum
from astrolib import Vector3



class CoordinateFrame(enum.Enum):
    """ Enumeration of supported coordinate frames.
    """
    ICRF = enum.auto()
    GCRF = enum.auto()
    J2000 = enum.auto()
    BODY = enum.auto()
    ECEF = enum.auto()


class ReferenceFrame():
    """ TODO: Class docstring
    """

    def __init__(self):
        self.x_axis: Vector3 = Vector3(1, 0, 0)
        self.y_axis: Vector3 = Vector3(0, 1, 0)
        self.z_axis: Vector3 = Vector3(0, 0, 1)


