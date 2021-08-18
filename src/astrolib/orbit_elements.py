""" TODO: Module docstring
"""
from astrolib import Matrix
from astrolib import Vector3
from astrolib.base_objects import ElementSetBase


class OrbitElementSet(ElementSetBase):
    """ Class represents a generic set of orbital elements.
    """


class CartesianElements(OrbitElementSet):
    """ Class represents a set of Cartesian orbital elements, e.g. position and velocity.
    """

    def __init__(self, position: Vector3 = Vector3.zeros(), velocity: Vector3 = Vector3.zeros()):
        super().__init__([position, velocity])

    def __str__(self) -> str:
        return f"Position: {self.position}\nVelocity: {self.velocity}"

    @property
    def x(self) -> float:
        return self._elements[0,0]

    @x.setter
    def x(self, value: float):
        self._elements[0,0] = value

    @property
    def y(self) -> float:
        return self._elements[1,0]

    @y.setter
    def y(self, value: float):
        self._elements[1,0] = value

    @property
    def z(self) -> float:
        return self._elements[2,0]

    @z.setter
    def z(self, value: float):
        self._elements[2,0] = value

    @property
    def v_x(self) -> float:
        return self._elements[3,0]

    @v_x.setter
    def v_x(self, value: float):
        self._elements[3,0] = value

    @property
    def v_y(self) -> float:
        return self._elements[4,0]

    @v_y.setter
    def v_y(self, value: float):
        self._elements[4,0] = value

    @property
    def v_z(self) -> float:
        return self._elements[5,0]

    @v_z.setter
    def v_z(self, value: float):
        self._elements[5,0] = value

    @property
    def position(self) -> Vector3:
        return Vector3(*[x[0] for x in self._elements[:3,0]])

    @position.setter
    def position(self, value: Vector3):
        self._elements[0,0] = value.x
        self._elements[1,0] = value.y
        self._elements[2,0] = value.z

    #TODO Figure out a way to enable "elements.position.x = float_value" usage. Currently does not make it back to self._elements data structure.

    @property
    def velocity(self) -> Vector3:
        return Vector3(*[x[0] for x in self._elements[3:,0]])

    @velocity.setter
    def velocity(self, value: Vector3):
        self._elements[3,0] = value.x
        self._elements[4,0] = value.y
        self._elements[5,0] = value.z

    #TODO Figure out a way to enable "elements.velocity.x = float_value" usage. Currently does not make it back to self._elements data structure.


class KeplerianElements(OrbitElementSet):
    """Class represents a set of classical Keplerian orbital elements."""

    def __init__(self, sma: float, ecc: float, inc: float, raan: float, arg_of_periapsis: float, true_anomaly: float):
        super().__init__([Matrix([[sma],[ecc],[inc],[raan],[arg_of_periapsis],[true_anomaly]])])

    def __str__(self) -> str:
        return f"Semimajor Axis: {self.sma}\n" + \
               f"Eccentricity: {self.ecc}\n" + \
               f"Inclination: {self.inc}\n" + \
               f"Right Ascension of the Ascending Node: {self.raan}\n" + \
               f"Argument of Periapsis: {self.arg_of_periapsis}\n" + \
               f"True Anomaly: {self.true_anomaly}"

    @property
    def sma(self) -> float:
        return self._elements[0,0]

    @sma.setter
    def sma(self, value: float):
        self._elements[0,0] = value

    @property
    def ecc(self) -> float:
        return self._elements[1,0]

    @ecc.setter
    def ecc(self, value: float):
        self._elements[1,0] = value

    @property
    def inc(self) -> float:
        return self._elements[2,0]

    @inc.setter
    def inc(self, value: float):
        self._elements[2,0] = value

    @property
    def raan(self) -> float:
        return self._elements[3,0]

    @raan.setter
    def raan(self, value: float):
        self._elements[3,0] = value

    @property
    def arg_of_periapsis(self) -> float:
        return self._elements[4,0]

    @arg_of_periapsis.setter
    def arg_of_periapsis(self, value: float):
        self._elements[4,0] = value

    @property
    def true_anomaly(self) -> float:
        return self._elements[5,0]

    @true_anomaly.setter
    def true_anomaly(self, value: float):
        self._elements[5,0] = value
