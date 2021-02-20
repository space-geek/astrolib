# from astrolib.util.file_io import parse_leap_seconds_file

# if __name__ == "__main__":
#     parse_leap_seconds_file("./data/utctai.dat")


from astrolib.base_objects import TimeSpan
from astrolib.base_objects import Vec3d
from astrolib.state_vector import StateVector
from astrolib.orbit_elements import CartesianElements
from astrolib.orbit_elements import KeplerianElements
from astrolib.attitude_elements import Quaternion
from astrolib.solar_system import Earth
from astrolib.solar_system import Sun
from astrolib.solar_system import Moon

if __name__ == "__main__":
    ts0 = TimeSpan.from_utc("Sep 23 2020 00:00:00Z")

    ts = ts0 + TimeSpan.from_days(1.0)

    orbit_state = KeplerianElements(10000, 0.0, 0.0, 0.0, 0.0, 0.0)
    orbit_state = CartesianElements(Vec3d.zeros(), Vec3d.zeros())
    attitude_state = Quaternion.identity()
    X_0 =  StateVector(ts0, [orbit_state, attitude_state])

    #Integrator-type propagators
    prop = Euler(step_size=TimeSpan.from_seconds(2.0))
    prop = RK4(step_size=TimeSpan.from_seconds(2.0))
    prop = RK45(step_size=TimeSpan.from_seconds(300), relative_tolerance=1e-6)
    prop.dynamics_model.add_force(Earth.gravity)
    prop.dynamics_model.add_force(Earth.magnetic_field)
    prop.dynamics_model.add_force(Earth.atmospheric_drag)
    prop.dynamics_model.add_force(Moon.gravity)
    prop.dynamics_model.add_force(Sun.gravity)
     # Analytic-type propagators
    prop = TwoBodyAnalytic(Earth) # __init__(self, central_body: CelestialObject)
    prop = J2MeanAnalytic(Mars)

    # Case 0: Simple single step:
    # Propagator.get_state(epoch: TimeSpan, X_0: StateVector) -> StateVector:
    X_f = prop.get_state(ts, X_0)
    print(X_f)
    
    # Case 1: Multiple epochs, generate state at each epoch:
    # Propagator.get_states(epochs: List[TimeSpan], X_0: StateVector) -> Ephemeris
    ephem = prop.get_states([X_0.epoch + i*TimeSpan.from_minutes(300.0) for i in range(0,6)], X_0)
    for state in ephem:
        print(state)
    print(ephem) # start/end epochs, number of state vectors, types of data contained
    
    # Case 2: Manual version of case 1
    tvec = [X_0.epoch + i*TimeSpan.from_minutes(300.0) for i in range(0,6)]
    states = [X_0]
    for epoch in tvec:
        states.append(prop.get_state(epoch, states[-1]))
    ephem = Ephemeris()
    ephem.add_states(states)
    for state in ephem:
        print(state)
    print(ephem)

    # Case 2: Propagate from ephemeris
    ephem = Ephemeris.load_from_file("foo.txt")
    print(ephem)
    for state in ephem:
        print(state)
    X = ephem.get_state(ts) # either return exact state or interpolate
    eph2 = ephem.get_states([x + TimeSpan.from_seconds(1) for x in tvec if ephem.start_epoch < x < ephem.end_epoch]) # either return exact state or interpolate
