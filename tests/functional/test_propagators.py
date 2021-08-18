from astrolib import TimeSpan
from astrolib import Vec3d
from astrolib.dynamics.gravity import PointMassGravityModel
from astrolib.propagators import RK4
from astrolib.propagators import RK45
from astrolib.solar_system.motion_models import OriginFixedMotionModel
from astrolib.state_vector import CartesianStateVector
from astrolib.util.constants import EARTH_MU

def main():

    num_steps = 10

    initial_state = CartesianStateVector()
    initial_state.epoch = TimeSpan.from_days(1.0)
    initial_state.elements.position = Vec3d(42164, 0, 0)
    initial_state.elements.velocity = Vec3d(0, 3.0746, 0.01)
    # initial_state.elements.position.x = 42164.0
    # initial_state.elements.position.y = 0.0
    # initial_state.elements.position.z = 0.0
    # initial_state.elements.velocity.x = 0.0
    # initial_state.elements.velocity.y = 3.0746
    # initial_state.elements.velocity.z = 0.0

    print(initial_state)

    earth_gravity = PointMassGravityModel(EARTH_MU, OriginFixedMotionModel())

    rk4 = RK4(TimeSpan.from_seconds(300))
    rk4.dynamics_model.forces.append(earth_gravity)
    rk45 = RK45()
    rk45.dynamics_model.forces.append(earth_gravity)

    ephem_rk4 = rk4.get_states([initial_state.epoch + i * rk4.step_size for i in range(0, num_steps)], initial_state)
    ephem_rk45 = rk45.get_states(list(ephem_rk4.epochs), initial_state)

    for i, (s_rk4, s_rk45) in enumerate(zip(ephem_rk4, ephem_rk45)):
        print(f"State Diff #{i+1}:")
        print(f"Epoch:    {s_rk45.epoch - s_rk4.epoch}")
        print(f"Position: {s_rk45.elements.position - s_rk4.elements.position}")
        print(f"Velocity: {s_rk45.elements.velocity - s_rk4.elements.velocity}")

    print(ephem_rk4)
    print(ephem_rk45)

if __name__ == "__main__":
    main()
