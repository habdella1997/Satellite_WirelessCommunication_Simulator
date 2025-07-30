# This file includes a template of how a simulation can be ran
from datetime import datetime, timedelta
from simulator.simulation import Simulation
from simulator.satellite import Constellation
import numpy as np

if __name__ == '__main__':
    # Simulation initialization
    start_time = datetime(2022, 1, 1)
    end_time = start_time + timedelta(hours=1)
    time_step = timedelta(minutes=1)
    sim = Simulation(t_start=start_time, t_end=end_time, t_step=timedelta(minutes=1))
    np.random.seed(0)

    const = Constellation.starlink_phase1(sim)
    const.plot_constellation(orbits=True)



