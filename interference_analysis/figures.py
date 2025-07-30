from datetime import datetime, timedelta
from simulator.simulation import Simulation
import numpy as np
from simulator import constants as c
from simulator.plot_utils import get_ax
from simulator.satellite import Orbit, Constellation
import imageio
import simulator.toolkit as tk
import simulator.astrodynamics as ad
from simulator.link_budget import rx_power
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from tqdm import tqdm
Re = c.EARTH_RADIUS_M


def system_figure():
    n_sats = 10
    h = 500 * 1e3
    h_c = 510 * 1e3

    # Simulation initialization
    start_time = datetime(2022, 1, 1)
    end_time = start_time + timedelta(seconds=60)
    time_step = (end_time - start_time) / 1
    sim = Simulation(t_start=start_time, t_end=end_time, t_step=time_step)
    np.random.seed(0)

    # Common orbital parameters
    e = 0.0  # eccentricity
    w = 0  # Argument of perigee in degrees
    inc = 50  # Inclination in degrees

    # Target orbit
    omega = 0  # Right ascension of ascending node in degrees
    a = c.EARTH_RADIUS_M + h  # Semimajor axis in meters
    target_orbit = Orbit(sim, e, a, omega, inc, w, initial_anomaly=0)
    target_orbit.add_satellites(n_sats=n_sats)

    # Shifted orbit 1
    omega_s = 120
    a = c.EARTH_RADIUS_M + h  # Semimajor axis in meters
    shifted_orbit = Orbit(sim, e, a, omega_s, inc, w, initial_anomaly=0)
    shifted_orbit.add_satellites(n_sats=n_sats)

    # Shfted orbit 2
    omega = 240  # Right ascension of ascending node in degrees
    a = c.EARTH_RADIUS_M + h  # Semimajor axis in meters
    coplanar_orbit = Orbit(sim, e, a, omega, inc, w, initial_anomaly=0)
    coplanar_orbit.add_satellites(n_sats=10)

    const = Constellation(sim, orbits=[target_orbit, shifted_orbit, coplanar_orbit])
    const.update_SSPs()  # Required before computing distances between satellites

    tx_id = 1
    rx_id = 0
    tx_sat = const.satellites[tx_id]
    rx_sat = const.satellites[rx_id]
    int_sat = const.orbits[1].satellites[0]

    const.plot_constellation(filename='const_plot_{}'.format(sim.t_current.strftime("%Y%m%d%H%M%S")), show=True,
                             orbits=True, file_extension='.pdf', equator=True,
                             color_by_orbit=True, highlights=[rx_sat, int_sat], annotate=False)


if __name__=="__main__":
    system_figure()
