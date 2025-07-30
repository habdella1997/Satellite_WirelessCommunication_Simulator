from datetime import datetime, timedelta
import os
import getpass
import numpy as np

main_path = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(main_path, "cache")
data_path = os.path.join(main_path, "../data")
results_path = os.path.join(main_path, "../results")


class Simulation:
    """Simulation object. Contains all the information of the analysis and path to store results
    """

    def __init__(self, t_start=datetime.now(), t_end=datetime.now() + timedelta(minutes=5), t_step=timedelta(minutes=1), name="Simulation", results_folder=None,
                 make_new_folder=True):
        """
        Parameters
        ----------
        t_start : datetime,
            The start time for the simulation. Default time
            is the current system time
        t_end : datetime, optional
            The end time for the simulation. If None is provided, one minute from
            the start time
        t_step : timedelta, optional
            The time step used during the simulation. If None is provided, one
            second will be used.
        results_folder : string, optional
            Path to the folder where results will be saved
        name : string, optional
            Name of the simulation. Default value is "Simulation"
        ts: array of timestamps of the simulation according to t_start, t_end, t_step
        """

        if t_start is None:
            t_start = datetime.now()
        if t_end is None:
            t_end = t_start + timedelta(minutes=1)
        if t_step is None:
            t_step = timedelta(seconds=1)

        self.t_start = t_start
        self.t_end = t_end
        self.t_step = t_step
        self.t_step_in_seconds = self.t_step.days * 24 * 3600 + self.t_step.seconds
        self.duration = t_end - t_start
        self.duration_in_seconds = self.duration.days * 24 * 3600 + self.duration.seconds
        self.t_current = t_start
        self.name = name

        # Create time vectors
        self.ts_utc = np.arange(self.t_start, self.t_start + timedelta(hours=24),
                                timedelta(minutes=5),
                                dtype='datetime64[ms]')
        self.ts = [self.t_start + self.t_step * i
                   for i in range(int((self.t_end - self.t_start) / self.t_step) + 1)]

        # Results folder
        if results_folder is None:
            if make_new_folder is True:
                results_folder = os.path.join(results_path,
                                              'simulation_' + \
                                              str(datetime.now().strftime("%Y%m%d%H%M%S")) + \
                                              '_' + getpass.getuser())
            else:
                results_folder = os.path.join(results_path, 'default_simulation_results')

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        figures_folder = os.path.join(results_folder, 'figures')
        if not os.path.exists(figures_folder):
            os.makedirs(figures_folder)

        self.results_folder = results_folder + "/"
        self.figures_folder = figures_folder + "/"

        self.objects = []


class SimulationObject:
    """Parent class for al simulation objects: Constellations, Satellites, GroundStations,..."""
    _ID = 0

    def __init__(self, simulation=None, name="Simulation Object"):
        SimulationObject._ID += 1
        self.__ID__ = SimulationObject._ID

        self.simulation = simulation
        self.name = name
        if self.simulation:
            self.simulation.objects.append(self)

    def __str__(self):
        return '<SimulationObject {0}>'.format(self.__ID__)
