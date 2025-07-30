import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import simulator.toolkit as tk
import simulator.constants as c
from datetime import datetime, timedelta
from simulator.plot_utils import get_ax
from itertools import product, cycle, islice
from matplotlib.ticker import FuncFormatter
import simulator.astrodynamics as ad

mpl.rc('font', family='Times New Roman')

# Simulator path
sys.path.append('../simulator')

# Constants
cache_path = '../interference_analysis/cache'
plots_path = '../interference_analysis/plots'
single_orbit_math_results_path = os.path.join(cache_path, 'single_orbit_math.dat')
single_orbit_simulation_results_path = os.path.join(cache_path, 'single_orbit_simulation.dat')
Re = c.EARTH_RADIUS_M

# Color data structures
markers = np.array(list(islice(cycle(list(['D', 'p', 'v', '^', 's', 'o'])), 7 + 1)))
markers2 = np.array(list(islice(cycle(list(['D', 'p', 'v', '^', 's', 'o'])), 7 + 1)))
sizesplot = np.array(list(islice(cycle(list([7, 11, 10, 10, 9, 10])), 7 + 1)))
sizes = np.array(list(islice(cycle(list([70, 110, 100, 100, 90, 100])), 7 + 1)))
sizes2 = np.array(list(islice(cycle(list([8, 11, 10, 10, 9, 10])), 7 + 1)))


# Function to adjust the columns with dB units after linearly averaging across one of the input columns
def adjust_db_columns(results_df):
    source_columns = ['interference_linear', 'rx_power_linear']
    results_columns = ['interference', 'rx_power']

    def compute_row_results(row):
        results = []
        for result in source_columns:
            value = row[result]
            if value:
                results.append(10 * np.log10(value))
            else:
                results.append(None)
        return results

    results_df[results_columns] = results_df.apply(compute_row_results, axis=1, result_type='expand')


# Function to format the time tick labels in time plots
def format_func(x, pos):
    hours = int(x // 3600)
    minutes = int((x % 3600) // 60)
    seconds = int(x % 60)

    return "{:d}:{:02d}".format(hours, minutes)
    # return "{:d}:{:02d}:{:02d}".format(hours, minutes, seconds)


class Plotter:
    """Class containing all the methods for plotting results

    This class contains all the methods to generate the results (passed as a pandas.Dataframe)
    for all of the analyzed scenarios (Single orbit, Co-planar orbits and Shifted Orbits)
    and with the different approaches (math, simulation).
    """

    def __init__(self, results_columns, link_budget_input_params=None, orbital_input_params=None,
                 math_results_df=None,
                 sim_results_df=None,
                 high_math_results_df=None,
                 high_sim_results_df=None,
                 thz_input_params=None,
                 mmwave_input_params=None,
                 thz_math_results_df=None,
                 mmwave_math_results_df=None,
                 thz_sim_results_df=None,
                 mmwave_sim_results_df=None):
        self.link_budget_input_params = link_budget_input_params
        self.orbital_input_params = orbital_input_params
        self.math_results_df = math_results_df
        self.sim_results_df = sim_results_df
        self.results_columns = results_columns
        self.font_line_sizes = tk.templateFormat()
        self.high_math_results_df = high_math_results_df
        self.high_sim_results_df = high_sim_results_df

        # Attributes for plots comparing mmWave with THz
        self.thz_input_params = thz_input_params
        self.mmWave_input_params = mmwave_input_params
        self.thz_math_results_df = thz_math_results_df
        self.thz_sim_results_df = thz_sim_results_df
        self.mmWave_math_results_df = mmwave_math_results_df
        self.mmWave_sim_results_df = mmwave_sim_results_df

    # Single Orbit analysis plots
    # X axis: altitude
    def tx_rx_distance_vs_orbit_altitude_plot(self, n_sats=50, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        math_results_df = self.math_results_df
        math_h_list = np.arange(self.orbital_input_params['h_lims'][0],
                                self.orbital_input_params['h_lims'][1],
                                self.orbital_input_params['h_resolution_math'])

        sim_results_df = self.sim_results_df
        sim_h_list = np.arange(self.orbital_input_params['h_lims'][0],
                               self.orbital_input_params['h_lims'][1],
                               self.orbital_input_params['h_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=True)

        fig = get_ax()
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            y_values = math_results_df.loc[
                           (math_results_df['alpha'] == alpha) & (
                                   math_results_df['n_sats'] == n_sats), 'tx_rx_distance'] * 1e-3
            plt.plot(math_h_list * 1e-3, y_values, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)
        # Simulation
        for i, alpha in enumerate(alpha_list):
            y_values = sim_results_df.loc[
                           (sim_results_df['alpha'] == alpha) & (
                                   sim_results_df['n_sats'] == n_sats), 'tx_rx_distance'] * 1e-3
            plt.plot(sim_h_list * 1e-3, y_values, color=colors[i],
                     label='Sim.: Alpha = {}'.format(alpha), markersize=5, markevery=None)

        plt.suptitle('Single orbit: Tx-Rx distance', fontsize=self.font_line_sizes['titleSize'])
        plt.title('Number of satellites = {}'.format(n_sats), fontsize=self.font_line_sizes['subTitleSize'])
        plt.xlabel('Orbit altitude [km]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('Tx-Rx distance [km]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        extension = 'h_TxRx_distance_' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def rx_power_vs_orbit_altitude_plot(self, n_sats=50, math=True, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        if math:
            results_df = self.math_results_df
            h_list = np.arange(self.orbital_input_params['h_lims'][0],
                               self.orbital_input_params['h_lims'][1],
                               self.orbital_input_params['h_resolution_math'])

        else:
            results_df = self.sim_results_df
            h_list = np.arange(self.orbital_input_params['h_lims'][0],
                               self.orbital_input_params['h_lims'][1],
                               self.orbital_input_params['h_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=True)

        fig = get_ax()
        plt.grid()
        for i, alpha in enumerate(alpha_list):
            y_values = results_df.loc[
                (results_df['alpha'] == alpha) & (results_df['n_sats'] == n_sats), 'rx_power']
            plt.plot(h_list * 1e-3, y_values, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)
        if math:
            plt.suptitle('Single orbit math: Rx power', fontsize=self.font_line_sizes['titleSize'])
        else:
            plt.suptitle('Single orbit sim: Rx power', fontsize=self.font_line_sizes['titleSize'])
        plt.title('Number of satellites = {}'.format(n_sats), fontsize=self.font_line_sizes['subTitleSize'])
        plt.xlabel('Orbit altitude [km]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('Rx power [dBm]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        if math:
            extension = 'h_Rx_power_' + 'math' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        else:
            extension = 'h_Rx_power_' + 'sim' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def interference_vs_orbit_altitude_plot(self, n_sats=50, math=True, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        if math:
            results_df = self.math_results_df
            h_list = np.arange(self.orbital_input_params['h_lims'][0],
                               self.orbital_input_params['h_lims'][1],
                               self.orbital_input_params['h_resolution_math'])

        else:
            results_df = self.sim_results_df
            h_list = np.arange(self.orbital_input_params['h_lims'][0],
                               self.orbital_input_params['h_lims'][1],
                               self.orbital_input_params['h_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=True)

        fig = get_ax()
        plt.grid()
        for i, alpha in enumerate(alpha_list):
            y_values = results_df.loc[
                (results_df['n_sats'] == n_sats) & (results_df['alpha'] == alpha), 'interference_linear']
            y_values = [10 * np.log10(y_value) if y_value != 0 else None for y_value in y_values]
            plt.plot(h_list * 1e-3, y_values, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)
        if math:
            plt.suptitle('Single orbit math: Expected interference', fontsize=self.font_line_sizes['titleSize'])
        else:
            plt.suptitle('Single orbit sim: Expected interference', fontsize=self.font_line_sizes['titleSize'])
        plt.title('Number of satellites = {}'.format(n_sats), fontsize=self.font_line_sizes['subTitleSize'])
        plt.xlabel('Orbit altitude [km]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('E[I] [dBm]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        if show:
            plt.show()
        if math:
            extension = 'h_interference_' + 'math' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        else:
            extension = 'h_interference_' + 'sim' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def sir_vs_orbit_altitude_plot(self, n_sats=50, file_ext='.jpg', show=True, just_math=False):
        alpha_list = self.link_budget_input_params['alpha_list']
        math_results_df = self.math_results_df
        math_h_list = np.arange(self.orbital_input_params['h_lims'][0],
                                self.orbital_input_params['h_lims'][1],
                                self.orbital_input_params['h_resolution_math'])

        sim_results_df = self.sim_results_df
        sim_h_list = np.arange(self.orbital_input_params['h_lims'][0],
                               self.orbital_input_params['h_lims'][1],
                               self.orbital_input_params['h_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list))

        fig = get_ax()
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            sir_linear = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (
                    math_results_df['alpha'] == alpha), 'sir_linear'])
            y_value = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(math_h_list * 1e-3, y_value, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)

        if not just_math:
            # Simulation
            for i, alpha in enumerate(alpha_list):
                sir_linear = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (
                        sim_results_df['alpha'] == alpha), 'sir_linear'])
                y_value = [10 * np.log10(value) if value is not None else None for value in sir_linear]
                plt.scatter(sim_h_list * 1e-3, y_value, color=colors[i],
                            label='Sim: Alpha = {}'.format(alpha), marker=markers[i], edgecolors='k', zorder=3,
                            s=sizes[i])

        plt.suptitle('Single orbit: Signal to Interference Ratio (SIR)', fontsize=self.font_line_sizes['titleSize'])
        plt.title('Number of satellites = {}'.format(n_sats), fontsize=self.font_line_sizes['subTitleSize'])
        plt.xlabel('Orbit altitude [km]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        extension = 'h_sir_' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def snr_vs_orbit_altitude_plot(self, n_sats=50, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        math_results_df = self.math_results_df
        math_h_list = np.arange(self.orbital_input_params['h_lims'][0],
                                self.orbital_input_params['h_lims'][1],
                                self.orbital_input_params['h_resolution_math'])

        sim_results_df = self.sim_results_df
        sim_h_list = np.arange(self.orbital_input_params['h_lims'][0],
                               self.orbital_input_params['h_lims'][1],
                               self.orbital_input_params['h_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=True)

        fig = get_ax()
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            snr_linear = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (
                    math_results_df['alpha'] == alpha), 'snr_linear'])
            y_values = [10 * np.log10(value) if value is not None else None for value in snr_linear]
            plt.plot(math_h_list * 1e-3, y_values, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)

        # Simulation
        for i, alpha in enumerate(alpha_list):
            snr_linear = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (
                    sim_results_df['alpha'] == alpha), 'snr_linear'])
            y_value = [10 * np.log10(value) if value is not None else None for value in snr_linear]
            plt.scatter(sim_h_list * 1e-3, y_value, color=colors[i],
                        label='Sim: Alpha = {}'.format(alpha), marker=markers[i], edgecolors='k', zorder=3,
                        s=sizes[i])

        plt.suptitle('Single orbit: Signal to Noise Ratio (SNR)', fontsize=self.font_line_sizes['titleSize'])
        plt.title('Number of satellites = {}'.format(n_sats), fontsize=self.font_line_sizes['subTitleSize'])
        plt.xlabel('Orbit altitude [km]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('SNR [dB]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        extension = 'h_snr_' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def sinr_vs_orbit_altitude_plot(self, n_sats=50, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        math_results_df = self.math_results_df
        math_h_list = np.arange(self.orbital_input_params['h_lims'][0],
                                self.orbital_input_params['h_lims'][1],
                                self.orbital_input_params['h_resolution_math'])
        sim_results_df = self.sim_results_df
        sim_h_list = np.arange(self.orbital_input_params['h_lims'][0],
                               self.orbital_input_params['h_lims'][1],
                               self.orbital_input_params['h_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=True)

        system_t = self.link_budget_input_params['T_system']
        bw = self.link_budget_input_params['bandwidth']

        fig = get_ax()
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            sinr_linear = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (
                    math_results_df['alpha'] == alpha), 'sinr_linear'])
            y_value = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(math_h_list * 1e-3, y_value, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)
        # Simulation
        for i, alpha in enumerate(alpha_list):
            sir_linear = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (
                    sim_results_df['alpha'] == alpha), 'sinr_linear'])
            y_value = [10 * np.log10(value) if value is not None else None for value in sir_linear]
            plt.scatter(sim_h_list * 1e-3, y_value, color=colors[i],
                        label='Sim: Alpha = {}'.format(alpha), marker=markers[i], edgecolors='k', zorder=3,
                        s=sizes[i])

        plt.suptitle('Single orbit: Signal to Interference and Noise Ratio (SINR)',
                     fontsize=self.font_line_sizes['titleSize'])
        plt.title('Number of satellites = {}'.format(n_sats), fontsize=self.font_line_sizes['subTitleSize'])
        plt.xlabel('Orbit altitude [km]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('SINR [dB]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        extension = 'h_sinr_' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def capacity_vs_orbit_altitude_plot(self, n_sats=50, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        math_results_df = self.math_results_df
        math_h_list = np.arange(self.orbital_input_params['h_lims'][0],
                                self.orbital_input_params['h_lims'][1],
                                self.orbital_input_params['h_resolution_math'])
        sim_results_df = self.sim_results_df
        sim_h_list = np.arange(self.orbital_input_params['h_lims'][0],
                               self.orbital_input_params['h_lims'][1],
                               self.orbital_input_params['h_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list))

        fig = get_ax()
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            y_value = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) &
                                               (math_results_df['alpha'] == alpha), 'capacity_gbps'])
            plt.plot(math_h_list * 1e-3, y_value, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)

        # Simulation
        for i, alpha in enumerate(alpha_list):
            y_value = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (
                    sim_results_df['alpha'] == alpha), 'capacity_gbps'])
            plt.scatter(sim_h_list * 1e-3, y_value, color=colors[i], label='Sim: Alpha = {}'.format(alpha),
                        marker=markers[i], edgecolors='k', zorder=3, s=sizes[i])

        plt.suptitle('Single orbit: Channel Capacity', fontsize=self.font_line_sizes['titleSize'])
        plt.title('Number of satellites = {}'.format(n_sats), fontsize=self.font_line_sizes['subTitleSize'])
        plt.xlabel('Orbit altitude [km]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('Channel Capacity [Gbps]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.yscale('log')
        plt.legend()
        extension = 'h_capacity_' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def n_interferers_vs_orbit_altitude_plot(self, n_sats=50, math=True, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        if math:
            results_df = self.math_results_df
            h_list = np.arange(self.orbital_input_params['h_lims'][0],
                               self.orbital_input_params['h_lims'][1],
                               self.orbital_input_params['h_resolution_math'])

        else:
            results_df = self.sim_results_df
            h_list = np.arange(self.orbital_input_params['h_lims'][0],
                               self.orbital_input_params['h_lims'][1],
                               self.orbital_input_params['h_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=True)

        fig = get_ax()
        plt.grid()
        for i, alpha in enumerate(alpha_list):
            y_values = results_df.loc[
                (results_df['n_sats'] == n_sats) & (results_df['alpha'] == alpha), 'n_interferers']
            plt.plot(h_list * 1e-3, y_values, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)
        if math:
            plt.suptitle('Single orbit math:\nNumber of satellites contributing to interference at the receiver',
                         fontsize=self.font_line_sizes['titleSize'])
        else:
            plt.suptitle('Single orbit sim:\nNumber of satellites contributing to interference at the receiver',
                         fontsize=self.font_line_sizes['titleSize'])
        plt.title('Number of satellites = {}'.format(n_sats), fontsize=self.font_line_sizes['subTitleSize'])
        plt.xlabel('Orbit altitude [km]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('Number of interferers', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        if show:
            plt.show()
        if math:
            extension = 'h_n_interferers_' + 'math' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        else:
            extension = 'h_n_interferers_' + 'sim' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def distance_to_nearest_interferer_vs_orbit_altitude_plot(self, n_sats=50, math=True, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        if math:
            results_df = self.math_results_df
            h_list = np.arange(self.orbital_input_params['h_lims'][0],
                               self.orbital_input_params['h_lims'][1],
                               self.orbital_input_params['h_resolution_math'])

        else:
            results_df = self.sim_results_df
            h_list = np.arange(self.orbital_input_params['h_lims'][0],
                               self.orbital_input_params['h_lims'][1],
                               self.orbital_input_params['h_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=True)

        fig = get_ax()
        plt.grid()
        for i, alpha in enumerate(alpha_list):
            y_values = results_df.loc[
                           (results_df['alpha'] == alpha) & (results_df['n_sats'] == n_sats), 'd_interferer'] * 1e-3
            plt.plot(h_list * 1e-3, y_values, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)
        if math:
            plt.suptitle('Single orbit math: Distance to closest interferer',
                         fontsize=self.font_line_sizes['titleSize'])
        else:
            plt.suptitle('Single orbit sim: Distance to closest interferer', fontsize=self.font_line_sizes['titleSize'])
        plt.title('Number of satellites = {}'.format(n_sats), fontsize=self.font_line_sizes['subTitleSize'])
        plt.xlabel('Orbit altitude [km]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('Distance [Km]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        if show:
            plt.show()
        if math:
            extension = 'h_d_closest_interferer_' + 'math' + str(
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        else:
            extension = 'h_d_closest_interferer_' + 'sim' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def single_sir_snr_sinr_capacity_vs_orbit_altitude_plot(self, file_ext='.jpg', n_sats=50, alpha_list=None,
                                                            show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list

        math_results_df = self.math_results_df
        math_h_list = np.arange(self.orbital_input_params['h_lims'][0],
                                self.orbital_input_params['h_lims'][1],
                                self.orbital_input_params['h_resolution_math'])

        sim_results_df = self.sim_results_df
        sim_h_list = np.arange(self.orbital_input_params['h_lims'][0],
                               self.orbital_input_params['h_lims'][1],
                               self.orbital_input_params['h_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list))

        fig = get_ax(figsize=(10 * 2, 10))
        scale_factor = 0.8

        # SIR
        ax1 = plt.subplot(221)
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            sir_linear = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (
                    math_results_df['alpha'] == alpha), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(math_h_list * 1e-3, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, alpha in enumerate(alpha_list):
            sir_linear = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (
                    sim_results_df['alpha'] == alpha), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(sim_h_list * 1e-3, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax1.set_title('Single orbit: SIR 2', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit altitude [km]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel(r'SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        # plt.legend(ncol=1, framealpha=0.7, fontsize=self.font_line_sizes['legendSize']*scale_factor, loc='upper right')

        # SNR
        ax2 = plt.subplot(222)
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            snr_linear = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (
                    math_results_df['alpha'] == alpha), 'snr_linear'])
            y_values = [10 * np.log10(value) if value is not None else None for value in snr_linear]
            plt.plot(math_h_list * 1e-3, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, alpha in enumerate(alpha_list):
            snr_linear = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (
                    sim_results_df['alpha'] == alpha), 'snr_linear'])
            y_values = [10 * np.log10(value) if value is not None else None for value in snr_linear]
            plt.scatter(sim_h_list * 1e-3, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax2.set_title('Single orbit: SNR', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit altitude [km]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel(r'SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        # plt.legend(ncol=2, framealpha=0.7, fontsize=self.font_line_sizes['legendSize']*scale_factor, loc='upper right')
        # plt.ylim([2.5, 10])

        # SINR
        ax3 = plt.subplot(223)
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            sinr_linear = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (
                    math_results_df['alpha'] == alpha), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value is not None else None for value in sinr_linear]
            plt.plot(math_h_list * 1e-3, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, alpha in enumerate(alpha_list):
            sinr_linear = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (
                    sim_results_df['alpha'] == alpha), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value is not None else None for value in sinr_linear]
            plt.scatter(sim_h_list * 1e-3, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax3.set_title('Single orbit: SINR', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit altitude [km]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel(r'SINR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])

        # Capacity
        ax3 = plt.subplot(224)
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            y_values = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (
                    math_results_df['alpha'] == alpha), 'capacity_gbps'])
            plt.plot(math_h_list * 1e-3, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, alpha in enumerate(alpha_list):
            y_values = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (
                    sim_results_df['alpha'] == alpha), 'capacity_gbps'])
            plt.scatter(sim_h_list * 1e-3, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax3.set_title('Single orbit: Capacity', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit altitude [km]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel(r'Channel Capacity [Gbps]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yscale('log')
        plt.legend(ncol=2, framealpha=0.7, fontsize=self.font_line_sizes['legendSize'] * scale_factor,
                   loc='upper right')

        extension = 'single_sir_snr_sinr_capacity_vs_orbital_altitude_' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    # X axis: satellites
    def tx_rx_distance_vs_number_of_satellites_plot(self, math=True, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        if math:
            results_df = self.math_results_df
            n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_math'])

        else:
            results_df = self.sim_results_df
            n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        results_df = results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(results_df)

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=True)

        fig = get_ax()
        plt.grid()
        for i, alpha in enumerate(alpha_list):
            y_values = results_df.loc[
                           (results_df['alpha'] == alpha), 'tx_rx_distance'] * 1e-3
            plt.plot(n_sats_list, y_values, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)
        if math:
            plt.suptitle('Single orbit math: Tx-Rx distance', fontsize=self.font_line_sizes['titleSize'])
        else:
            plt.suptitle('Single orbit sim: Tx-Rx distance', fontsize=self.font_line_sizes['titleSize'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('Tx-Rx distance [km]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        if show:
            plt.show()
        if math:
            extension = 'n_sats_TxRx_distance_' + 'math' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        else:
            extension = 'n_sats_TxRx_distance_' + 'sim' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def rx_power_vs_number_of_satellites_plot(self, math=True, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        if math:
            results_df = self.math_results_df
            n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_math'])

        else:
            results_df = self.sim_results_df
            n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        results_df = results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(results_df)

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=True)

        fig = get_ax()
        plt.grid()
        for i, alpha in enumerate(alpha_list):
            y_values = results_df.loc[
                (results_df['alpha'] == alpha), 'rx_power']
            plt.plot(n_sats_list, y_values, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)
        if math:
            plt.suptitle('Single orbit math: Rx power', fontsize=self.font_line_sizes['titleSize'])
        else:
            plt.suptitle('Single orbit sim: Rx power', fontsize=self.font_line_sizes['titleSize'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('Rx power [dBm]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        if show:
            plt.show()
        if math:
            extension = 'n_sats_Rx_power_' + 'math' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        else:
            extension = 'n_sats_Rx_power_' + 'sim' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def interference_vs_number_of_satellites_plot(self, math=True, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        if math:
            results_df = self.math_results_df
            n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_math'])

        else:
            results_df = self.sim_results_df
            n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        results_df = results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(results_df)

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=True)

        fig = get_ax()
        plt.grid()
        for i, alpha in enumerate(alpha_list):
            y_values = results_df.loc[
                (results_df['alpha'] == alpha), 'interference']
            plt.plot(n_sats_list, y_values, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)
        if math:
            plt.suptitle('Single orbit math: Expected interference', fontsize=self.font_line_sizes['titleSize'])
        else:
            plt.suptitle('Single orbit sim: Expected interference', fontsize=self.font_line_sizes['titleSize'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('E[I] [dBm]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        if show:
            plt.show()
        if math:
            extension = 'n_sats_interference_' + 'math' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        else:
            extension = 'n_sats_interference_' + 'sim' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def sir_vs_number_of_satellites_plot(self, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        math_results_df = self.math_results_df
        math_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                     self.orbital_input_params['n_sats_lims'][1],
                                     self.orbital_input_params['n_sats_resolution_math'])
        sim_results_df = self.sim_results_df
        sim_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        math_results_df = math_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(math_results_df)
        sim_results_df = sim_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(sim_results_df)

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=True)

        fig = get_ax()
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            sir_linear = list(math_results_df.loc[(math_results_df['alpha'] == alpha), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(math_n_sats_list, y_values, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)
        # Simulation
        for i, alpha in enumerate(alpha_list):
            sir_linear = list(sim_results_df.loc[(sim_results_df['alpha'] == alpha), 'sir_linear'])
            y_value = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(sim_n_sats_list, y_value, color=colors[i],
                        label='Sim: Alpha = {}'.format(alpha), marker=markers[i], edgecolors='k', zorder=3,
                        s=sizes[i])

        plt.suptitle('Single orbit math: Signal to Interference Ratio (SIR)',
                     fontsize=self.font_line_sizes['titleSize'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        extension = 'n_sats_sir_' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def snr_vs_number_of_satellites_plot(self, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        math_results_df = self.math_results_df
        math_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                     self.orbital_input_params['n_sats_lims'][1],
                                     self.orbital_input_params['n_sats_resolution_math'])
        sim_results_df = self.sim_results_df
        sim_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        math_results_df = math_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(math_results_df)
        sim_results_df = sim_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(sim_results_df)

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=True)

        fig = get_ax()
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            sir_linear = list(math_results_df.loc[(math_results_df['alpha'] == alpha), 'snr_linear'])
            y_values = [10 * np.log10(value) if value is not None else None for value in sir_linear]
            plt.plot(math_n_sats_list, y_values, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)
        # Simulation
        for i, alpha in enumerate(alpha_list):
            sir_linear = list(sim_results_df.loc[(sim_results_df['alpha'] == alpha), 'snr_linear'])
            y_value = [10 * np.log10(value) if value is not None else None for value in sir_linear]
            plt.scatter(sim_n_sats_list, y_value, color=colors[i],
                        label='Sim: Alpha = {}'.format(alpha), marker=markers[i], edgecolors='k', zorder=3,
                        s=sizes[i])
        plt.suptitle('Single orbit: Signal to Noise Ratio (SNR)', fontsize=self.font_line_sizes['titleSize'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('SNR [dB]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        extension = 'n_sats_snr_' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def sinr_vs_number_of_satellites_plot(self, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        math_results_df = self.math_results_df
        math_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                     self.orbital_input_params['n_sats_lims'][1],
                                     self.orbital_input_params['n_sats_resolution_math'])
        sim_results_df = self.sim_results_df
        sim_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        math_results_df = math_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(math_results_df)
        sim_results_df = sim_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(sim_results_df)

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=True)

        fig = get_ax()
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            sir_linear = list(math_results_df.loc[(math_results_df['alpha'] == alpha), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(math_n_sats_list, y_values, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)
        # Simulation
        for i, alpha in enumerate(alpha_list):
            sir_linear = list(sim_results_df.loc[(sim_results_df['alpha'] == alpha), 'sinr_linear'])
            y_value = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(sim_n_sats_list, y_value, color=colors[i],
                        label='Sim: Alpha = {}'.format(alpha), marker=markers[i], edgecolors='k', zorder=3,
                        s=sizes[i])
        plt.suptitle('Single orbit: Signal to Interference and Noise Ratio (SINR)',
                     fontsize=self.font_line_sizes['titleSize'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('SINR [dB]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        extension = 'n_sats_sinr_' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def capacity_vs_number_of_satellites_plot(self, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        math_results_df = self.math_results_df
        math_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                     self.orbital_input_params['n_sats_lims'][1],
                                     self.orbital_input_params['n_sats_resolution_math'])
        sim_results_df = self.sim_results_df
        sim_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        math_results_df = math_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(math_results_df)
        sim_results_df = sim_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(sim_results_df)

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list))

        fig = get_ax()
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            y_values = list(math_results_df.loc[(math_results_df['alpha'] == alpha), 'capacity_gbps'])
            plt.plot(math_n_sats_list, y_values, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)
        # Simulation
        for i, alpha in enumerate(alpha_list):
            y_value = list(sim_results_df.loc[(sim_results_df['alpha'] == alpha), 'capacity_gbps'])
            plt.scatter(sim_n_sats_list, y_value, color=colors[i],
                        label='Sim: Alpha = {}'.format(alpha), marker=markers[i], edgecolors='k', zorder=3,
                        s=sizes[i])

        plt.suptitle('Single orbit: Channel Capacity', fontsize=self.font_line_sizes['titleSize'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('Channel capacity [Gbps]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.yscale('log')
        plt.legend()
        extension = 'n_sats_capacity_' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def n_interferers_vs_number_of_satellites_plot(self, file_ext='.jpg', show=True, h=500e3):
        alpha_list = self.link_budget_input_params['alpha_list']
        math_results_df = self.math_results_df
        math_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                     self.orbital_input_params['n_sats_lims'][1],
                                     self.orbital_input_params['n_sats_resolution_math'])
        sim_results_df = self.sim_results_df
        sim_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        if not h:
            math_results_df = math_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
            adjust_db_columns(math_results_df)
            sim_results_df = sim_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
            adjust_db_columns(sim_results_df)

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=True)

        fig = get_ax()
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            if h:
                y_values = list(math_results_df.loc[
                                    (math_results_df['alpha'] == alpha) & (math_results_df['h'] == h), 'n_interferers'])
            else:
                y_values = list(math_results_df.loc[(math_results_df['alpha'] == alpha), 'n_interferers'])
            plt.plot(math_n_sats_list, y_values, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)
        # Simulation
        for i, alpha in enumerate(alpha_list):
            if h:
                y_values = list(sim_results_df.loc[
                                    (sim_results_df['alpha'] == alpha) & (sim_results_df['h'] == h), 'n_interferers'])
            else:
                y_values = list(sim_results_df.loc[(sim_results_df['alpha'] == alpha), 'n_interferers'])
            plt.scatter(sim_n_sats_list, y_values, color=colors[i],
                        label='Sim: Alpha = {}'.format(alpha), marker=markers[i], edgecolors='k', zorder=3,
                        s=sizes[i])

        plt.suptitle('Single orbit:\nNumber of satellites contributing to interference at the receiver',
                     fontsize=self.font_line_sizes['titleSize'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('Number of interferers', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        extension = 'n_sats_n_interferers_' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def distance_to_nearest_interferer_vs_number_of_satellites_plot(self, math=True, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        if math:
            results_df = self.math_results_df
            n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_math'])

        else:
            results_df = self.sim_results_df
            n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        results_df = results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(results_df)

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=True)

        fig = get_ax()
        plt.grid()
        for i, alpha in enumerate(alpha_list):
            y_values = results_df.loc[
                           (results_df['alpha'] == alpha), 'd_interferer'] * 1e-3
            plt.plot(n_sats_list, y_values, color=colors[i],
                     label='Alpha = {}'.format(alpha), markersize=5, markevery=None)
        if math:
            plt.suptitle('Single orbit math: Distance to closest interferer',
                         fontsize=self.font_line_sizes['titleSize'])
        else:
            plt.suptitle('Single orbit sim: Distance to closest interferer', fontsize=self.font_line_sizes['titleSize'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('Distance [km]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        if show:
            plt.show()
        if math:
            extension = 'n_sats_d_closes_interferer_' + 'math' + str(
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        else:
            extension = 'n_sats_d_closes_interferer_' + 'sim' + str(
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def single_sir1_and_sir2_vs_number_of_satellites_plot(self, file_ext='.jpg', alpha_list=None, show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(sim_results_df)

        # Number of satellites vectors
        math_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                     self.orbital_input_params['n_sats_lims'][1],
                                     self.orbital_input_params['n_sats_resolution_math'])

        sim_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list))

        fig = get_ax(figsize=(10 * 2, 6))

        # SIR1
        ax1 = plt.subplot(121)
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            interference = list(math_results_df.loc[(math_results_df['alpha'] == alpha), 'interference'])
            rx_power = list(math_results_df.loc[(math_results_df['alpha'] == alpha), 'rx_power'])
            y_values = [rx_power[i] - interference[i] if interference[i] else None for i in range(len(rx_power))]
            plt.plot(math_n_sats_list, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, alpha in enumerate(alpha_list):
            interference = list(sim_results_df.loc[(sim_results_df['alpha'] == alpha), 'interference'])
            rx_power = list(sim_results_df.loc[(sim_results_df['alpha'] == alpha), 'rx_power'])
            y_values = [rx_power[i] - interference[i] if interference[i] else None for i in range(len(rx_power))]
            plt.scatter(sim_n_sats_list, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax1.set_title('Single orbit: SIR 1', fontsize=self.font_line_sizes['titleSize'],
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel(r'SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'], fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'], fontname=self.font_line_sizes['fontname'])
        plt.legend(ncol=2, framealpha=0.7, fontsize=15, loc='upper right')
        plt.ylim([2.5, 10])

        # SIR2
        ax2 = plt.subplot(122)
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            sir_linear = list(math_results_df.loc[(math_results_df['alpha'] == alpha), 'sir_linear'])
            y_value = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(math_n_sats_list, y_value, color=colors[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, alpha in enumerate(alpha_list):
            sir_linear = list(sim_results_df.loc[(sim_results_df['alpha'] == alpha), 'sir_linear'])
            y_value = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(sim_n_sats_list, y_value, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax2.set_title('Single orbit: SIR 2', fontsize=self.font_line_sizes['titleSize'],
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel(r'SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'], fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'], fontname=self.font_line_sizes['fontname'])
        plt.legend(ncol=2, framealpha=0.7, fontsize=15, loc='upper right')
        plt.ylim([2.5, 10])

        extension = 'single_sir1_and_sir2_vs_n_satellites_' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def single_sir_snr_sinr_capacity_vs_number_of_satellites_plot(self, file_ext='.jpg', alpha_list=None, show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(sim_results_df)

        # Number of satellites vectors
        math_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                     self.orbital_input_params['n_sats_lims'][1],
                                     self.orbital_input_params['n_sats_resolution_math'])

        sim_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list))

        fig = get_ax(figsize=(10 * 2, 10))
        scale_factor = 0.8

        # SIR (2)
        ax1 = plt.subplot(221)
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            sir_linear = list(math_results_df.loc[(math_results_df['alpha'] == alpha), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(math_n_sats_list, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, alpha in enumerate(alpha_list):
            sir_linear = list(sim_results_df.loc[(sim_results_df['alpha'] == alpha), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(sim_n_sats_list, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax1.set_title('Single orbit: SIR 2', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel(r'SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        # plt.legend(ncol=1, framealpha=0.7, fontsize=self.font_line_sizes['legendSize']*scale_factor, loc='upper right')

        # SNR
        ax2 = plt.subplot(222)
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            snr_linear = list(math_results_df.loc[(math_results_df['alpha'] == alpha), 'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.plot(math_n_sats_list, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, alpha in enumerate(alpha_list):
            snr_linear = list(sim_results_df.loc[(sim_results_df['alpha'] == alpha), 'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.scatter(sim_n_sats_list, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax2.set_title('Single orbit: SNR', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel(r'SNR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        # plt.legend(ncol=2, framealpha=0.7, fontsize=self.font_line_sizes['legendSize']*scale_factor, loc='upper right')
        # plt.ylim([2.5, 10])

        # SINR
        ax3 = plt.subplot(223)
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            sinr_linear = list(math_results_df.loc[(math_results_df['alpha'] == alpha), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(math_n_sats_list, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, alpha in enumerate(alpha_list):
            sinr_linear = list(sim_results_df.loc[(sim_results_df['alpha'] == alpha), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.scatter(sim_n_sats_list, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax3.set_title('Single orbit: SINR', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel(r'SINR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xlim([950, 1000])
        plt.ylim([1.5, 3])

        # Capacity
        ax4 = plt.subplot(224)
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            y_values = list(math_results_df.loc[(math_results_df['alpha'] == alpha), 'capacity_gbps'])
            plt.plot(math_n_sats_list, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, alpha in enumerate(alpha_list):
            y_values = list(sim_results_df.loc[(sim_results_df['alpha'] == alpha), 'capacity_gbps'])
            plt.scatter(sim_n_sats_list, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax4.set_title('Single orbit: Capacity', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel(r'Channel Capacity [Gbps]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yscale('log')
        plt.legend(ncol=2, framealpha=0.7, fontsize=self.font_line_sizes['legendSize'] * scale_factor,
                   loc='upper right')

        extension = 'single_sir_snr_sinr_capacity_vs_n_satellites_' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def single_sir_snr_sinr_capacity_vs_number_of_satellites_plot_h(self, file_ext='.jpg', alpha_list=None, h=500e3,
                                                                    show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list

        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df

        # Number of satellites vectors
        math_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                     self.orbital_input_params['n_sats_lims'][1],
                                     self.orbital_input_params['n_sats_resolution_math'])

        sim_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list))

        fig = get_ax(figsize=(10 * 2, 10))
        scale_factor = 0.8

        # SIR (2)
        ax1 = plt.subplot(221)
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            sir_linear = list(
                math_results_df.loc[(math_results_df['alpha'] == alpha) & (math_results_df['h'] == h), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(math_n_sats_list, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, alpha in enumerate(alpha_list):
            sir_linear = list(
                sim_results_df.loc[(sim_results_df['alpha'] == alpha) & (sim_results_df['h'] == h), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(sim_n_sats_list, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax1.set_title('Single orbit: SIR 2', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel(r'SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        # plt.legend(ncol=1, framealpha=0.7, fontsize=self.font_line_sizes['legendSize']*scale_factor, loc='upper right')

        # SNR
        ax2 = plt.subplot(222)
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            snr_linear = list(
                math_results_df.loc[(math_results_df['alpha'] == alpha) & (math_results_df['h'] == h), 'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.plot(math_n_sats_list, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, alpha in enumerate(alpha_list):
            snr_linear = list(
                sim_results_df.loc[(sim_results_df['alpha'] == alpha) & (sim_results_df['h'] == h), 'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.scatter(sim_n_sats_list, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax2.set_title('Single orbit: SNR', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel(r'SNR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        # plt.legend(ncol=2, framealpha=0.7, fontsize=self.font_line_sizes['legendSize']*scale_factor, loc='upper right')
        # plt.ylim([2.5, 10])

        # SINR
        ax3 = plt.subplot(223)
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            sinr_linear = list(
                math_results_df.loc[(math_results_df['alpha'] == alpha) & (math_results_df['h'] == h), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(math_n_sats_list, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, alpha in enumerate(alpha_list):
            sinr_linear = list(
                sim_results_df.loc[(sim_results_df['alpha'] == alpha) & (sim_results_df['h'] == h), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.scatter(sim_n_sats_list, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax3.set_title('Single orbit: SINR', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel(r'SINR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])

        # Capacity
        ax4 = plt.subplot(224)
        plt.grid()
        # Math
        for i, alpha in enumerate(alpha_list):
            y_values = list(
                math_results_df.loc[(math_results_df['alpha'] == alpha) & (math_results_df['h'] == h), 'capacity_gbps'])
            plt.plot(math_n_sats_list, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, alpha in enumerate(alpha_list):
            y_values = list(
                sim_results_df.loc[(sim_results_df['alpha'] == alpha) & (sim_results_df['h'] == h), 'capacity_gbps'])
            plt.scatter(sim_n_sats_list, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax4.set_title('Single orbit: Capacity', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel(r'Channel Capacity [Gbps]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yscale('log')
        plt.legend(ncol=2, framealpha=0.7, fontsize=self.font_line_sizes['legendSize'] * scale_factor,
                   loc='upper right')

        extension = 'single_sir_snr_sinr_capacity_vs_n_satellites_' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    # X axis: beamwidth
    def sir_vs_beamwidth(self, file_ext='.jpg', show=True, h=500e3, n_sats_list=None):
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        math_alpha_list = np.arange(self.orbital_input_params['alpha_lims'][0],
                                    self.orbital_input_params['alpha_lims'][1],
                                    self.orbital_input_params['alpha_resolution_math'])
        math_results_df = self.math_results_df
        sim_alpha_list = np.arange(self.orbital_input_params['alpha_lims'][0],
                                   self.orbital_input_params['alpha_lims'][1],
                                   self.orbital_input_params['alpha_resolution_sim'])
        sim_results_df = self.sim_results_df

        _, _, colors = tk.templateColors(iterable_size=len(n_sats_list), reverse=True)

        fig = get_ax()
        plt.grid()
        # Math
        for i, n_sats in enumerate(n_sats_list):
            sir_linear = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(math_alpha_list, y_values, color=colors[i],
                     label='N = {}'.format(n_sats), markersize=5, markevery=None)
        # Simulation
        for i, n_sats in enumerate(n_sats_list):
            sir_linear = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h), 'sir_linear'])
            y_value = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(sim_alpha_list, y_value, color=colors[i],
                        label='Sim: N = {}'.format(n_sats), marker=markers[i], edgecolors='k', zorder=3,
                        s=sizes[i])
        plt.ylim([0, 6.1])
        plt.vlines([1, 2, 5, 10, 20, 30], 0, 6.1, color='k', linestyles='dotted')

        plt.suptitle('Single orbit math: Signal to Interference Ratio (SIR)',
                     fontsize=self.font_line_sizes['titleSize'])
        plt.xlabel(r'Beamwidth $\alpha$ [$^\circ$]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.legend()
        extension = 'journal_beamwidth_sir_' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    # Paper plots
    def sir_vs_number_of_satellites_math_and_sim_plot(self, file_ext='.jpg', show=True):
        alpha_list = self.link_budget_input_params['alpha_list']
        h_plots = self.orbital_input_params['h_lims'][0]
        h_plots_list = [h_plots]
        n_sats_list_math = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                     self.orbital_input_params['n_sats_lims'][1],
                                     self.orbital_input_params['n_sats_resolution_math'])
        n_sats_list_sim = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=False)

        fig = get_ax()
        plt.grid()
        pairs = list(product(alpha_list, h_plots_list))
        # Math
        for i, (alpha, h) in enumerate(pairs):
            color = colors[i]
            interference = list(
                math_results_df.loc[(math_results_df['alpha'] == alpha) & (math_results_df['h'] == h), 'interference'])
            rx_power = list(
                math_results_df.loc[(math_results_df['alpha'] == alpha) & (math_results_df['h'] == h), 'rx_power'])
            SIR = [rx_power[i] - interference[i] if interference[i] else None for i in
                   range(len(rx_power))]
            plt.plot(n_sats_list_math, SIR, color=color,
                     label=r'$\alpha$={}$^\circ$'.format(alpha), linewidth=3)

        # Simulation (done separately so markers display on top of lines)
        for i, (alpha, h) in enumerate(pairs):
            color = colors[i]
            interference = list(
                sim_results_df.loc[(sim_results_df['alpha'] == alpha) & (sim_results_df['h'] == h), 'interference'])
            rx_power = list(
                sim_results_df.loc[(sim_results_df['alpha'] == alpha) & (sim_results_df['h'] == h), 'rx_power'])
            SIR = [rx_power[i] - interference[i] if interference[i] else None for i in
                   range(len(rx_power))]
            # plt.plot(n_sats_list_sims, SIR, markers2[i], color=color,
            #            label=r'Sim. $\alpha$={}$^\circ$'.format(alpha), markersize=sizes2[i])

            plt.scatter(n_sats_list_sim, SIR, color=color,
                        label=r'Sim. $\alpha$={}$^\circ$'.format(alpha), marker=markers[i], edgecolors='k', zorder=3,
                        s=sizes[i])

        # plt.title('Single orbit: Signal to Interference Ratio (SIR)', fontsize=18, fontname='Helvetica')
        plt.xlabel('Number of satellites in orbit', fontsize=25, fontname="Times New Roman")
        plt.ylabel('SIR [dB]', fontsize=25, fontname="Times New Roman")
        plt.xticks([20, 40, 60, 80, 100], fontsize=20, fontname="Times New Roman")
        plt.yticks(fontsize=20, fontname="Times New Roman")
        # plt.xlim([20, 100])
        plt.ylim([1, 6.5])
        plt.legend(ncol=4, framealpha=1, fontsize=15, loc='lower left')
        extension = 'single_orbit_SIR' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def sir_snr_sinr_capacity_vs_number_of_satellites(self, file_ext='.jpg', show=True, h=500e3,
                                                      thz_alpha_list=None, mmWave_alpha_list=None):
        thz_alpha_list = self.thz_input_params['alpha_list'] if thz_alpha_list is None else thz_alpha_list
        mmWave_alpha_list = self.mmWave_input_params['alpha_list'] if mmWave_alpha_list is None else mmWave_alpha_list
        h_plots = self.orbital_input_params['h_lims'][0]
        h_plots_list = [h_plots]
        math_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                     self.orbital_input_params['n_sats_lims'][1],
                                     self.orbital_input_params['n_sats_resolution_math'])
        sim_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        thz_math_results_df = self.thz_math_results_df
        thz_sim_results_df = self.thz_sim_results_df
        mmWave_math_results_df = self.mmWave_math_results_df
        mmWave_sim_results_df = self.mmWave_sim_results_df

        line_styles = ['solid', 'solid', 'dashed', 'dotted']
        mmWave_colors = ['g', 'r', 'r', 'r']

        fig = get_ax(figsize=(10 * 2, 10))
        scale_factor = 0.8

        # SIR (2)
        ax1 = plt.subplot(221)
        plt.grid()
        # THz Math
        for i, alpha in enumerate(thz_alpha_list):
            sir_linear = list(
                thz_math_results_df.loc[
                    (thz_math_results_df['alpha'] == alpha) & (thz_math_results_df['h'] == h), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(math_n_sats_list, y_values, color='b', linestyle=line_styles[i],
                     label=r'THz: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # mmWave Math
        for i, alpha in enumerate(mmWave_alpha_list):
            sir_linear = list(
                mmWave_math_results_df.loc[
                    (mmWave_math_results_df['alpha'] == alpha) & (mmWave_math_results_df['h'] == h), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(math_n_sats_list, y_values, color=mmWave_colors[i], linestyle=line_styles[i],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])

        # THz Simulation
        for i, alpha in enumerate(thz_alpha_list):
            sir_linear = list(
                thz_sim_results_df.loc[
                    (thz_sim_results_df['alpha'] == alpha) & (thz_sim_results_df['h'] == h), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(sim_n_sats_list, y_values, color='b',
                        label=r'THz sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])
        # mmWave Simulation
        for i, alpha in enumerate(mmWave_alpha_list):
            sir_linear = list(
                mmWave_sim_results_df.loc[
                    (mmWave_sim_results_df['alpha'] == alpha) & (mmWave_sim_results_df['h'] == h), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(sim_n_sats_list, y_values, color=mmWave_colors[i],
                        label=r'mmWave sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax1.set_title('Single orbit: SIR 2', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel(r'SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        # plt.legend(ncol=1, framealpha=0.7, fontsize=self.font_line_sizes['legendSize']*scale_factor, loc='upper right')

        # SNR
        ax2 = plt.subplot(222)
        plt.grid()
        # THz Math
        for i, alpha in enumerate(thz_alpha_list):
            snr_linear = list(
                thz_math_results_df.loc[
                    (thz_math_results_df['alpha'] == alpha) & (thz_math_results_df['h'] == h), 'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.plot(math_n_sats_list, y_values, color='b', linestyle=line_styles[i],
                     label=r'THz: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # mmWave Math
        for i, alpha in enumerate(mmWave_alpha_list):
            snr_linear = list(
                mmWave_math_results_df.loc[
                    (mmWave_math_results_df['alpha'] == alpha) & (mmWave_math_results_df['h'] == h), 'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.plot(math_n_sats_list, y_values, color=mmWave_colors[i], linestyle=line_styles[i],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])

        # THz Simulation
        for i, alpha in enumerate(thz_alpha_list):
            snr_linear = list(
                thz_sim_results_df.loc[
                    (thz_sim_results_df['alpha'] == alpha) & (thz_sim_results_df['h'] == h), 'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.scatter(sim_n_sats_list, y_values, color='b',
                        label=r'THz sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])
        # mmWave Simulation
        for i, alpha in enumerate(mmWave_alpha_list):
            snr_linear = list(
                mmWave_sim_results_df.loc[
                    (mmWave_sim_results_df['alpha'] == alpha) & (mmWave_sim_results_df['h'] == h), 'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.scatter(sim_n_sats_list, y_values, color=mmWave_colors[i],
                        label=r'mmWave sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax2.set_title('Single orbit: SNR', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel(r'SNR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        # plt.legend(ncol=2, framealpha=0.7, fontsize=self.font_line_sizes['legendSize']*scale_factor, loc='upper right')

        # SINR
        ax3 = plt.subplot(223)
        plt.grid()
        # THz Math
        for i, alpha in enumerate(thz_alpha_list):
            sinr_linear = list(
                thz_math_results_df.loc[
                    (thz_math_results_df['alpha'] == alpha) & (thz_math_results_df['h'] == h), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(math_n_sats_list, y_values, color='b', linestyle=line_styles[i],
                     label=r'THz: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # mmWave Math
        for i, alpha in enumerate(mmWave_alpha_list):
            sinr_linear = list(
                mmWave_math_results_df.loc[
                    (mmWave_math_results_df['alpha'] == alpha) & (mmWave_math_results_df['h'] == h), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(math_n_sats_list, y_values, color=mmWave_colors[i], linestyle=line_styles[i],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])

        # THz Simulation
        for i, alpha in enumerate(thz_alpha_list):
            sinr_linear = list(
                thz_sim_results_df.loc[
                    (thz_sim_results_df['alpha'] == alpha) & (thz_sim_results_df['h'] == h), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.scatter(sim_n_sats_list, y_values, color='b',
                        label=r'THz sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])
        # mmWave Simulation
        for i, alpha in enumerate(mmWave_alpha_list):
            sinr_linear = list(
                mmWave_sim_results_df.loc[
                    (mmWave_sim_results_df['alpha'] == alpha) & (mmWave_sim_results_df['h'] == h), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.scatter(sim_n_sats_list, y_values, color=mmWave_colors[i],
                        label=r'mmWave sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax3.set_title('Single orbit: SINR', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel(r'SINR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])

        # Capacity
        ax4 = plt.subplot(224)
        plt.grid()
        # THz Math
        for i, alpha in enumerate(thz_alpha_list):
            y_values = list(
                thz_math_results_df.loc[
                    (thz_math_results_df['alpha'] == alpha) & (thz_math_results_df['h'] == h), 'capacity_gbps'])
            plt.plot(math_n_sats_list, y_values, color='b', linestyle=line_styles[i],
                     label=r'THz: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])
        # mmWave Math
        for i, alpha in enumerate(mmWave_alpha_list):
            y_values = list(
                mmWave_math_results_df.loc[
                    (mmWave_math_results_df['alpha'] == alpha) & (mmWave_math_results_df['h'] == h), 'capacity_gbps'])
            plt.plot(math_n_sats_list, y_values, color=mmWave_colors[i], linestyle=line_styles[i],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])

        # THz Simulation
        for i, alpha in enumerate(thz_alpha_list):
            y_values = list(
                thz_sim_results_df.loc[
                    (thz_sim_results_df['alpha'] == alpha) & (thz_sim_results_df['h'] == h), 'capacity_gbps'])
            plt.scatter(sim_n_sats_list, y_values, color='b',
                        label=r'THz sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])
        # mmWave Simulation
        for i, alpha in enumerate(mmWave_alpha_list):
            y_values = list(
                mmWave_sim_results_df.loc[
                    (mmWave_sim_results_df['alpha'] == alpha) & (mmWave_sim_results_df['h'] == h), 'capacity_gbps'])
            plt.scatter(sim_n_sats_list, y_values, color=mmWave_colors[i],
                        label=r'mmWave sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax4.set_title('Single orbit: Capacity', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel(r'Channel Capacity [Gbps]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yscale('log')
        plt.legend(ncol=2, framealpha=0.7, fontsize=self.font_line_sizes['legendSize'] * scale_factor,
                   loc='upper right')

        extension = 'journal_single_sir_snr_sinr_capacity_vs_n_satellites_' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    # Journal paper plots
    def journal_single_sir_vs_number_of_satellites(self, file_ext='.jpg', show=True, alpha_list=None):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        h_plots = self.orbital_input_params['h_lims'][0]
        h_plots_list = [h_plots]
        math_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                     self.orbital_input_params['n_sats_lims'][1],
                                     self.orbital_input_params['n_sats_resolution_math'])
        sim_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list), reverse=False)

        fig = get_ax()
        plt.grid()
        pairs = list(product(alpha_list, h_plots_list))

        # Horizontal asymptote at the limit where N-->inf
        plt.hlines(1.9, 0, 200, colors='k', linestyles='dashed', linewidth=self.font_line_sizes['linewidth'])

        # Math
        for i, (alpha, h) in enumerate(pairs):
            color = colors[i]
            sir_linear = math_results_df.loc[
                (math_results_df['alpha'] == alpha) & (math_results_df['h'] == h), 'sir_linear']
            # Find last Nan element to insert 'infinite' SIR
            sir_linear = sir_linear.reset_index(drop=True)
            j = sir_linear.first_valid_index()
            sir_linear = list(sir_linear)
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            # Insert 'infinite' SIR
            y_values[j] = 300
            plt.plot(math_n_sats_list, y_values, color=color, label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])

        # Simulation (done separately so markers display on top of lines)
        for i, (alpha, h) in enumerate(pairs):
            color = colors[i]
            sir_linear = list(
                sim_results_df.loc[(sim_results_df['alpha'] == alpha) & (sim_results_df['h'] == h), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(sim_n_sats_list, y_values, color=color,
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        # plt.title('Single orbit: Signal to Interference Ratio (SIR)', fontsize=18, fontname='Helvetica')
        plt.xlabel('Satellites per orbit', fontsize=self.font_line_sizes['titleSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylim([0, 6.5])
        plt.xlim([10, 200])
        plt.xticks(fontsize=self.font_line_sizes['axisLabelSize'], fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['axisLabelSize'], fontname=self.font_line_sizes['fontname'])
        plt.legend(ncol=4, framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='lower left', columnspacing=0.2)

        # Adjustemnt of bbox to better fit figures 5b and 5c which have the legent outside of the bbox
        box = fig.get_position()
        fig.set_position([box.x0 * 0.9, box.y0 * 0.9, box.width * 1.1, box.height * 1.1])

        extension = 'journal_single_1_SIR_vs_n_sats' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def journal_single_sinr_vs_number_of_satellites(self, file_ext='.jpg', show=True, thz_alpha_list=None,
                                                    mmwave_alpha_list=None):
        thz_alpha_list = self.thz_input_params['alpha_list'] if thz_alpha_list is None else thz_alpha_list
        mmwave_alpha_list = self.mmWave_input_params['alpha_list'] if mmwave_alpha_list is None else mmwave_alpha_list
        h = self.orbital_input_params['h_lims'][0]
        h_plots_list = [h]
        math_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                     self.orbital_input_params['n_sats_lims'][1],
                                     self.orbital_input_params['n_sats_resolution_math'])
        sim_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        thz_math_results_df = self.thz_math_results_df
        thz_sim_results_df = self.thz_sim_results_df
        mmwave_math_results_df = self.mmWave_math_results_df
        mmwave_sim_results_df = self.mmWave_sim_results_df

        line_styles = ['solid', 'dashdot', 'dotted', 'solid']
        _, _, colors = tk.templateColors(iterable_size=2, reverse=False)

        ax = get_ax(figsize=(10, 6))
        plt.grid()

        # Horizontal asymptote at the limit where N-->inf
        plt.hlines(1.9, 0, 200, colors='k', linestyles='dashed', linewidth=self.font_line_sizes['linewidth'])

        # Theoretical maximum (SNR) as a dashed line, from the math model
        alpha = min(mmwave_alpha_list)
        snr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha) & (
                mmwave_math_results_df['h'] == h), 'snr_linear'])
        y_values = [10 * np.log10(value) if value else None for value in snr_linear]
        plt.plot(math_n_sats_list, y_values, color='dimgray', linestyle='dashed',
                 label=None,
                 linewidth=self.font_line_sizes['linewidth'])

        # THz Math
        for i, alpha in enumerate(thz_alpha_list):
            sinr_linear = list(
                thz_math_results_df.loc[
                    (thz_math_results_df['alpha'] == alpha) & (thz_math_results_df['h'] == h), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(math_n_sats_list, y_values, color=colors[1], linestyle=line_styles[i],
                     label=r'THz: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])

        # THz Simulation
        for i, alpha in enumerate(thz_alpha_list):
            sinr_linear = list(
                thz_sim_results_df.loc[
                    (thz_sim_results_df['alpha'] == alpha) & (thz_sim_results_df['h'] == h), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.scatter(sim_n_sats_list, y_values, color=colors[1],
                        label=r'THz sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        # mmWave Math
        for i, alpha in enumerate(mmwave_alpha_list):
            sinr_linear = list(
                mmwave_math_results_df.loc[
                    (mmwave_math_results_df['alpha'] == alpha) & (mmwave_math_results_df['h'] == h), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(math_n_sats_list, y_values, color=colors[0], linestyle=line_styles[i],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])

        # mmWave Simulation
        for i, alpha in enumerate(mmwave_alpha_list):
            sinr_linear = list(
                mmwave_sim_results_df.loc[
                    (mmwave_sim_results_df['alpha'] == alpha) & (mmwave_sim_results_df['h'] == h), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.scatter(sim_n_sats_list, y_values, color=colors[0],
                        label=r'mmWave sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        # plt.title('Single orbit: SINR', fontsize=self.font_line_sizes['titleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Satellites per orbit', fontsize=self.font_line_sizes['axisLabelSize']* 1.2)
        plt.ylabel(r'SINR [dB]', fontsize=self.font_line_sizes['axisLabelSize']* 1.2,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize']* 1.2,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize']*1.2,
                   fontname=self.font_line_sizes['fontname'])
        plt.xlim(0, 200)

        '''
        # Version 1: Legend to the right
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='center left', bbox_to_anchor=(1, 0.5))
        '''

        # # Version 2: Legend below
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.3, box.width, box.height * 0.85])
        # plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize']*1.05, loc='upper center',
        #            bbox_to_anchor=(0.485, -0.2), ncol=4, fancybox=True, columnspacing=0.25, handletextpad=0.3)

        # Version 3: Crunched legend
        h, l = ax.get_legend_handles_labels()
        ph = [plt.plot([], marker="", ls="")[0]] * 4
        handles = ph[:1] + h[:3] + ph[1:2] + h[3:6] + ph[2:3] + h[6:9] + ph[3:] + h[9:]
        labels = ["Sub-THz:"] + l[:3] + [""] + l[3:6] + ["mmWave"] + l[6:9] + [""] + l[9:]
        plt.legend(handles, labels, ncol=4, fontsize=self.font_line_sizes['legendSize']*1.05)

        extension = 'journal_single_2_SINR_vs_n_sats' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def journal_single_sinr_vs_number_of_satellites_v2(self, file_ext='.jpg', show=True, thz_alpha_list=None,
                                                    mmwave_alpha_list=None):
        thz_alpha_list = self.thz_input_params['alpha_list'] if thz_alpha_list is None else thz_alpha_list
        mmwave_alpha_list = self.mmWave_input_params['alpha_list'] if mmwave_alpha_list is None else mmwave_alpha_list
        h = self.orbital_input_params['h_lims'][0]
        h_plots_list = [h]
        math_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                     self.orbital_input_params['n_sats_lims'][1],
                                     self.orbital_input_params['n_sats_resolution_math'])
        sim_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        thz_math_results_df = self.thz_math_results_df
        thz_sim_results_df = self.thz_sim_results_df
        mmwave_math_results_df = self.mmWave_math_results_df
        mmwave_sim_results_df = self.mmWave_sim_results_df

        line_styles = ['solid', 'dashdot', 'dotted', 'solid']
        _, _, colors = tk.templateColors(iterable_size=2, reverse=False)

        ax = get_ax(figsize=(10, 6))
        plt.grid()

        # Horizontal asymptote at the limit where N-->inf
        plt.hlines(1.9, 0, 200, colors='k', linestyles='dashed', linewidth=self.font_line_sizes['linewidth'])

        # Theoretical maximum (SNR) as a dashed line, from the math model
        alpha = min(mmwave_alpha_list)
        snr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha) & (
                mmwave_math_results_df['h'] == h), 'snr_linear'])
        y_values = [10 * np.log10(value) if value else None for value in snr_linear]
        plt.plot(math_n_sats_list, y_values, color='dimgray', linestyle='dashed',
                 label=None,
                 linewidth=self.font_line_sizes['linewidth'])

        # THz Math
        for i, alpha in enumerate(thz_alpha_list):
            sinr_linear = list(
                thz_math_results_df.loc[
                    (thz_math_results_df['alpha'] == alpha) & (thz_math_results_df['h'] == h), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(math_n_sats_list, y_values, color=colors[1], linestyle=line_styles[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])


        # THz Simulation
        for i, alpha in enumerate(thz_alpha_list):
            sinr_linear = list(
                thz_sim_results_df.loc[
                    (thz_sim_results_df['alpha'] == alpha) & (thz_sim_results_df['h'] == h), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.scatter(sim_n_sats_list, y_values, color=colors[1],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        # mmWave Math
        for i, alpha in enumerate(mmwave_alpha_list):
            sinr_linear = list(
                mmwave_math_results_df.loc[
                    (mmwave_math_results_df['alpha'] == alpha) & (mmwave_math_results_df['h'] == h), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(math_n_sats_list, y_values, color=colors[0], linestyle=line_styles[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])

        # mmWave Simulation
        for i, alpha in enumerate(mmwave_alpha_list):
            sinr_linear = list(
                mmwave_sim_results_df.loc[
                    (mmwave_sim_results_df['alpha'] == alpha) & (mmwave_sim_results_df['h'] == h), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.scatter(sim_n_sats_list, y_values, color=colors[0],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        fontsize_scale_factor = 1.1
        # plt.title('Single orbit: SINR', fontsize=self.font_line_sizes['titleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Satellites per orbit', fontsize=self.font_line_sizes['axisLabelSize'] * fontsize_scale_factor)
        plt.ylabel(r'SINR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * fontsize_scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks([10, 25, 50, 75, 100, 125, 150, 175, 200], fontsize=self.font_line_sizes['ticksLabelSize'] * fontsize_scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * fontsize_scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xlim(0, 200)
        plt.ylim(-70, 50)

        '''
        # Version 1: Legend to the right
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='center left', bbox_to_anchor=(1, 0.5))
        '''

        # # Version 2: Legend below
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.3, box.width, box.height * 0.85])
        # plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize']*1.05, loc='upper center',
        #            bbox_to_anchor=(0.485, -0.2), ncol=4, fancybox=True, columnspacing=0.25, handletextpad=0.3)

        # Version 3: Crunched legend
        box = ax.get_position()
        ax.set_position([box.x0 * 0.9, box.y0 * 0.9, box.width * 1.1, box.height * 1.1])
        h, l = ax.get_legend_handles_labels()
        ph = [plt.scatter([], [], marker="o", color=colors[1], s=100),
              plt.scatter([], [], marker=""),
              plt.scatter([], [], marker="o", color=colors[0], s=100),
              plt.scatter([], [], marker="")]
        handles = ph[:1] + h[:3] + ph[1:2] + h[3:6] + ph[2:3] + h[6:9] + ph[3:] + h[9:]
        labels = ["Sub-THz:"] + l[:3] + [""] + l[3:6] + ["mmWave:"] + l[6:9] + [""] + l[9:]
        leg = plt.legend(handles, labels, ncol=4, fontsize=self.font_line_sizes['legendSize'] * 1.05, columnspacing=0.5,
                         handletextpad=0.1, loc='lower right', framealpha=1)

        extension = 'journal_single_2_SINR_vs_n_sats' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def journal_single_capacity_vs_number_of_satellites(self, file_ext='.jpg', show=True, thz_alpha_list=None,
                                                        mmwave_alpha_list=None):
        thz_alpha_list = self.thz_input_params['alpha_list'] if thz_alpha_list is None else thz_alpha_list
        mmwave_alpha_list = self.mmWave_input_params['alpha_list'] if mmwave_alpha_list is None else mmwave_alpha_list
        bw_mmwave = self.mmWave_input_params['bandwidth']
        bw_thz = self.thz_input_params['bandwidth']
        h = self.orbital_input_params['h_lims'][0]
        h_plots_list = [h]
        math_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                     self.orbital_input_params['n_sats_lims'][1],
                                     self.orbital_input_params['n_sats_resolution_math'])
        sim_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        thz_math_results_df = self.thz_math_results_df
        thz_sim_results_df = self.thz_sim_results_df
        mmwave_math_results_df = self.mmWave_math_results_df
        mmwave_sim_results_df = self.mmWave_sim_results_df

        line_styles = ['solid', 'dashed', 'dotted', 'solid']
        _, _, colors = tk.templateColors(iterable_size=2, reverse=False)

        ax = get_ax()
        plt.grid()

        # Horizontal asymptote at the limit where N-->inf
        plt.hlines(bw_mmwave * np.log2(1 + 1.9) * 1e-9, 0, 200, colors='k', linestyles='dashed',
                   linewidth=self.font_line_sizes['linewidth'])
        print(bw_mmwave * np.log2(1 + 1.9) * 1e-9)
        plt.hlines(bw_thz * np.log2(1 + 1.9) * 1e-9, 0, 200, colors='k', linestyles='dashed',
                   linewidth=self.font_line_sizes['linewidth'])
        print(bw_thz * np.log2(1 + 1.9) * 1e-9)

        # Theoretical maximum (B*log2(SNR)) as a dashed line, from the math model
        alpha = min(mmwave_alpha_list)
        snr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha) & (
                mmwave_math_results_df['h'] == h), 'snr_linear'])
        y_values = [bw_mmwave * np.log2(1 + value) * 1e-9 if value != 0 else None for value in snr_linear]
        plt.plot(math_n_sats_list, y_values, color='dimgray', linestyle='dashed',
                 label=None, linewidth=self.font_line_sizes['linewidth'])

        # THz Math
        for i, alpha in enumerate(thz_alpha_list):
            y_values = list(
                thz_math_results_df.loc[
                    (thz_math_results_df['alpha'] == alpha) & (thz_math_results_df['h'] == h), 'capacity_gbps'])
            plt.plot(math_n_sats_list, y_values, color=colors[1], linestyle=line_styles[i],
                     label=r'THz: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])

        # mmWave Math
        for i, alpha in enumerate(mmwave_alpha_list):
            y_values = list(
                mmwave_math_results_df.loc[
                    (mmwave_math_results_df['alpha'] == alpha) & (
                            mmwave_math_results_df['h'] == h), 'capacity_gbps'])
            plt.plot(math_n_sats_list, y_values, color=colors[0], linestyle=line_styles[i],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])

        # THz Simulation
        for i, alpha in enumerate(thz_alpha_list):
            y_values = list(
                thz_sim_results_df.loc[
                    (thz_sim_results_df['alpha'] == alpha) & (thz_sim_results_df['h'] == h), 'capacity_gbps'])
            plt.scatter(sim_n_sats_list, y_values, color=colors[1],
                        label=r'THz sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        # mmWave Simulation
        for i, alpha in enumerate(mmwave_alpha_list):
            y_values = list(
                mmwave_sim_results_df.loc[
                    (mmwave_sim_results_df['alpha'] == alpha) & (mmwave_sim_results_df['h'] == h), 'capacity_gbps'])
            plt.scatter(sim_n_sats_list, y_values, color=colors[0],
                        label=r'mmWave sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        # plt.title('Single orbit: SINR', fontsize=self.font_line_sizes['titleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Satellites per orbit', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel(r'Channel capacity [Gbps]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yscale('log')
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks([0.01, 0.1, 1, 10, 100],
                   [0.01, 0.1, 1, 10, 100],
                   fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])

        plt.xlim(0, 200)

        '''
        # Version 1: Legend to the right
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='center left', bbox_to_anchor=(1, 0.5))
        '''

        # Version 2: Legend below
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.9])
        plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='upper center',
                   bbox_to_anchor=(0.47, -0.15), ncol=4, fancybox=True, columnspacing=0.25, handletextpad=0.3)

        extension = 'journal_single_3_capacity_vs_n_sats' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def journal_single_capacity_vs_number_of_satellites_v2(self, file_ext='.jpg', show=True, thz_alpha_list=None,
                                                        mmwave_alpha_list=None):
        thz_alpha_list = self.thz_input_params['alpha_list'] if thz_alpha_list is None else thz_alpha_list
        mmwave_alpha_list = self.mmWave_input_params['alpha_list'] if mmwave_alpha_list is None else mmwave_alpha_list
        bw_mmwave = self.mmWave_input_params['bandwidth']
        bw_thz = self.thz_input_params['bandwidth']
        h = self.orbital_input_params['h_lims'][0]
        h_plots_list = [h]
        math_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                     self.orbital_input_params['n_sats_lims'][1],
                                     self.orbital_input_params['n_sats_resolution_math'])
        sim_n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                    self.orbital_input_params['n_sats_lims'][1],
                                    self.orbital_input_params['n_sats_resolution_sim'])

        thz_math_results_df = self.thz_math_results_df
        thz_sim_results_df = self.thz_sim_results_df
        mmwave_math_results_df = self.mmWave_math_results_df
        mmwave_sim_results_df = self.mmWave_sim_results_df

        line_styles = ['solid', 'dashed', 'dotted', 'solid']
        _, _, colors = tk.templateColors(iterable_size=2, reverse=False)

        ax = get_ax()
        plt.grid()

        # Horizontal asymptote at the limit where N-->inf
        plt.hlines(bw_mmwave * np.log2(1 + 1.9) * 1e-9, 0, 200, colors='k', linestyles='dashed',
                   linewidth=self.font_line_sizes['linewidth'])
        print(bw_mmwave * np.log2(1 + 1.9) * 1e-9)
        plt.hlines(bw_thz * np.log2(1 + 1.9) * 1e-9, 0, 200, colors='k', linestyles='dashed',
                   linewidth=self.font_line_sizes['linewidth'])
        print(bw_thz * np.log2(1 + 1.9) * 1e-9)

        # Theoretical maximum (B*log2(SNR)) as a dashed line, from the math model
        alpha = min(mmwave_alpha_list)
        snr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha) & (
                mmwave_math_results_df['h'] == h), 'snr_linear'])
        y_values = [bw_mmwave * np.log2(1 + value) * 1e-9 if value != 0 else None for value in snr_linear]
        plt.plot(math_n_sats_list, y_values, color='dimgray', linestyle='dashed',
                 label=None, linewidth=self.font_line_sizes['linewidth'])

        # THz Math
        for i, alpha in enumerate(thz_alpha_list):
            y_values = list(
                thz_math_results_df.loc[
                    (thz_math_results_df['alpha'] == alpha) & (thz_math_results_df['h'] == h), 'capacity_gbps'])
            plt.plot(math_n_sats_list, y_values, color=colors[1], linestyle=line_styles[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])


        # THz Simulation
        for i, alpha in enumerate(thz_alpha_list):
            y_values = list(
                thz_sim_results_df.loc[
                    (thz_sim_results_df['alpha'] == alpha) & (thz_sim_results_df['h'] == h), 'capacity_gbps'])
            plt.scatter(sim_n_sats_list, y_values, color=colors[1],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        # mmWave Math
        for i, alpha in enumerate(mmwave_alpha_list):
            y_values = list(
                mmwave_math_results_df.loc[
                    (mmwave_math_results_df['alpha'] == alpha) & (
                            mmwave_math_results_df['h'] == h), 'capacity_gbps'])
            plt.plot(math_n_sats_list, y_values, color=colors[0], linestyle=line_styles[i],
                     label=r'$\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'])

        # mmWave Simulation
        for i, alpha in enumerate(mmwave_alpha_list):
            y_values = list(
                mmwave_sim_results_df.loc[
                    (mmwave_sim_results_df['alpha'] == alpha) & (mmwave_sim_results_df['h'] == h), 'capacity_gbps'])
            plt.scatter(sim_n_sats_list, y_values, color=colors[0],
                        label=r'Sim: $\alpha$={}$^\circ$'.format(alpha), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        fontsize_scale_factor = 1.1
        # plt.title('Single orbit: SINR', fontsize=self.font_line_sizes['titleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Satellites per orbit', fontsize=self.font_line_sizes['axisLabelSize'] * fontsize_scale_factor)
        plt.ylabel(r'Capacity [Gbps]', fontsize=self.font_line_sizes['axisLabelSize'] * fontsize_scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yscale('log')
        plt.xticks([10, 25, 50, 75, 100, 125, 150, 175, 200],
                   fontsize=self.font_line_sizes['ticksLabelSize'] * fontsize_scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks([0.01, 0.1, 1, 10, 100],
                   [0.01, 0.1, 1, 10, 10],
                   fontsize=self.font_line_sizes['ticksLabelSize'] * fontsize_scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.ylim(0.001, 100)
        plt.xlim(0, 200)

        '''
        # Version 1: Legend to the right
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='center left', bbox_to_anchor=(1, 0.5))
        '''

        # # Version 2: Legend below
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.9])
        # plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='upper center',
        #            bbox_to_anchor=(0.47, -0.15), ncol=4, fancybox=True, columnspacing=0.25, handletextpad=0.3)

        # Version 3: Crunched legend
        box = ax.get_position()
        ax.set_position([box.x0 * 0.9, box.y0 * 0.9, box.width * 1.1, box.height * 1.1])
        h, l = ax.get_legend_handles_labels()
        ph = [plt.scatter([], [], marker="o", color=colors[1], s=100),
              plt.scatter([], [], marker=""),
              plt.scatter([], [], marker="o", color=colors[0], s=100),
              plt.scatter([], [], marker="")]
        handles = ph[:1] + h[:3] + ph[1:2] + h[3:6] + ph[2:3] + h[6:9] + ph[3:] + h[9:]
        labels = ["Sub-THz:"] + l[:3] + [""] + l[3:6] + ["mmWave:"] + l[6:9] + [""] + l[9:]
        leg = plt.legend(handles, labels, ncol=4, fontsize=self.font_line_sizes['legendSize'] * 1.05, columnspacing=0.4,
                         handletextpad=0.1, loc='lower right', framealpha=0.7)

        extension = 'journal_single_3_capacity_vs_n_sats' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    # Co-planar orbits analysis plots
    # X axis: time
    def coplanar_tx_rx_distance_vs_time_plot(self, file_ext='.jpg', alpha_list=None, n_sats_list=None,
                                             show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        h_high = self.orbital_input_params['h_lims'][0]
        h_low = self.orbital_input_params['h_low']

        # Foramtting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[
            (math_results_df['n_sats'] == n_sats_list[0]) & (math_results_df['h_high'] == h_high) & (
                    math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time_math[i] - t_start for i in range(len(time_math))]
        time_math = [time_math[i].total_seconds() for i in range(len(time_math))]

        time_sim = sim_results_df.loc[
            (sim_results_df['n_sats'] == n_sats_list[1]) & (sim_results_df['h_high'] == h_high) & (
                    sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time_sim[i] - t_start for i in range(len(time_sim))]
        time_sim = [time_sim[i].total_seconds() for i in range(len(time_sim))]

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[
                           (math_results_df['n_sats'] == n_sats) & (math_results_df['h_high'] == h_high) & (
                                   math_results_df['alpha'] == alpha), 'tx_rx_distance'] * 1e-3
            plt.plot(time_math, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[
                           (sim_results_df['n_sats'] == n_sats) & (sim_results_df['h_high'] == h_high) & (
                                   sim_results_df['alpha'] == alpha), 'tx_rx_distance'] * 1e-3
            plt.plot(time_sim, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        plt.suptitle('Coplanar orbits: Tx-Rx distance', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title('orbit separation = {} km'.format((h_high - h_low) * 1e-3),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Tx-Rx distance [km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'coplanar_TxRx_distance_vs_time' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def coplanar_rx_power_vs_time_plot(self, file_ext='.jpg', alpha_list=None, n_sats_list=None,
                                       show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        h_high = self.orbital_input_params['h_lims'][0]
        h_low = self.orbital_input_params['h_low']

        # Foramtting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[
            (math_results_df['n_sats'] == n_sats_list[0]) & (math_results_df['h_high'] == h_high) & (
                    math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time_math[i] - t_start for i in range(len(time_math))]
        time_math = [time_math[i].total_seconds() for i in range(len(time_math))]

        time_sim = sim_results_df.loc[
            (sim_results_df['n_sats'] == n_sats_list[1]) & (sim_results_df['h_high'] == h_high) & (
                    sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time_sim[i] - t_start for i in range(len(time_sim))]
        time_sim = [time_sim[i].total_seconds() for i in range(len(time_sim))]

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[
                (math_results_df['n_sats'] == n_sats) & (math_results_df['h_high'] == h_high) & (
                        math_results_df['alpha'] == alpha), 'rx_power']
            plt.plot(time_math, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[
                (sim_results_df['n_sats'] == n_sats) & (sim_results_df['h_high'] == h_high) & (
                        sim_results_df['alpha'] == alpha), 'rx_power']
            plt.plot(time_sim, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        plt.suptitle('Coplanar orbits: Rx power', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title('orbit separation = {} km'.format((h_high - h_low) * 1e-3),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Tx-Rx distance [km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'coplanar_rx_power_vs_time' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def coplanar_interference_vs_time_plot(self, file_ext='.jpg', alpha_list=None,
                                           n_sats_list=None, show=True, separation_index=0):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        h_high = self.orbital_input_params['h_lims'][0] + self.orbital_input_params[
            'h_resolution_sim'] * separation_index
        h_low = self.orbital_input_params['h_low']

        # Foramtting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[
            (math_results_df['n_sats'] == n_sats_list[0]) & (math_results_df['h_high'] == h_high) & (
                    math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time_math[i] - t_start for i in range(len(time_math))]
        time_math = [time_math[i].total_seconds() for i in range(len(time_math))]

        time_sim = sim_results_df.loc[
            (sim_results_df['n_sats'] == n_sats_list[0]) & (sim_results_df['h_high'] == h_high) & (
                    sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time_sim[i] - t_start for i in range(len(time_sim))]
        time_sim = [time_sim[i].total_seconds() for i in range(len(time_sim))]

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[
                (math_results_df['n_sats'] == n_sats) & (math_results_df['h_high'] == h_high) & (
                        math_results_df['alpha'] == alpha), 'interference']
            plt.plot(time_math, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[
                (sim_results_df['n_sats'] == n_sats) & (sim_results_df['h_high'] == h_high) & (
                        sim_results_df['alpha'] == alpha), 'interference']
            plt.plot(time_sim, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        plt.suptitle('Coplanar orbits: Expected interference', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title('orbit separation = {} km'.format((h_high - h_low) * 1e-3),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('E[I] [dBm]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'coplanar_interference_vs_time' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def coplanar_interference_vs_time_histogram(self, file_ext='.jpg', alpha_list=None, n_sats_list=None, show=True,
                                                math=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))

        h_high = self.orbital_input_params['h_lims'][0]
        h_low = self.orbital_input_params['h_low']
        orbit_separation = h_high - h_low

        if math:
            results_df = self.math_results_df
        else:
            results_df = self.sim_results_df

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        for i, (alpha, n_sats) in enumerate(pairs):
            x_values = results_df.loc[(results_df['n_sats'] == n_sats) & (results_df['h_high'] == h_high) & (
                    results_df['alpha'] == alpha), 'interference']
            if not x_values.isnull().all():
                plt.hist(x_values, color=colors[i],
                         label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), density=True, alpha=0.7,
                         edgecolor='k', zorder=10, bins=40, range=(-85, -50))

        if math:
            plt.suptitle('Coplanar orbits math: Expected interference probability',
                         fontsize=self.font_line_sizes['titleSize'],
                         fontname=self.font_line_sizes['fontname'])
            extension = 'coplanar_interference_histogram_math' + file_ext
        else:
            plt.suptitle('Coplanar orbits simulation: Expected interference probability',
                         fontsize=self.font_line_sizes['titleSize'],
                         fontname=self.font_line_sizes['fontname'])
            extension = 'coplanar_interference_histogram_simulation' + file_ext
        plt.title('orbit separation = {} km'.format(orbit_separation * 1e-3),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('E[I] [dBm]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Probability', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def coplanar_sir_vs_time_plot(self, file_ext='.jpg', alpha_list=None, n_sats_list=None,
                                  show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        h_high = self.orbital_input_params['h_lims'][0]
        h_low = self.orbital_input_params['h_low']

        # Foramtting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[
            (math_results_df['n_sats'] == n_sats_list[0]) & (math_results_df['h_high'] == h_high) & (
                    math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time_math[i] - t_start for i in range(len(time_math))]
        time_math = [time_math[i].total_seconds() for i in range(len(time_math))]

        time_sim = sim_results_df.loc[
            (sim_results_df['n_sats'] == n_sats_list[1]) & (sim_results_df['h_high'] == h_high) & (
                    sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time_sim[i] - t_start for i in range(len(time_sim))]
        time_sim = [time_sim[i].total_seconds() for i in range(len(time_sim))]

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            interference = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h_high'] == h_high) & (
                        math_results_df['alpha'] == alpha), 'interference'])
            rx_power = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h_high'] == h_high) & (
                        math_results_df['alpha'] == alpha), 'rx_power'])
            SIR = [rx_power[i] - interference[i] if interference[i] else None for i in range(len(rx_power))]
            plt.plot(time_math, SIR, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            interference = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h_high'] == h_high) & (
                        sim_results_df['alpha'] == alpha), 'interference'])
            rx_power = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h_high'] == h_high) & (
                        sim_results_df['alpha'] == alpha), 'rx_power'])
            SIR = [rx_power[i] - interference[i] if interference[i] else None for i in
                   range(len(rx_power))]
            plt.plot(time_sim, SIR, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5,
                     markevery=None)
        plt.suptitle('Coplanar orbits: Expected interference', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title('orbit separation = {} km'.format((h_high - h_low) * 1e-3),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        extension = 'coplanar_sir_vs_time' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def coplanar_n_interferers_vs_time_plot(self, file_ext='.jpg', alpha_list=None, n_sats_list=None, show=True,
                                            separation_index=0):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        h_high = self.orbital_input_params['h_lims'][0] + self.orbital_input_params[
            'h_resolution_sim'] * separation_index
        h_low = self.orbital_input_params['h_low']

        # Foramtting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[
            (math_results_df['n_sats'] == n_sats_list[0]) & (math_results_df['h_high'] == h_high) & (
                    math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time_math[i] - t_start for i in range(len(time_math))]
        time_math = [time_math[i].total_seconds() for i in range(len(time_math))]

        time_sim = sim_results_df.loc[
            (sim_results_df['n_sats'] == n_sats_list[0]) & (sim_results_df['h_high'] == h_high) & (
                    sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time_sim[i] - t_start for i in range(len(time_sim))]
        time_sim = [time_sim[i].total_seconds() for i in range(len(time_sim))]

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[
                (math_results_df['n_sats'] == n_sats) & (math_results_df['h_high'] == h_high) & (
                        math_results_df['alpha'] == alpha), 'n_interferers']
            plt.plot(time_math, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[
                (sim_results_df['n_sats'] == n_sats) & (sim_results_df['h_high'] == h_high) & (
                        sim_results_df['alpha'] == alpha), 'n_interferers']
            plt.plot(time_sim, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        plt.suptitle('Coplanar orbits:\nNumber of satellites contributing to interference at the receiver',
                     fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title('orbit separation = {} km'.format((h_high - h_low) * 1e-3),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Number of interferers', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'coplanar_n_interferers_vs_time' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def coplanar_distance_to_nearest_interferer_vs_time_plot(self, orbit_separation=10e3, file_ext='.jpg',
                                                             alpha_list=None,
                                                             n_sats_list=None,
                                                             show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        h_high = self.orbital_input_params['h_lims'][0]
        h_low = self.orbital_input_params['h_low']

        # Foramtting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[
            (math_results_df['n_sats'] == n_sats_list[0]) & (math_results_df['h_high'] == h_high) & (
                    math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time_math[i] - t_start for i in range(len(time_math))]
        time_math = [time_math[i].total_seconds() for i in range(len(time_math))]

        time_sim = sim_results_df.loc[
            (sim_results_df['n_sats'] == n_sats_list[1]) & (sim_results_df['h_high'] == h_high) & (
                    sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time_sim[i] - t_start for i in range(len(time_sim))]
        time_sim = [time_sim[i].total_seconds() for i in range(len(time_sim))]

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[
                           (math_results_df['n_sats'] == n_sats) & (math_results_df['h_high'] == h_high) & (
                                   math_results_df['alpha'] == alpha), 'd_interferer'] * 1e-3
            plt.plot(time_math, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[
                           (sim_results_df['n_sats'] == n_sats) & (sim_results_df['h_high'] == h_high) & (
                                   sim_results_df['alpha'] == alpha), 'd_interferer'] * 1e-3
            plt.plot(time_sim, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        plt.suptitle('Coplanar orbits: Distance to closest interferer', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title('orbit separation = {} km'.format((h_high - h_low) * 1e-3),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Distance [km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'coplanar_d_closest_interferer_vs_time' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def coplanar_sir_snr_sinr_capacity_vs_time_plot_h(self, file_ext='.jpg', alpha_list=None, n_sats_list=None,
                                                      show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        h_high = self.orbital_input_params['h_lims'][0]
        h_low = self.orbital_input_params['h_low']

        # Foramtting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[
            (math_results_df['n_sats'] == n_sats_list[0]) & (math_results_df['h_high'] == h_high) & (
                    math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time_math[i] - t_start for i in range(len(time_math))]
        time_math = [time_math[i].total_seconds() for i in range(len(time_math))]

        time_sim = sim_results_df.loc[
            (sim_results_df['n_sats'] == n_sats_list[1]) & (sim_results_df['h_high'] == h_high) & (
                    sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time_sim[i] - t_start for i in range(len(time_sim))]
        time_sim = [time_sim[i].total_seconds() for i in range(len(time_sim))]

        _, _, colors = tk.templateColors(iterable_size=len(alpha_list))

        fig = get_ax(figsize=(10 * 2, 10))
        scale_factor = 0.8

        # SIR (2)
        ax1 = plt.subplot(221)
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            sir_linear = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) &
                                                  (math_results_df['alpha'] == alpha) &
                                                  (math_results_df['h_high'] == h_high), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(time_math, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            sir_linear = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) &
                                                 (sim_results_df['alpha'] == alpha) &
                                                 (sim_results_df['h_high'] == h_high), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(time_sim, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax1.set_title('Coplanar orbit: SIR 2', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel(r'SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        # plt.legend(ncol=1, framealpha=0.7, fontsize=self.font_line_sizes['legendSize']*scale_factor, loc='upper right')

        # SNR
        ax2 = plt.subplot(222)
        plt.grid()
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            snr_linear = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) &
                                                  (math_results_df['alpha'] == alpha) &
                                                  (math_results_df['h_high'] == h_high), 'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.plot(time_math, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            snr_linear = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) &
                                                 (sim_results_df['alpha'] == alpha) &
                                                 (sim_results_df['h_high'] == h_high), 'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.scatter(time_sim, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax2.set_title('Coplanar orbit: SNR', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel(r'SNR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        # plt.legend(ncol=2, framealpha=0.7, fontsize=self.font_line_sizes['legendSize']*scale_factor, loc='upper right')
        # plt.ylim([2.5, 10])

        # SINR
        ax3 = plt.subplot(223)
        plt.grid()
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            sinr_linear = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) &
                                                   (math_results_df['alpha'] == alpha) &
                                                   (math_results_df['h_high'] == h_high), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(time_math, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            sinr_linear = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) &
                                                  (sim_results_df['alpha'] == alpha) &
                                                  (sim_results_df['h_high'] == h_high), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.scatter(time_sim, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax3.set_title('Coplanar orbit: SINR', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel(r'SINR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])

        # Capacity
        ax4 = plt.subplot(224)
        plt.grid()
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) &
                                                (math_results_df['alpha'] == alpha) &
                                                (math_results_df['h_high'] == h_high), 'capacity_gbps'])
            plt.plot(time_math, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) &
                                               (sim_results_df['alpha'] == alpha) &
                                               (sim_results_df['h_high'] == h_high), 'capacity_gbps'])
            plt.scatter(time_sim, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax4.set_title('Coplanar orbit: Capacity', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel(r'Channel Capacity [Gbps]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yscale('log')
        plt.legend(ncol=2, framealpha=0.7, fontsize=self.font_line_sizes['legendSize'] * scale_factor,
                   loc='upper right')

        extension = 'coplanar_sir_snr_sinr_capacity_vs_time_' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    # X axis: orbit separation
    def tx_rx_distance_vs_orbit_separation_plot(self, file_ext='.jpg', alpha_list=None,
                                                n_sats_list=None,
                                                show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h_low = self.orbital_input_params['h_low']
        pairs = list(product(alpha_list, n_sats_list))

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(sim_results_df)

        # Orbit separation vectors
        orbit_separation_list_math = np.arange(self.orbital_input_params['h_lims'][0],
                                               self.orbital_input_params['h_lims'][1],
                                               self.orbital_input_params['h_resolution_math'])
        orbit_separation_list_math = orbit_separation_list_math - h_low

        orbit_separation_list_sim = np.arange(self.orbital_input_params['h_lims'][0],
                                              self.orbital_input_params['h_lims'][1],
                                              self.orbital_input_params['h_resolution_sim'])
        orbit_separation_list_sim = orbit_separation_list_sim - h_low

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[
                           (math_results_df['n_sats'] == n_sats) & (
                                   math_results_df['alpha'] == alpha), 'tx_rx_distance'] * 1e-3
            plt.plot(orbit_separation_list_math * 1e-3, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['alpha'] == alpha),
                                          'tx_rx_distance'] * 1e-3
            plt.plot(orbit_separation_list_sim * 1e-3, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)

        plt.suptitle('Coplanar orbits: Tx-Rx distance', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit separation [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Tx-Rx distance [km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'coplanar_TxRx_distance_vs_orbit_separation' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def rx_power_vs_orbit_separation_plot(self, file_ext='.jpg', alpha_list=None,
                                          n_sats_list=None,
                                          show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h_low = self.orbital_input_params['h_low']
        pairs = list(product(alpha_list, n_sats_list))

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(sim_results_df)

        # Orbit separation vectors
        orbit_separation_list_math = np.arange(self.orbital_input_params['h_lims'][0],
                                               self.orbital_input_params['h_lims'][1],
                                               self.orbital_input_params['h_resolution_math'])
        orbit_separation_list_math = orbit_separation_list_math - h_low

        orbit_separation_list_sim = np.arange(self.orbital_input_params['h_lims'][0],
                                              self.orbital_input_params['h_lims'][1],
                                              self.orbital_input_params['h_resolution_sim'])
        orbit_separation_list_sim = orbit_separation_list_sim - h_low

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[
                (math_results_df['n_sats'] == n_sats) & (
                        math_results_df['alpha'] == alpha), 'rx_power']
            plt.plot(orbit_separation_list_math * 1e-3, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['alpha'] == alpha),
                                          'rx_power']
            plt.plot(orbit_separation_list_sim * 1e-3, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)

        plt.suptitle('Coplanar orbits: Rx power', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit separation [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Rx power [dBm]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'coplanar_rx_power_vs_orbit_separation' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def interference_vs_orbit_separation_plot(self, file_ext='.jpg', alpha_list=None,
                                              n_sats_list=None,
                                              show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h_low = self.orbital_input_params['h_low']
        pairs = list(product(alpha_list, n_sats_list))

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(sim_results_df)

        # Orbit separation vectors
        orbit_separation_list_math = np.arange(self.orbital_input_params['h_lims'][0],
                                               self.orbital_input_params['h_lims'][1],
                                               self.orbital_input_params['h_resolution_math'])
        orbit_separation_list_math = orbit_separation_list_math - h_low

        orbit_separation_list_sim = np.arange(self.orbital_input_params['h_lims'][0],
                                              self.orbital_input_params['h_lims'][1],
                                              self.orbital_input_params['h_resolution_sim'])
        orbit_separation_list_sim = orbit_separation_list_sim - h_low

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[
                (math_results_df['n_sats'] == n_sats) & (
                        math_results_df['alpha'] == alpha), 'interference']
            plt.plot(orbit_separation_list_math * 1e-3, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['alpha'] == alpha),
                                          'interference']
            plt.plot(orbit_separation_list_sim * 1e-3, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)

        plt.suptitle('Coplanar orbits: Expected interference', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit separation [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('E[I] [dBm]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'coplanar_interference_vs_orbit_separation' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def sir_vs_orbit_separation_plot(self, file_ext='.jpg', alpha_list=None,
                                     n_sats_list=None,
                                     show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h_low = self.orbital_input_params['h_low']
        pairs = list(product(alpha_list, n_sats_list))

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(sim_results_df)

        # Orbit separation vectors
        orbit_separation_list_math = np.arange(self.orbital_input_params['h_lims'][0],
                                               self.orbital_input_params['h_lims'][1],
                                               self.orbital_input_params['h_resolution_math'])
        orbit_separation_list_math = orbit_separation_list_math - h_low

        orbit_separation_list_sim = np.arange(self.orbital_input_params['h_lims'][0],
                                              self.orbital_input_params['h_lims'][1],
                                              self.orbital_input_params['h_resolution_sim'])
        orbit_separation_list_sim = orbit_separation_list_sim - h_low

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            interference = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (
                    math_results_df['alpha'] == alpha), 'interference_linear'])
            interference = [10 * np.log10(y_value) if y_value != 0 else None for y_value in interference]
            rx_power = list(
                math_results_df.loc[
                    (math_results_df['n_sats'] == n_sats) & (math_results_df['alpha'] == alpha), 'rx_power'])
            y_values = [rx_power[i] - interference[i] if interference[i] else None for i in
                        range(len(rx_power))]
            plt.plot(orbit_separation_list_math * 1e-3, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            interference = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (
                    sim_results_df['alpha'] == alpha), 'interference_linear'])
            interference = [10 * np.log10(y_value) if y_value != 0 else None for y_value in interference]
            rx_power = list(
                sim_results_df.loc[
                    (sim_results_df['n_sats'] == n_sats) & (sim_results_df['alpha'] == alpha), 'rx_power'])
            y_values = [rx_power[i] - interference[i] if interference[i] else None for i in
                        range(len(rx_power))]
            plt.plot(orbit_separation_list_sim * 1e-3, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)

        plt.suptitle('Coplanar orbits: Signal to Interference Ratio (SIR)', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit separation [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'coplanar_sir_vs_orbit_separation' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def n_interferers_vs_orbit_separation_plot(self, file_ext='.jpg', alpha_list=None,
                                               n_sats_list=None,
                                               show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h_low = self.orbital_input_params['h_low']
        pairs = list(product(alpha_list, n_sats_list))

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(sim_results_df)

        # Orbit separation vectors
        orbit_separation_list_math = np.arange(self.orbital_input_params['h_lims'][0],
                                               self.orbital_input_params['h_lims'][1],
                                               self.orbital_input_params['h_resolution_math'])
        orbit_separation_list_math = orbit_separation_list_math - h_low

        orbit_separation_list_sim = np.arange(self.orbital_input_params['h_lims'][0],
                                              self.orbital_input_params['h_lims'][1],
                                              self.orbital_input_params['h_resolution_sim'])
        orbit_separation_list_sim = orbit_separation_list_sim - h_low

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[
                (math_results_df['n_sats'] == n_sats) & (
                        math_results_df['alpha'] == alpha), 'n_interferers']
            plt.plot(orbit_separation_list_math * 1e-3, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['alpha'] == alpha),
                                          'n_interferers']
            plt.plot(orbit_separation_list_sim * 1e-3, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)

        plt.suptitle('Coplanar orbits:\nNumber of satellites contributing to interference at the receiver',
                     fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit separation [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Number of interferers ', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'coplanar_n_interferers_vs_orbit_separation' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def distance_to_nearest_interferer_vs_orbit_separation_plot(self, file_ext='.jpg', alpha_list=None,
                                                                n_sats_list=None,
                                                                show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h_low = self.orbital_input_params['h_low']
        pairs = list(product(alpha_list, n_sats_list))

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(sim_results_df)

        # Orbit separation vectors
        orbit_separation_list_math = np.arange(self.orbital_input_params['h_lims'][0],
                                               self.orbital_input_params['h_lims'][1],
                                               self.orbital_input_params['h_resolution_math'])
        orbit_separation_list_math = orbit_separation_list_math - h_low

        orbit_separation_list_sim = np.arange(self.orbital_input_params['h_lims'][0],
                                              self.orbital_input_params['h_lims'][1],
                                              self.orbital_input_params['h_resolution_sim'])
        orbit_separation_list_sim = orbit_separation_list_sim - h_low

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[
                           (math_results_df['n_sats'] == n_sats) & (
                                   math_results_df['alpha'] == alpha), 'd_interferer'] * 1e-3
            plt.plot(orbit_separation_list_math * 1e-3, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['alpha'] == alpha),
                                          'd_interferer'] * 1e-3
            plt.plot(orbit_separation_list_sim * 1e-3, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)

        plt.suptitle('Coplanar orbits: Distance to closest interferer', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit separation [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Distance [km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'coplanar_d_closest_interferer_vs_orbit_separation' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def coplanar_sir1_and_sir2_vs_orbit_separation_paper_plot(self, file_ext='.jpg', alpha_list=None, n_sats_list=None,
                                                              show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h_low = self.orbital_input_params['h_low']
        pairs = list(product(alpha_list, n_sats_list))

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(sim_results_df)

        # Orbit separation vectors
        orbit_separation_list_math = np.arange(self.orbital_input_params['h_lims'][0],
                                               self.orbital_input_params['h_lims'][1],
                                               self.orbital_input_params['h_resolution_math'])
        orbit_separation_list_math = orbit_separation_list_math - h_low

        orbit_separation_list_sim = np.arange(self.orbital_input_params['h_lims'][0],
                                              self.orbital_input_params['h_lims'][1],
                                              self.orbital_input_params['h_resolution_sim'])
        orbit_separation_list_sim = orbit_separation_list_sim - h_low

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax(figsize=(10 * 2, 6))

        # SIR1
        ax1 = plt.subplot(121)
        plt.grid()
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            interference = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) &
                                                    (math_results_df['alpha'] == alpha), 'interference'])
            rx_power = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) &
                                                (math_results_df['alpha'] == alpha), 'rx_power'])
            y_values = [rx_power[i] - interference[i] if interference[i] else None for i in range(len(rx_power))]
            plt.plot(orbit_separation_list_math * 1e-3, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            interference = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) &
                                                   (sim_results_df['alpha'] == alpha), 'interference'])
            rx_power = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) &
                                               (sim_results_df['alpha'] == alpha), 'rx_power'])
            y_values = [rx_power[i] - interference[i] if interference[i] else None for i in range(len(rx_power))]
            plt.scatter(orbit_separation_list_sim * 1e-3, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax1.set_title('Coplanar orbits: SIR 1', fontsize=self.font_line_sizes['titleSize'],
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit separation [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xlim([0, 500])
        plt.ylabel(r'SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'], fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'], fontname=self.font_line_sizes['fontname'])
        plt.legend(ncol=2, framealpha=0.7, fontsize=15, loc='upper right')

        # SIR2
        ax2 = plt.subplot(122)
        plt.grid()
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            sir_linear = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (
                    math_results_df['alpha'] == alpha), 'sir_linear'])
            y_value = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(orbit_separation_list_math * 1e-3, y_value, color=colors[i],
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            sir_linear = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (
                    sim_results_df['alpha'] == alpha), 'sir_linear'])
            y_value = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(orbit_separation_list_sim * 1e-3, y_value, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax2.set_title('Coplanar orbits: SIR 2', fontsize=self.font_line_sizes['titleSize'],
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit separation [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xlim([0, 500])
        plt.ylabel(r'SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'], fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'], fontname=self.font_line_sizes['fontname'])
        plt.legend(ncol=2, framealpha=0.7, fontsize=15, loc='upper right')
        extension = 'coplanar_sir1_and_sir2_vs_orbit_separation_' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def coplanar_sir_snr_sinr_capacity_vs_orbit_separation_plot(self, file_ext='.jpg', alpha_list=None,
                                                                n_sats_list=None, show=True):
        higher_orbit_only = self.orbital_input_params['higher_orbit_only']
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h_low = self.orbital_input_params['h_low']
        pairs = list(product(alpha_list, n_sats_list))

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(sim_results_df)

        # Orbit separation vectors
        orbit_separation_list_math = np.arange(self.orbital_input_params['h_lims'][0],
                                               self.orbital_input_params['h_lims'][1],
                                               self.orbital_input_params['h_resolution_math'])
        orbit_separation_list_math = orbit_separation_list_math - h_low

        orbit_separation_list_sim = np.arange(self.orbital_input_params['h_lims'][0],
                                              self.orbital_input_params['h_lims'][1],
                                              self.orbital_input_params['h_resolution_sim'])
        orbit_separation_list_sim = orbit_separation_list_sim - h_low

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax(figsize=(10 * 2, 10))
        scale_factor = 0.8
        # SIR (2)
        ax1 = plt.subplot(221)
        plt.grid()
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            sir_linear = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) &
                                                  (math_results_df['alpha'] == alpha), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(orbit_separation_list_math * 1e-3, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            sir_linear = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) &
                                                 (sim_results_df['alpha'] == alpha), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(orbit_separation_list_sim * 1e-3, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax1.set_title('Coplanar orbit: SIR 2', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit separation [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel(r'SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        # plt.legend(ncol=1, framealpha=0.7, fontsize=self.font_line_sizes['legendSize']*scale_factor, loc='upper right')

        # SNR
        ax2 = plt.subplot(222)
        plt.grid()
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            snr_linear = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) &
                                                  (math_results_df['alpha'] == alpha), 'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.plot(orbit_separation_list_math * 1e-3, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            snr_linear = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) &
                                                 (sim_results_df['alpha'] == alpha), 'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.scatter(orbit_separation_list_sim * 1e-3, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax2.set_title('Coplanar orbit: SNR', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit separation [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel(r'SNR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        # plt.legend(ncol=2, framealpha=0.7, fontsize=self.font_line_sizes['legendSize']*scale_factor, loc='upper right')
        # plt.ylim([2.5, 10])

        # SINR
        ax3 = plt.subplot(223)
        plt.grid()
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            sinr_linear = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) &
                                                   (math_results_df['alpha'] == alpha), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(orbit_separation_list_math * 1e-3, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            sinr_linear = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) &
                                                  (sim_results_df['alpha'] == alpha), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.scatter(orbit_separation_list_sim * 1e-3, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax3.set_title('Coplanar orbit: SINR', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit separation [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel(r'SINR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])

        # Capacity
        ax4 = plt.subplot(224)
        plt.grid()
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) &
                                                (math_results_df['alpha'] == alpha), 'capacity_gbps'])
            plt.plot(orbit_separation_list_math * 1e-3, y_values, color=colors[i],
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) &
                                               (sim_results_df['alpha'] == alpha), 'capacity_gbps'])
            plt.scatter(orbit_separation_list_sim * 1e-3, y_values, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        ax4.set_title('Coplanar orbit: Capacity', fontsize=self.font_line_sizes['titleSize'] * scale_factor,
                      fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit separation [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel(r'Channel Capacity [Gbps]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yscale('log')
        plt.legend(ncol=2, framealpha=0.7, fontsize=self.font_line_sizes['legendSize'] * scale_factor,
                   loc='upper right')

        extension = 'coplanar_sir_snr_sinr_capacity_vs_orbit_separation' + '_higher_orbit_only_' + str(
            higher_orbit_only) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    # X axis number of satellites
    def coplanar_sir_snr_sinr_capacity_vs_n_sats_plot(self, file_ext='.jpg', show=True):
        thz_alpha_list = self.thz_input_params['alpha_list']
        mmwave_alpha_list = self.mmWave_input_params['alpha_list']
        n_sats_list = self.orbital_input_params['n_sats_list']

        # Results dataframes
        thz_math_results_df = self.thz_math_results_df
        mmwave_math_results_df = self.mmWave_math_results_df

        # Time averaging of results
        thz_math_results_df = thz_math_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(thz_math_results_df)
        mmwave_math_results_df = mmwave_math_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(mmwave_math_results_df)

        mmwave_colors = ['#e31a1c', '#e41bb2']  # Dark red, Light red
        thz_colors = ['#3b9dde', '#1e1eb3']  # Dark blue, Light blue
        linestyles = ['solid', 'dashdot', 'dotted']

        fig = get_ax(figsize=(10 * 2, 10))

        # SIR
        ax1 = plt.subplot(221)
        plt.grid()
        # mmWave
        for i, alpha in enumerate(mmwave_alpha_list):
            sir_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(n_sats_list, y_values, color=mmwave_colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        # THz
        for i, alpha in enumerate(thz_alpha_list):
            sir_linear = list(thz_math_results_df.loc[(thz_math_results_df['alpha'] == alpha), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(n_sats_list, y_values, color=thz_colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        plt.xlabel('Satellites per orbit', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        # plt.xlim(0, 500)
        # plt.ylim(-25, 14)

        # SNR
        ax2 = plt.subplot(222)
        plt.grid()
        # mmWave
        for i, alpha in enumerate(mmwave_alpha_list):
            snr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha), 'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.plot(n_sats_list, y_values, color=mmwave_colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        # THz
        for i, alpha in enumerate(thz_alpha_list):
            snr_linear = list(thz_math_results_df.loc[(thz_math_results_df['alpha'] == alpha), 'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.plot(n_sats_list, y_values, color=thz_colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        plt.xlabel('Satellites per orbit', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SNR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        # plt.xlim(0, 500)
        # plt.ylim(-25, 14)

        # SINR
        ax3 = plt.subplot(223)
        plt.grid()
        # mmWave
        for i, alpha in enumerate(mmwave_alpha_list):
            sinr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(n_sats_list, y_values, color=mmwave_colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        # THz
        for i, alpha in enumerate(thz_alpha_list):
            sinr_linear = list(thz_math_results_df.loc[(thz_math_results_df['alpha'] == alpha), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(n_sats_list, y_values, color=thz_colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        plt.xlabel('Satellites per orbit', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SINR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        # plt.xlim(0, 500)
        # plt.ylim(-25, 14)

        # Channel capacity
        ax4 = plt.subplot(224)
        plt.grid()
        # mmWave
        for i, alpha in enumerate(mmwave_alpha_list):
            y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha), 'capacity_gbps'])
            plt.plot(n_sats_list, y_values, color=mmwave_colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        # THz
        for i, alpha in enumerate(thz_alpha_list):
            y_values = list(thz_math_results_df.loc[(thz_math_results_df['alpha'] == alpha), 'capacity_gbps'])
            plt.plot(n_sats_list, y_values, color=thz_colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        plt.xlabel('Satellites per orbit', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Channel Capacity [Gbps]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yscale('log')
        plt.legend(ncol=2, framealpha=0.7, fontsize=self.font_line_sizes['legendSize'],
                   loc='upper right')
        # plt.xlim(0, 500)
        # plt.ylim(-25, 14)

        extension = 'coplanar_sir_snr_sinr_capacity_vs_n_sats_' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    # Paper plots
    def coplanar_sir_vs_time_paper_plot(self, orbit_separation=5e3, file_ext='.jpg', alpha_list=None, n_sats_list=None,
                                        show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        h_high = self.orbital_input_params['h_low'] + orbit_separation

        # Formatting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[
            (math_results_df['n_sats'] == n_sats_list[0]) & (math_results_df['h_high'] == h_high) & (
                    math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time - t_start for time in time_math]
        time_math = [time.total_seconds() for time in time_math]

        time_sim = sim_results_df.loc[
            (sim_results_df['n_sats'] == n_sats_list[0]) & (sim_results_df['h_high'] == h_high) & (
                    sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time - t_start for time in time_sim]
        time_sim = [time.total_seconds() for time in time_sim]

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            interference = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h_high'] == h_high) & (
                        math_results_df['alpha'] == alpha), 'interference'])
            rx_power = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h_high'] == h_high) & (
                        math_results_df['alpha'] == alpha), 'rx_power'])
            SIR = [rx_power[i] - interference[i] if interference[i] else None for i in
                   range(len(rx_power))]
            plt.plot(time_math, SIR, color=colors[i],
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            interference = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h_high'] == h_high) & (
                        sim_results_df['alpha'] == alpha), 'interference'])
            rx_power = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h_high'] == h_high) & (
                        sim_results_df['alpha'] == alpha), 'rx_power'])
            SIR = [rx_power[i] - interference[i] if interference[i] else None for i in
                   range(len(rx_power))]
            plt.scatter(time_sim, SIR, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])
        plt.suptitle('Coplanar orbits: Signal to Interference Ratio (SIR)', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title('orbit separation = {} km'.format(orbit_separation * 1e-3),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'], fontname=self.font_line_sizes['fontname'])
        plt.yticks([10, 0, -10, -20, -30, -40], fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend(ncol=2, framealpha=0.7, fontsize=15, loc='lower left')
        extension = 'coplanar_sir_vs_time_paper' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def coplanar_interference_vs_orbit_separation_paper_plot(self, file_ext='.jpg', alpha_list=None, n_sats_list=None,
                                                             show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h_low = self.orbital_input_params['h_low']
        pairs = list(product(alpha_list, n_sats_list))

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(sim_results_df)

        # Orbit separation vectors
        orbit_separation_list_math = np.arange(self.orbital_input_params['h_lims'][0],
                                               self.orbital_input_params['h_lims'][1],
                                               self.orbital_input_params['h_resolution_math'])
        orbit_separation_list_math = orbit_separation_list_math - h_low

        orbit_separation_list_sim = np.arange(self.orbital_input_params['h_lims'][0],
                                              self.orbital_input_params['h_lims'][1],
                                              self.orbital_input_params['h_resolution_sim'])
        orbit_separation_list_sim = orbit_separation_list_sim - h_low

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_value = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (
                    math_results_df['alpha'] == alpha), 'interference'])
            plt.plot(orbit_separation_list_math * 1e-3, y_value, color=colors[i],
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                     linewidth=self.font_line_sizes['linewidth'])
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_value = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (
                    sim_results_df['alpha'] == alpha), 'interference'])
            plt.scatter(orbit_separation_list_sim * 1e-3, y_value, color=colors[i],
                        label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])
        plt.suptitle('Coplanar orbits: Interference', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit separation [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xlim([0, 500])
        plt.ylabel(r'$E[I_2]$ [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'], fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'], fontname=self.font_line_sizes['fontname'])
        plt.legend(ncol=2, framealpha=0.7, fontsize=15, loc='upper right')
        extension = 'coplanar_interference_vs_orbit_separation_paper' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def coplanar_sir_vs_time_higher_vs_lower_paper(self, orbit_separation=5e3, file_ext='.jpg', alpha_list=None,
                                                   n_sats_list=None,
                                                   show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        h_high = self.orbital_input_params['h_low'] + orbit_separation

        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        high_sim_results_df = self.high_sim_results_df
        high_math_results_df = self.high_math_results_df

        # Foramtting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[
            (math_results_df['n_sats'] == n_sats_list[0]) & (math_results_df['h_high'] == h_high) & (
                    math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time_math[i] - t_start for i in range(len(time_math))]
        time_math = [time_math[i].total_seconds() for i in range(len(time_math))]

        time_sim = sim_results_df.loc[
            (sim_results_df['n_sats'] == n_sats_list[0]) & (sim_results_df['h_high'] == h_high) & (
                    sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time_sim[i] - t_start for i in range(len(time_sim))]
        time_sim = [time_sim[i].total_seconds() for i in range(len(time_sim))]

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs) * 2)

        fig = get_ax()
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)
        # Lower orbit Math
        for i, (alpha, n_sats) in enumerate(pairs):
            sir_linear = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h_high'] == h_high) & (
                        math_results_df['alpha'] == alpha), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(time_math, y_values, color=colors[i],
                     label=r'Low: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)

        # Higher orbit Math
        for i, (alpha, n_sats) in enumerate(pairs):
            sir_linear = list(high_math_results_df.loc[(high_math_results_df['n_sats'] == n_sats) & (
                    high_math_results_df['h_high'] == h_high) & (
                                                               high_math_results_df['alpha'] == alpha), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(time_math, y_values, color=colors[i + len(pairs)],
                     label=r'High: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5,
                     markevery=None)

        # Lower orbit Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            sir_linear = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h_high'] == h_high) & (
                        sim_results_df['alpha'] == alpha), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(time_sim, y_values, color=colors[i],
                        label=r'Sim.-Low: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[i],
                        edgecolors='k', zorder=3,
                        s=sizes[i])

        # Higher orbit Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            sir_linear = list(high_sim_results_df.loc[(high_sim_results_df['n_sats'] == n_sats) & (
                    high_sim_results_df['h_high'] == h_high) & (
                                                              high_sim_results_df['alpha'] == alpha), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(time_sim, y_values, color=colors[i + len(pairs)],
                        label=r'Sim.-High: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[i],
                        edgecolors='k', zorder=3,
                        s=sizes[i])

        plt.suptitle('Coplanar orbits: Expected interference comparison high/low orbits',
                     fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title('orbit separation = {} km'.format(orbit_separation * 1e-3),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend(ncol=2)
        extension = 'coplanar_sir_vs_time_high_vs_low_paper' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def coplanar_interference_vs_orbit_separation_higher_vs_lower_paper(self, file_ext='.jpg', alpha_list=None,
                                                                        n_sats_list=None,
                                                                        show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h_low = self.orbital_input_params['h_low']
        pairs = list(product(alpha_list, n_sats_list))

        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        high_sim_results_df = self.high_sim_results_df
        high_math_results_df = self.high_math_results_df

        # Groupby and adjust db columns of results dataframes
        math_results_df = math_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(math_results_df)
        sim_results_df = sim_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(sim_results_df)
        high_sim_results_df = high_sim_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(high_sim_results_df)
        high_math_results_df = high_math_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(high_math_results_df)

        # Orbit separation vectors
        orbit_separation_list_math = np.arange(self.orbital_input_params['h_lims'][0],
                                               self.orbital_input_params['h_lims'][1],
                                               self.orbital_input_params['h_resolution_math'])
        orbit_separation_list_math = orbit_separation_list_math - h_low

        orbit_separation_list_sim = np.arange(self.orbital_input_params['h_lims'][0],
                                              self.orbital_input_params['h_lims'][1],
                                              self.orbital_input_params['h_resolution_sim'])
        orbit_separation_list_sim = orbit_separation_list_sim - h_low

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs) * 2)

        fig = get_ax()
        plt.grid()
        # Lower Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_value = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (
                    math_results_df['alpha'] == alpha), 'interference'])
            plt.plot(orbit_separation_list_math * 1e-3, y_value, color=colors[i],
                     label=r'Low: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                     linewidth=self.font_line_sizes['linewidth'])

        # Higher Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_value = list(high_math_results_df.loc[(high_math_results_df['n_sats'] == n_sats) & (
                    high_math_results_df['alpha'] == alpha), 'interference'])
            plt.plot(orbit_separation_list_math * 1e-3, y_value, color=colors[i + len(pairs)],
                     label=r'High: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                     linewidth=self.font_line_sizes['linewidth'])

        # Lower Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_value = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (
                    sim_results_df['alpha'] == alpha), 'interference'])
            plt.scatter(orbit_separation_list_sim * 1e-3, y_value, color=colors[i],
                        label=r'Sim. Low: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        # Lower Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_value = list(high_sim_results_df.loc[(high_sim_results_df['n_sats'] == n_sats) & (
                    high_sim_results_df['alpha'] == alpha), 'interference'])
            plt.scatter(orbit_separation_list_sim * 1e-3, y_value, color=colors[i + len(pairs)],
                        label=r'Sim. High: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[i],
                        edgecolor='k', zorder=3, s=sizes[i])

        plt.suptitle('Coplanar orbits: Interference', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit separation [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xlim([0, 500])
        plt.ylabel(r'$E[I_2]$ [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'], fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'], fontname=self.font_line_sizes['fontname'])
        plt.legend(ncol=2, framealpha=0.7, fontsize=15, loc='upper right')
        extension = 'coplanar_interference_vs_orbit_separation_high_vs_low_paper' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    # Journal paper plots
    def journal_coplanar_sir_vs_time_higher_vs_lower(self, orbit_separation=10e3, file_ext='.jpg', alpha_list=None,
                                                     n_sats_list=None, show=True):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        n_timesteps_math = self.orbital_input_params['beta_steps_math']
        n_timesteps_sim = self.orbital_input_params['beta_steps_sim']
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        high_math_results_df = self.high_math_results_df
        sim_results_df = self.sim_results_df
        high_sim_results_df = self.high_sim_results_df
        h_high = self.orbital_input_params['h_low'] + orbit_separation
        h_low = self.orbital_input_params['h_low']

        # Formating stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        beta_max = 2 * np.pi * 1 / min(n_sats_list)
        t1 = ad.period_from_semi_major_axis(Re + h_low)
        t2 = ad.period_from_semi_major_axis(Re + h_high)

        t_beta_max = beta_max / (2 * np.pi * (1 / t1 - 1 / t2))

        time_math = [timedelta(seconds=t_beta_max) / n_timesteps_math * i for i in
                     range(n_timesteps_math)]
        time_math = [time.total_seconds() for time in time_math]
        time_sim = [timedelta(seconds=t_beta_max) / n_timesteps_sim * i for i in
                    range(n_timesteps_sim)]
        time_sim = [time.total_seconds() for time in time_sim]
        # Resize time axis to the period corresponding to the number of satellites to show
        time_math = time_math[:int(len(time_math) * min(self.orbital_input_params['n_sats_list']) / min(n_sats_list))]
        time_sim = time_sim[:int(len(time_sim) * min(self.orbital_input_params['n_sats_list']) / min(n_sats_list))]

        _, _, colors = tk.templateColors(iterable_size=4)

        ax = get_ax(figsize=(10, 8))
        plt.grid()
        ax.xaxis.set_major_formatter(formatter)
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            # Lower orbit
            sir_linear = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) &
                                                  (math_results_df['alpha'] == alpha) &
                                                  (math_results_df['h_high'] == h_high), 'sir_linear'])
            sir_linear = sir_linear[:len(time_math)]
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(time_math, y_values, color=colors[i],
                     label=r'SIR$_2$, N={}'.format(n_sats),
                     linewidth=self.font_line_sizes['linewidth'])

            # Higher orbit
            sir_linear = list(high_math_results_df.loc[(high_math_results_df['n_sats'] == n_sats) &
                                                       (high_math_results_df['alpha'] == alpha) &
                                                       (high_math_results_df['h_high'] == h_high), 'sir_linear'])
            sir_linear = sir_linear[:len(time_math)]
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(time_math, y_values, color=colors[i],
                     label=r'SIR$_{2C}$, ' + 'N={sats}'.format(sats=n_sats),
                     linewidth=self.font_line_sizes['linewidth'],
                     linestyle='dashdot')

        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            # Lower orbit
            sir_linear = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) &
                                                 (sim_results_df['alpha'] == alpha) &
                                                 (sim_results_df['h_high'] == h_high), 'sir_linear'])
            sir_linear = sir_linear[:len(time_sim)]
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(time_sim, y_values, color=colors[i],
                        label=r'SIR$_2$ Sim, N={}'.format(n_sats), marker=markers[0],
                        edgecolor='k', zorder=3, s=sizes[0])

            # Higher orbit
            sir_linear = list(high_sim_results_df.loc[(high_sim_results_df['n_sats'] == n_sats) &
                                                      (high_sim_results_df['alpha'] == alpha) &
                                                      (high_sim_results_df['h_high'] == h_high), 'sir_linear'])
            sir_linear = sir_linear[:len(time_sim)]
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.scatter(time_sim, y_values, color=colors[i],
                        label=r'SIR$_{2C}$ Sim, ' + 'N={sats}'.format(sats=n_sats), marker=markers[1],
                        edgecolor='k', zorder=3,
                        s=sizes[1])

        ticks_size = 35
        label_size = 45
        legend_size = 28

        plt.xlabel('Time [h]', fontsize=label_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SIR [dB]', fontsize=label_size,
                   fontname=self.font_line_sizes['fontname'])

        if alpha_list[0] == 30:
            plt.yticks([-40, -35, -30, -25, -20, -15, -10, -5, 0],
                       fontsize=ticks_size,
                       fontname=self.font_line_sizes['fontname'])
            plt.xticks(fontsize=ticks_size,
                       fontname=self.font_line_sizes['fontname'])
            plt.xlim(0, time_math[-1])
            plt.ylim(-42, 0.5)
        elif alpha_list[0] == 10:
            plt.yticks([-15, -10, -5, 0],
                       fontsize=ticks_size,
                       fontname=self.font_line_sizes['fontname'])
            plt.xticks(fontsize=ticks_size,
                       fontname=self.font_line_sizes['fontname'])
            plt.xlim(0, time_math[-1])
            plt.ylim(-19, 0.5)
        else:
            plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                       fontname=self.font_line_sizes['fontname'])
            plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                       fontname=self.font_line_sizes['fontname'])

        plt.legend(framealpha=0.7,
                   fontsize=legend_size,
                   loc='lower center',
                   ncol=2,
                   fancybox=True,
                   columnspacing=0)

        # Adjustemnt of bbox to better fit
        box = ax.get_position()
        ax.set_position([box.x0 * 1.2, box.y0 * 1.34, box.width * 1.06, box.height * 1.1])

        # # Version 1: Legend to the right
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='center left', bbox_to_anchor=(1, 0.5))

        # # Version 2: Legend below
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.29, box.width, box.height * 0.85])
        # plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='upper center',
        #            bbox_to_anchor=(0.5, -0.15), ncol=2, fancybox=True)

        extension = 'journal_coplanar_4_sir_vs_time_high_vs_low' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def journal_coplanar_pdf_interference_time(self, file_ext='.jpg', alpha_list=None, n_sats_list=None, show=True,
                                               bins=200):
        alpha_list = self.link_budget_input_params['alpha_list'] if alpha_list is None else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))

        h_high = self.orbital_input_params['h_lims'][0]
        h_low = self.orbital_input_params['h_low']
        orbit_separation = h_high - h_low

        results_df = self.math_results_df
        higher_orbit_results_df = self.high_math_results_df

        figsize=(10, 8)
        ticks_size = 35
        label_size = 45
        legend_size = 28
        style = ['solid', 'dashdot']
        colors = ['#e41a1c', '#4daf4a', '#377eb8']

        fig = get_ax(figsize=figsize)
        plt.grid()

        for i, alpha in enumerate(alpha_list):
            for j, n_sats in enumerate(n_sats_list):
                x_values = results_df.loc[(results_df['n_sats'] == n_sats) &
                                          (results_df['h_high'] == h_high) &
                                          (results_df['alpha'] == alpha), 'interference']
                # x_values = 10 * np.log10(sir_linear)

                if not x_values.isnull().all():
                    ####### Estimation of the underlying prob density function
                    # # create a probability density function
                    # pdf = gaussian_kde(x_values)
                    #
                    # # create a range of values for the x-axis
                    # x_axis = np.linspace(np.min(x_values), np.max(x_values), 100)
                    #
                    # # plot the PDF
                    # plt.plot(x_axis, pdf(x_axis), color=colors[i], label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                    #          linewidth=self.font_line_sizes['linewidth'])

                    ####### Step histogram
                    # n, bins, patches = plt.hist(x_values, color=colors[i],
                    #                             label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), density=True,
                    #                             linewidth=self.font_line_sizes['linewidth'], histtype='step', zorder=10,
                    #                             bins=40, range=(-85, -50))

                    ###### Curve joining all histogram bar tops
                    # create a histogram
                    # hist, edges = np.histogram(x_values, bins=bins, range=(-25, 0), density=True)
                    hist, edges = np.histogram(x_values, bins=bins, range=(-85, -50), density=True)

                    # find the x and y coordinates of the tips of the histogram bars
                    x = np.concatenate(([-100], edges[:-1] + np.diff(edges) / 2, [-20]))
                    y = np.concatenate((np.zeros(1), hist, np.zeros(1)))

                    # # perform curve fitting to generate a smooth curve
                    # f = interp1d(x, y, kind='cubic')
                    # x_new = np.linspace(x[0], x[-1], num=1000)
                    # y_new = f(x_new)

                    # Deleting outliers manually
                    if alpha == 20 and n_sats == 50:
                        index = int(bins * 0.255) + 1
                        # y[index] = 0.4
                        mask = np.ones(len(x), dtype=bool)
                        mask[[index]] = False
                        x = x[mask, ...]
                        y = y[mask, ...]

                    if alpha == 20 and n_sats == 100:
                        index = int(bins * 0.53) + 1
                        # y[index] = 0.4
                        mask = np.ones(len(x), dtype=bool)
                        mask[[index]] = False
                        x = x[mask, ...]
                        y = y[mask, ...]

                    if alpha == 10 and n_sats == 50:
                        index = int(bins * 0.777) + 1
                        # y[index] = 0.4
                        mask = np.ones(len(x), dtype=bool)
                        mask[[index]] = False
                        x = x[mask, ...]
                        y = y[mask, ...]

                    if alpha == 10 and n_sats == 100:
                        index1 = int(bins * 0.82) + 1
                        index2 = int(bins * 0.817) + 1
                        # y[index1] = 0.4
                        # y[index2] = 0.4
                        mask = np.ones(len(x), dtype=bool)
                        mask[[index1, index2]] = False
                        x = x[mask, ...]
                        y = y[mask, ...]

                    # plot the original histogram and the smooth curve
                    plt.plot(x, y, color=colors[i], label=r'$\alpha$={}$^\circ$'.format(alpha) +'\nN={}'.format(n_sats),
                             linewidth=self.font_line_sizes['linewidth'], linestyle=style[j], zorder= 5-i,
                             marker=markers[j], markersize=sizesplot[j], markevery=10, markeredgecolor='k')

        extension = 'journal_coplanar_5_interference_pdf' + file_ext
        # plt.suptitle('Coplanar orbits: Expected interference probability', fontsize=self.font_line_sizes['titleSize'],
        #              fontname=self.font_line_sizes['fontname'])
        # plt.title('orbit separation = {} km'.format(orbit_separation * 1e-3),
        #           fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Interference [dBm]', fontsize=label_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Probability', fontsize=label_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=ticks_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=ticks_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.xlim(-86, -36)

        # Adjustemnt of bbox to better fit
        box = fig.get_position()
        fig.set_position([box.x0 * 1.27, box.y0 * 1.34, box.width * 1.08, box.height * 1.1])

        plt.legend(fontsize=legend_size, columnspacing=0.5, framealpha=1)
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def journal_coplanar_sinr_vs_orbit_separation(self, file_ext='.jpg', thz_alpha_list=None,
                                                  mmwave_alpha_list=None, n_sats_list=None, show=True):
        thz_alpha_list = self.thz_input_params['alpha_list'] if thz_alpha_list is None else thz_alpha_list
        mmwave_alpha_list = self.mmWave_input_params['alpha_list'] if mmwave_alpha_list is None else mmwave_alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h_low = self.orbital_input_params['h_low']

        # Results dataframes
        thz_math_results_df = self.thz_math_results_df
        # thz_sim_results_df = self.thz_sim_results_df
        mmwave_math_results_df = self.mmWave_math_results_df
        # mmwave_sim_results_df = self.mmWave_sim_results_df

        # Time averaging of results
        thz_math_results_df = thz_math_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(thz_math_results_df)
        mmwave_math_results_df = mmwave_math_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(mmwave_math_results_df)

        # thz_sim_results_df = thz_sim_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        # adjust_db_columns(thz_sim_results_df)
        # mmwave_sim_results_df = mmwave_sim_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        # adjust_db_columns(mmwave_sim_results_df)

        # Orbit separation vectors
        orbit_separation_list_math = np.arange(self.orbital_input_params['h_lims'][0],
                                               self.orbital_input_params['h_lims'][1],
                                               self.orbital_input_params['h_resolution_math'])
        orbit_separation_list_math = orbit_separation_list_math - h_low

        # orbit_separation_list_sim = np.arange(self.orbital_input_params['h_lims'][0],
        #                                       self.orbital_input_params['h_lims'][1],
        #                                       self.orbital_input_params['h_resolution_sim'])
        # orbit_separation_list_sim = orbit_separation_list_sim - h_low

        mmwave_colors = ['#e41a1c', '#ff7f00']  # Red, Orange
        thz_colors = ['#377eb8', '#984ea3']  # Blue, Purple
        linestyles = ['solid', 'dashed', 'dotted']

        ax = get_ax(figsize=(10, 6))
        plt.grid()

        # THz
        # Math
        for i, alpha in enumerate(thz_alpha_list):
            for j, n_sats in enumerate(n_sats_list):
                sinr_linear = list(thz_math_results_df.loc[(thz_math_results_df['n_sats'] == n_sats) &
                                                           (thz_math_results_df['alpha'] == alpha), 'sinr_linear'])
                y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
                plt.plot(orbit_separation_list_math * 1e-3, y_values, color=thz_colors[i],
                         label=r'Sub-THz$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                         linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[j], marker=markers[j],
                         markersize=sizesplot[j], markeredgecolor='k', markevery=5)
        # # Simulation
        # for i, alpha in enumerate(thz_alpha_list):
        #     for j, n_sats in enumerate(n_sats_list):
        #         t = i * len(n_sats_list) + j
        #         sinr_linear = list(thz_sim_results_df.loc[(thz_sim_results_df['n_sats'] == n_sats) &
        #                                                   (thz_sim_results_df['alpha'] == alpha), 'sinr_linear'])
        #         y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
        #         plt.scatter(orbit_separation_list_sim * 1e-3, y_values, color=thz_colors[j],
        #                     label=r'THz Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[t],
        #                     edgecolor='k', zorder=3, s=sizes[t])

        # Theoretical maximum (SNR) as a dashed line, from the math model
        # alpha = min(mmwave_alpha_list)
        # for n_sats in n_sats_list:
        #     snr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['n_sats'] == n_sats) & (
        #             mmwave_math_results_df['alpha'] == alpha), 'snr_linear'])
        #     y_values = [10 * np.log10(value) if value else None for value in snr_linear]
        #     plt.plot(orbit_separation_list_math * 1e-3, y_values, color='k', linestyle='dashed',
        #              label=r'SNR mmWave: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
        #              linewidth=self.font_line_sizes['linewidth'])

        # mmWave
        # Math
        for i, alpha in enumerate(mmwave_alpha_list):
            for j, n_sats in enumerate(n_sats_list):
                sinr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['n_sats'] == n_sats) &
                                                              (mmwave_math_results_df[
                                                                   'alpha'] == alpha), 'sinr_linear'])
                y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
                plt.plot(orbit_separation_list_math * 1e-3, y_values, color=mmwave_colors[i],
                         label=r'mmWave $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                         linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[j], marker=markers[j],
                         markersize=sizesplot[j], markeredgecolor='k', markevery=5)
        # # Simulation
        # for i, alpha in enumerate(mmwave_alpha_list):
        #     for j, n_sats in enumerate(n_sats_list):
        #         t = i * len(n_sats_list) + j
        #         sinr_linear = list(mmwave_sim_results_df.loc[(mmwave_sim_results_df['n_sats'] == n_sats) &
        #                                                      (mmwave_sim_results_df['alpha'] == alpha), 'sinr_linear'])
        #         y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
        #         plt.scatter(orbit_separation_list_sim * 1e-3, y_values, color=mmwave_colors[j],
        #                     label=r'mmWave Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[t],
        #                     edgecolor='k', zorder=3, s=sizes[t])


        fontsize_scale_factor = 1.1
        # plt.suptitle('Coplanar orbits: SINR', fontsize=self.font_line_sizes['titleSize'],
        #              fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Orbit separation [Km]', fontsize=self.font_line_sizes['axisLabelSize'] * fontsize_scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SINR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * fontsize_scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks([10, 100, 200, 300, 400, 500], fontsize=self.font_line_sizes['ticksLabelSize'] * fontsize_scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'] * fontsize_scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xlim(0, 500)
        plt.ylim(-30, 14)

        '''
        # Version 1: Legend to the right
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='center left', bbox_to_anchor=(1, 0.5))
        '''

        # Version 2: Legend below
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.9])
        # plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='upper center',
        #            bbox_to_anchor=(0.5, -0.12), ncol=3, fancybox=True)

        # Version 3: Crunched legend
        box = ax.get_position()
        ax.set_position([box.x0 * 0.83, box.y0 * 1.19, box.width * 1.12, box.height * 1.1])
        leg = plt.legend(ncol=2, fontsize=self.font_line_sizes['legendSize'] * 1.05, columnspacing=0.5,
                         handletextpad=0.1, loc='lower right', framealpha=1)

        extension = 'journal_coplanar_6_sinr_vs_orbit_separation' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def journal_coplanar_capacity_vs_orbit_separation(self, file_ext='.jpg', thz_alpha_list=None,
                                                      mmwave_alpha_list=None, n_sats_list=None, show=True):
        thz_alpha_list = self.thz_input_params['alpha_list'] if thz_alpha_list is None else thz_alpha_list
        mmwave_alpha_list = self.mmWave_input_params['alpha_list'] if mmwave_alpha_list is None else mmwave_alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h_low = self.orbital_input_params['h_low']
        bw = self.mmWave_input_params['bandwidth']

        # Results dataframes
        thz_math_results_df = self.thz_math_results_df
        # thz_sim_results_df = self.thz_sim_results_df
        mmwave_math_results_df = self.mmWave_math_results_df
        # mmwave_sim_results_df = self.mmWave_sim_results_df

        # Time averaging of results
        thz_math_results_df = thz_math_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(thz_math_results_df)
        mmwave_math_results_df = mmwave_math_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        adjust_db_columns(mmwave_math_results_df)

        # thz_sim_results_df = thz_sim_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        # adjust_db_columns(thz_sim_results_df)
        # mmwave_sim_results_df = mmwave_sim_results_df.groupby(['n_sats', 'alpha', 'h_high'], as_index=False).mean()
        # adjust_db_columns(mmwave_sim_results_df)

        # Orbit separation vectors
        orbit_separation_list_math = np.arange(self.orbital_input_params['h_lims'][0],
                                               self.orbital_input_params['h_lims'][1],
                                               self.orbital_input_params['h_resolution_math'])
        orbit_separation_list_math = orbit_separation_list_math - h_low

        # orbit_separation_list_sim = np.arange(self.orbital_input_params['h_lims'][0],
        #                                       self.orbital_input_params['h_lims'][1],
        #                                       self.orbital_input_params['h_resolution_sim'])
        # orbit_separation_list_sim = orbit_separation_list_sim - h_low

        mmwave_colors = ['#e41a1c', '#ff7f00']  # Red, Orange
        thz_colors = ['#377eb8', '#984ea3']  # Blue, Purple
        linestyles = ['solid', 'dashed', 'dotted']

        ax = get_ax()
        plt.grid()

        # Theoretical maximum (B*log2(SNR)) as a dashed line, from the math model
        # alpha = min(mmwave_alpha_list)
        # for n_sats in n_sats_list:
        #     snr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['n_sats'] == n_sats) & (
        #             mmwave_math_results_df['alpha'] == alpha), 'snr_linear'])
        #     y_values = [bw * np.log2(1 + value) * 1e-9 if value != 0 else None for value in snr_linear]
        #     plt.plot(orbit_separation_list_math * 1e-3, y_values, color='k', linestyle='dashed',
        #              label=r'Max capacity mmWave: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
        #              linewidth=self.font_line_sizes['linewidth'])

        # THz
        # Math
        for i, alpha in enumerate(thz_alpha_list):
            for j, n_sats in enumerate(n_sats_list):
                y_values = list(thz_math_results_df.loc[(thz_math_results_df['n_sats'] == n_sats) &
                                                        (thz_math_results_df['alpha'] == alpha), 'capacity_gbps'])
                plt.plot(orbit_separation_list_math * 1e-3, y_values, color=thz_colors[i],
                         label=r'THz: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                         linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[j],  marker=markers[j],
                         markersize=sizesplot[j], markeredgecolor='k', markevery=5, zorder=5)
        # Simulation
        # for i, alpha in enumerate(thz_alpha_list):
        #     for j, n_sats in enumerate(n_sats_list):
        #         t = i * len(n_sats_list) + j
        #         y_values = list(thz_sim_results_df.loc[(thz_sim_results_df['n_sats'] == n_sats) &
        #                                                (thz_sim_results_df['alpha'] == alpha), 'capacity_gbps'])
        #         plt.scatter(orbit_separation_list_sim * 1e-3, y_values, color=thz_colors[j],
        #                     label=r'THz Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[t],
        #                     edgecolor='k', zorder=3, s=sizes[t])

        # mmWave
        # Math
        for i, alpha in enumerate(mmwave_alpha_list):
            for j, n_sats in enumerate(n_sats_list):
                y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['n_sats'] == n_sats) &
                                                           (mmwave_math_results_df[
                                                                'alpha'] == alpha), 'capacity_gbps'])
                plt.plot(orbit_separation_list_math * 1e-3, y_values, color=mmwave_colors[i],
                         label=r'mmWave: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                         linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[j],
                         marker=markers[j],
                         markersize=sizesplot[j], markeredgecolor='k', markevery=5)
        # Simulation
        # for i, alpha in enumerate(mmwave_alpha_list):
        #     for j, n_sats in enumerate(n_sats_list):
        #         t = i * len(n_sats_list) + j
        #         y_values = list(mmwave_sim_results_df.loc[(mmwave_sim_results_df['n_sats'] == n_sats) &
        #                                                   (mmwave_sim_results_df['alpha'] == alpha), 'capacity_gbps'])
        #         plt.scatter(orbit_separation_list_sim * 1e-3, y_values, color=mmwave_colors[j],
        #                     label=r'mmWave Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), marker=markers[t],
        #                     edgecolor='k', zorder=3, s=sizes[t])

        # plt.suptitle('Coplanar orbits: SINR', fontsize=self.font_line_sizes['titleSize'],
        #              fontname=self.font_line_sizes['fontname'])

        fontsize_scale_factor = 1.1
        plt.yscale('log')
        plt.xlabel('Orbit separation [Km]', fontsize=self.font_line_sizes['axisLabelSize'] * fontsize_scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Capacity [Gbps]', fontsize=self.font_line_sizes['axisLabelSize'] * fontsize_scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'] * fontsize_scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks([0.01, 0.1, 1, 10, 60],
                   [0.01, 0.1, 1, 10, 60],
                   fontsize=self.font_line_sizes['ticksLabelSize'] * fontsize_scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.xlim(0, 500)
        plt.ylim(1e-2, 60)

        '''
        # Version 1: Legend to the right
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='center left', bbox_to_anchor=(1, 0.5))
        '''

        # Version 2: Legend below
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.9])
        # plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='upper center',
        #            bbox_to_anchor=(0.5, -0.15), ncol=2, fancybox=True)

        # Version 3: Crunched legend
        box = ax.get_position()
        ax.set_position([box.x0 * 0.86, box.y0 * 1.21, box.width * 1.12, box.height * 1.1])
        leg = plt.legend(ncol=2, fontsize=self.font_line_sizes['legendSize'] * 1.05, columnspacing=0.5,
                         handletextpad=0.1, loc='lower right', framealpha=1)

        plt.legend(framealpha=0.7,
                   fontsize=self.font_line_sizes['legendSize'],
                   loc='lower right',
                   ncol=2,
                   fancybox=True)

        extension = 'journal_coplanar_7_capacity_vs_orbit_separation' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    # Shifted orbits analysis plots
    # X axis: time
    def shifted_tx_rx_distance_vs_time_plot(self, file_ext='.jpg', n_sats_list=None,
                                            show=True, alpha_list=30, inclination=3):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        h = self.orbital_input_params['h_lims'][0]

        # Formatting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[
            (math_results_df['n_sats'] == n_sats_list[0]) & (math_results_df['h'] == h) & (
                    math_results_df['inclination'] == inclination) & (
                    math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time_math[i] - t_start for i in range(len(time_math))]
        time_math = [time_math[i].total_seconds() for i in range(len(time_math))]

        time_sim = sim_results_df.loc[
            (sim_results_df['n_sats'] == n_sats_list[1]) & (sim_results_df['h'] == h) & (
                    sim_results_df['inclination'] == inclination) & (sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time_sim[i] - t_start for i in range(len(time_sim))]
        time_sim = [time_sim[i].total_seconds() for i in range(len(time_sim))]

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[
                           (math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                                   math_results_df['inclination'] == inclination) & (
                                   math_results_df['alpha'] == alpha), 'tx_rx_distance'] * 1e-3
            plt.plot(time_math, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[
                           (sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                                   sim_results_df['inclination'] == inclination) & (
                                   sim_results_df['alpha'] == alpha), 'tx_rx_distance'] * 1e-3
            plt.plot(time_sim, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        plt.suptitle('Shifted orbits: Tx-Rx distance', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km, inclination = {}$^\circ$'.format(h * 1e-3, inclination),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Tx-Rx distance [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_TxRx_distance_vs_time' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def shifted_rx_power_vs_time_plot(self, file_ext='.jpg', n_sats_list=None,
                                      show=True, alpha_list=30, inclination=3):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        h = self.orbital_input_params['h_lims'][0]

        # Formatting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[
            (math_results_df['n_sats'] == n_sats_list[0]) & (math_results_df['h'] == h) & (
                    math_results_df['inclination'] == inclination) & (
                    math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time_math[i] - t_start for i in range(len(time_math))]
        time_math = [time_math[i].total_seconds() for i in range(len(time_math))]

        time_sim = sim_results_df.loc[
            (sim_results_df['n_sats'] == n_sats_list[1]) & (sim_results_df['h'] == h) & (
                    sim_results_df['inclination'] == inclination) & (sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time_sim[i] - t_start for i in range(len(time_sim))]
        time_sim = [time_sim[i].total_seconds() for i in range(len(time_sim))]

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[
                (math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                        math_results_df['inclination'] == inclination) & (
                        math_results_df['alpha'] == alpha), 'rx_power']
            plt.plot(time_math, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[
                (sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                        sim_results_df['inclination'] == inclination) & (
                        sim_results_df['alpha'] == alpha), 'rx_power']
            plt.plot(time_sim, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        plt.suptitle('Shifted orbits: Rx power', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km, inclination = {}$^\circ$'.format(h * 1e-3, inclination),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Rx power [dBm]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_rx_power_vs_time' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def shifted_interference_vs_time_plot(self, file_ext='.jpg', n_sats_list=None,
                                          show=True, alpha_list=30, inclination=3):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        h = self.orbital_input_params['h_lims'][0]

        # Formatting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[
            (math_results_df['n_sats'] == n_sats_list[0]) & (math_results_df['h'] == h) & (
                    math_results_df['inclination'] == inclination) & (
                    math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time_math[i] - t_start for i in range(len(time_math))]
        time_math = [time_math[i].total_seconds() for i in range(len(time_math))]

        time_sim = sim_results_df.loc[
            (sim_results_df['n_sats'] == n_sats_list[1]) & (sim_results_df['h'] == h) & (
                    sim_results_df['inclination'] == inclination) & (sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time_sim[i] - t_start for i in range(len(time_sim))]
        time_sim = [time_sim[i].total_seconds() for i in range(len(time_sim))]

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[
                (math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                        math_results_df['inclination'] == inclination) & (
                        math_results_df['alpha'] == alpha), 'interference']
            plt.plot(time_math, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[
                (sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                        sim_results_df['inclination'] == inclination) & (
                        sim_results_df['alpha'] == alpha), 'interference']
            plt.plot(time_sim, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        plt.suptitle('Shifted orbits: Expected interference', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km, inclination = {}$^\circ$'.format(h * 1e-3, inclination),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('E[I] [dBm]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_interference_vs_time' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def shifted_interference_vs_time_histogram(self, file_ext='.jpg', alpha_list=30,
                                               inclination=3, n_sats_list=None, show=True, math=True):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        h = self.orbital_input_params['h_lims'][0]

        if math:
            results_df = self.math_results_df
        else:
            results_df = self.sim_results_df

        # Foramtting stuff for x axis
        formatter = FuncFormatter(format_func)

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)
        for i, (alpha, n_sats) in enumerate(pairs):
            x_values = results_df.loc[(results_df['n_sats'] == n_sats) & (results_df['h'] == h) & (
                    results_df['inclination'] == inclination) & (results_df['alpha'] == alpha), 'interference']
            if not x_values.isnull().all():
                plt.hist(x_values, color=colors[i],
                         label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), density=True, alpha=0.5,
                         edgecolor='k', zorder=10)

        if math:
            plt.suptitle('Shifted orbits math: Expected interference probability',
                         fontsize=self.font_line_sizes['titleSize'],
                         fontname=self.font_line_sizes['fontname'])
            extension = 'shifted_interference_histogram_math' + file_ext
        else:
            plt.suptitle('Shifted orbits simulation: Expected interference probability',
                         fontsize=self.font_line_sizes['titleSize'],
                         fontname=self.font_line_sizes['fontname'])
            extension = 'shifted_interference_histogram_simulation' + file_ext
        plt.title(r'orbit altitude = {} km, inclination = {}$^\circ$'.format(h * 1e-3, inclination),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('E[I] [dBm]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Probability', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def shifted_sir_vs_time_plot(self, file_ext='.jpg', n_sats_list=None,
                                 show=True, alpha_list=30, inclination=3):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        h = self.orbital_input_params['h_lims'][0]

        # Formatting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[
            (math_results_df['n_sats'] == n_sats_list[0]) & (math_results_df['h'] == h) & (
                    math_results_df['inclination'] == inclination) & (
                    math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time_math[i] - t_start for i in range(len(time_math))]
        time_math = [time_math[i].total_seconds() for i in range(len(time_math))]

        time_sim = sim_results_df.loc[
            (sim_results_df['n_sats'] == n_sats_list[1]) & (sim_results_df['h'] == h) & (
                    sim_results_df['inclination'] == inclination) & (sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time_sim[i] - t_start for i in range(len(time_sim))]
        time_sim = [time_sim[i].total_seconds() for i in range(len(time_sim))]

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            interference = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                        math_results_df['inclination'] == inclination) & (
                                            math_results_df['alpha'] == alpha), 'interference'])
            rx_power = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                        math_results_df['inclination'] == inclination) & (
                                            math_results_df['alpha'] == alpha), 'rx_power'])
            y_values = [rx_power[i] - interference[i] if interference[i] else None for i in
                        range(len(rx_power))]
            plt.plot(time_math, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            interference = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                        sim_results_df['inclination'] == inclination) & (
                                           sim_results_df['alpha'] == alpha), 'interference'])
            rx_power = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                        sim_results_df['inclination'] == inclination) & (
                                           sim_results_df['alpha'] == alpha), 'rx_power'])
            y_values = [rx_power[i] - interference[i] if interference[i] else None for i in
                        range(len(rx_power))]
            plt.plot(time_sim, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        plt.suptitle('Shifted orbits: Signal to Interference Ratio (SIR)', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km, inclination = {}$^\circ$'.format(h * 1e-3, inclination),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_sir_vs_time' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def shifted_n_interferers_vs_time_plot(self, file_ext='.jpg', n_sats_list=None,
                                           show=True, alpha_list=30, inclination=3):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        h = self.orbital_input_params['h_lims'][0]

        # Formatting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[
            (math_results_df['n_sats'] == n_sats_list[0]) & (math_results_df['h'] == h) & (
                    math_results_df['inclination'] == inclination) & (
                    math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time_math[i] - t_start for i in range(len(time_math))]
        time_math = [time_math[i].total_seconds() for i in range(len(time_math))]

        time_sim = sim_results_df.loc[
            (sim_results_df['n_sats'] == n_sats_list[1]) & (sim_results_df['h'] == h) & (
                    sim_results_df['inclination'] == inclination) & (sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time_sim[i] - t_start for i in range(len(time_sim))]
        time_sim = [time_sim[i].total_seconds() for i in range(len(time_sim))]

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[
                (math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                        math_results_df['inclination'] == inclination) & (
                        math_results_df['alpha'] == alpha), 'n_interferers']
            plt.plot(time_math, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[
                (sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                        sim_results_df['inclination'] == inclination) & (
                        sim_results_df['alpha'] == alpha), 'n_interferers']
            plt.plot(time_sim, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        plt.suptitle('Shifted orbits: Expected interference', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km, inclination = {}$^\circ$'.format(h * 1e-3, inclination),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Number of interferers', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_n_interferers_vs_time' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def shifted_distance_to_nearest_interferer_vs_time_plot(self, file_ext='.jpg', n_sats_list=None,
                                                            show=True, alpha_list=30, inclination=3):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        h = self.orbital_input_params['h_lims'][0]

        # Formatting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[
            (math_results_df['n_sats'] == n_sats_list[0]) & (math_results_df['h'] == h) & (
                    math_results_df['inclination'] == inclination) & (
                    math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time_math[i] - t_start for i in range(len(time_math))]
        time_math = [time_math[i].total_seconds() for i in range(len(time_math))]

        time_sim = sim_results_df.loc[
            (sim_results_df['n_sats'] == n_sats_list[1]) & (sim_results_df['h'] == h) & (
                    sim_results_df['inclination'] == inclination) & (sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time_sim[i] - t_start for i in range(len(time_sim))]
        time_sim = [time_sim[i].total_seconds() for i in range(len(time_sim))]

        _, _, colors = tk.templateColorsPairs(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)
        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[
                           (math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                                   math_results_df['inclination'] == inclination) & (
                                   math_results_df['alpha'] == alpha), 'd_interferer'] * 1e-3
            plt.plot(time_math, y_values, color='r',
                     label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[
                           (sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                                   sim_results_df['inclination'] == inclination) & (
                                   sim_results_df['alpha'] == alpha), 'd_interferer'] * 1e-3
            plt.plot(time_sim, y_values, color='k',
                     label=r'Sim: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats), markersize=5, markevery=None)
        plt.suptitle('Shifted orbits: Distance to nearest interferer', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km, inclination = {}$^\circ$'.format(h * 1e-3, inclination),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Distance [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_d_closest_interferer_vs_time' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    # X axis: Beamwidth
    def shifted_tx_rx_distance_vs_beamwidth_plot(self, file_ext='.jpg', n_sats_list=None,
                                                 show=True, inclination=3):
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h = self.orbital_input_params['h_lims'][0]

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                  as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                as_index=False).mean()
        adjust_db_columns(sim_results_df)

        alpha_list_math = np.arange(self.orbital_input_params['alpha_lims'][0],
                                    self.orbital_input_params['alpha_lims'][1],
                                    self.orbital_input_params['alpha_resolution_math'])
        alpha_list_sim = np.arange(self.orbital_input_params['alpha_lims'][0],
                                   self.orbital_input_params['alpha_lims'][1],
                                   self.orbital_input_params['alpha_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(n_sats_list))

        fig = get_ax()
        plt.grid()
        # math
        for i, n_sats in enumerate(n_sats_list):
            y_values = math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                    math_results_df['inclination'] == inclination), 'tx_rx_distance'] * 1e-3
            plt.plot(alpha_list_math, y_values, color='r',
                     label=r'N={}'.format(n_sats), markersize=5, markevery=None)
        # simulation
        for i, n_sats in enumerate(n_sats_list):
            y_values = sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                    sim_results_df['inclination'] == inclination), 'tx_rx_distance'] * 1e-3
            plt.plot(alpha_list_sim, y_values, color='k',
                     label=r'Sim.: N={}'.format(n_sats), markersize=5, markevery=None)

        plt.suptitle('Shifted orbits: Tx-Rx distance', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km, inclination = {}$^\circ$'.format(h * 1e-3, inclination),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel(r'Beamwidth [$^\circ$]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Tx-Rx distance [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_TxRx_distance_vs_beamwidth' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def shifted_rx_power_vs_beamwidth_plot(self, file_ext='.jpg', n_sats_list=None,
                                           show=True, inclination=3):
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h = self.orbital_input_params['h_lims'][0]

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                  as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                as_index=False).mean()
        adjust_db_columns(sim_results_df)

        alpha_list_math = np.arange(self.orbital_input_params['alpha_lims'][0],
                                    self.orbital_input_params['alpha_lims'][1],
                                    self.orbital_input_params['alpha_resolution_math'])
        alpha_list_sim = np.arange(self.orbital_input_params['alpha_lims'][0],
                                   self.orbital_input_params['alpha_lims'][1],
                                   self.orbital_input_params['alpha_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(n_sats_list))

        fig = get_ax()
        plt.grid()
        # math
        for i, n_sats in enumerate(n_sats_list):
            y_values = math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                    math_results_df['inclination'] == inclination), 'rx_power']
            plt.plot(alpha_list_math, y_values, color='r',
                     label=r'N={}'.format(n_sats), markersize=5, markevery=None)
        # simulation
        for i, n_sats in enumerate(n_sats_list):
            y_values = sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                    sim_results_df['inclination'] == inclination), 'rx_power']
            plt.plot(alpha_list_sim, y_values, color='k',
                     label=r'Sim.: N={}'.format(n_sats), markersize=5, markevery=None)

        plt.suptitle('Shifted orbits: Rx power', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km, inclination = {}$^\circ$'.format(h * 1e-3, inclination),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel(r'Beamwidth [$^\circ$]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Rx power [dBm]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_rx_power_vs_beamwidth' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def shifted_interference_vs_beamwidth_plot(self, file_ext='.jpg', n_sats_list=None,
                                               show=True, inclination=3):
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h = self.orbital_input_params['h_lims'][0]

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                  as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                as_index=False).mean()
        adjust_db_columns(sim_results_df)

        alpha_list_math = np.arange(self.orbital_input_params['alpha_lims'][0],
                                    self.orbital_input_params['alpha_lims'][1],
                                    self.orbital_input_params['alpha_resolution_math'])
        alpha_list_sim = np.arange(self.orbital_input_params['alpha_lims'][0],
                                   self.orbital_input_params['alpha_lims'][1],
                                   self.orbital_input_params['alpha_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(n_sats_list))

        fig = get_ax()
        plt.grid()
        # math
        for i, n_sats in enumerate(n_sats_list):
            y_values = math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                    math_results_df['inclination'] == inclination), 'interference']
            plt.plot(alpha_list_math, y_values, color='r',
                     label=r'N={}'.format(n_sats), markersize=5, markevery=None)
        # simulation
        for i, n_sats in enumerate(n_sats_list):
            y_values = sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                    sim_results_df['inclination'] == inclination), 'interference']
            plt.plot(alpha_list_sim, y_values, color='k',
                     label=r'Sim.: N={}'.format(n_sats), markersize=5, markevery=None)

        plt.suptitle('Shifted orbits: Expected Interference', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km, inclination = {}$^\circ$'.format(h * 1e-3, inclination),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel(r'Beamwidth [$^\circ$]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('E[I] [dBm]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_interference_vs_beamwidth' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def shifted_sir_vs_beamwidth_plot(self, file_ext='.jpg', n_sats_list=None,
                                      show=True, inclination=3):
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h = self.orbital_input_params['h_lims'][0]

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                  as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                as_index=False).mean()
        adjust_db_columns(sim_results_df)

        alpha_list_math = np.arange(self.orbital_input_params['alpha_lims'][0],
                                    self.orbital_input_params['alpha_lims'][1],
                                    self.orbital_input_params['alpha_resolution_math'])
        alpha_list_sim = np.arange(self.orbital_input_params['alpha_lims'][0],
                                   self.orbital_input_params['alpha_lims'][1],
                                   self.orbital_input_params['alpha_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(n_sats_list))

        fig = get_ax()
        plt.grid()
        # math
        for i, n_sats in enumerate(n_sats_list):
            interference = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                        math_results_df['inclination'] == inclination), 'interference'])
            rx_power = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                        math_results_df['inclination'] == inclination), 'rx_power'])
            y_values = [rx_power[i] - interference[i] if interference[i] else None for i in range(len(rx_power))]
            plt.plot(alpha_list_math, y_values, color='r',
                     label=r'N={}'.format(n_sats), markersize=5, markevery=None)
        # simulation
        for i, n_sats in enumerate(n_sats_list):
            interference = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                        sim_results_df['inclination'] == inclination), 'interference'])
            rx_power = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                        sim_results_df['inclination'] == inclination), 'rx_power'])
            y_values = [rx_power[i] - interference[i] if interference[i] else None for i in range(len(rx_power))]
            plt.plot(alpha_list_sim, y_values, color='k',
                     label=r'Sim.: N={}'.format(n_sats), markersize=5, markevery=None)

        plt.suptitle('Shifted orbits: Signal to Interference Ratio (SIR)', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km, inclination = {}$^\circ$'.format(h * 1e-3, inclination),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel(r'Beamwidth [$^\circ$]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_sir_vs_beamwidth' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def shifted_n_interferers_vs_beamwidth_plot(self, file_ext='.jpg', n_sats_list=None,
                                                show=True, inclination=3):
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h = self.orbital_input_params['h_lims'][0]

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                  as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                as_index=False).mean()
        adjust_db_columns(sim_results_df)

        alpha_list_math = np.arange(self.orbital_input_params['alpha_lims'][0],
                                    self.orbital_input_params['alpha_lims'][1],
                                    self.orbital_input_params['alpha_resolution_math'])
        alpha_list_sim = np.arange(self.orbital_input_params['alpha_lims'][0],
                                   self.orbital_input_params['alpha_lims'][1],
                                   self.orbital_input_params['alpha_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(n_sats_list))

        fig = get_ax()
        plt.grid()
        # math
        for i, n_sats in enumerate(n_sats_list):
            y_values = math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                    math_results_df['inclination'] == inclination), 'n_interferers']
            plt.plot(alpha_list_math, y_values, color='r',
                     label=r'N={}'.format(n_sats), markersize=5, markevery=None)
        # simulation
        for i, n_sats in enumerate(n_sats_list):
            y_values = sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                    sim_results_df['inclination'] == inclination), 'n_interferers']
            plt.plot(alpha_list_sim, y_values, color='k',
                     label=r'Sim.: N={}'.format(n_sats), markersize=5, markevery=None)

        plt.suptitle('Shifted orbits:\nNumber of satellites contributing to interference at the receiver',
                     fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km, inclination = {}$^\circ$'.format(h * 1e-3, inclination),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel(r'Beamwidth [$^\circ$]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Number of interferers', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_n_interferers_vs_beamwidth' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def shifted_distance_to_nearest_interferer_vs_beamwidth_plot(self, file_ext='.jpg', n_sats_list=None,
                                                                 show=True, inclination=3):
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h = self.orbital_input_params['h_lims'][0]

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                  as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                as_index=False).mean()
        adjust_db_columns(sim_results_df)

        alpha_list_math = np.arange(self.orbital_input_params['alpha_lims'][0],
                                    self.orbital_input_params['alpha_lims'][1],
                                    self.orbital_input_params['alpha_resolution_math'])
        alpha_list_sim = np.arange(self.orbital_input_params['alpha_lims'][0],
                                   self.orbital_input_params['alpha_lims'][1],
                                   self.orbital_input_params['alpha_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(n_sats_list))

        fig = get_ax()
        plt.grid()
        # math
        for i, n_sats in enumerate(n_sats_list):
            y_values = math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                    math_results_df['inclination'] == inclination), 'd_interferer']
            plt.plot(alpha_list_math, y_values, color='r',
                     label=r'N={}'.format(n_sats), markersize=5, markevery=None)
        # simulation
        for i, n_sats in enumerate(n_sats_list):
            y_values = sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                    sim_results_df['inclination'] == inclination), 'd_interferer']
            plt.plot(alpha_list_sim, y_values, color='k',
                     label=r'Sim.: N={}'.format(n_sats), markersize=5, markevery=None)

        plt.suptitle('Shifted orbits: Distance to closest interferer', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km, inclination = {}$^\circ$'.format(h * 1e-3, inclination),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel(r'Beamwidth [$^\circ$]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Distance [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_s_closest_interferer_vs_beamwidth' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    # X axis: inclination
    def shifted_tx_rx_distance_vs_inclination_plot(self, file_ext='.jpg', n_sats_list=None,
                                                   show=True, alpha_list=30):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        h = self.orbital_input_params['h_lims'][0]

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                  as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                as_index=False).mean()
        adjust_db_columns(sim_results_df)

        inclination_list_math = np.arange(self.orbital_input_params['inclination_lims'][0],
                                          self.orbital_input_params['inclination_lims'][1],
                                          self.orbital_input_params['inclination_resolution_math'])
        inclination_list_sim = np.arange(self.orbital_input_params['inclination_lims'][0],
                                         self.orbital_input_params['inclination_lims'][1],
                                         self.orbital_input_params['inclination_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(n_sats_list))

        fig = get_ax()
        plt.grid()
        # math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                    math_results_df['alpha'] == alpha), 'tx_rx_distance'] * 1e-3
            plt.plot(inclination_list_math, y_values, color='r',
                     label=r'N={}'.format(n_sats), markersize=5, markevery=None)
        # simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                    sim_results_df['alpha'] == alpha), 'tx_rx_distance'] * 1e-3
            plt.plot(inclination_list_sim, y_values, color='k',
                     label=r'Sim.: N={}'.format(n_sats), markersize=5, markevery=None)

        plt.suptitle('Shifted orbits: Tx-Rx distance', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km'.format(h * 1e-3),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel(r'Inclination [$^\circ$]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Tx-Rx distance [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_TxRx_distance_vs_inclination' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def shifted_rx_power_vs_inclination_plot(self, file_ext='.jpg', n_sats_list=None,
                                             show=True, alpha_list=30):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        h = self.orbital_input_params['h_lims'][0]

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                  as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                as_index=False).mean()
        adjust_db_columns(sim_results_df)

        inclination_list_math = np.arange(self.orbital_input_params['inclination_lims'][0],
                                          self.orbital_input_params['inclination_lims'][1],
                                          self.orbital_input_params['inclination_resolution_math'])
        inclination_list_sim = np.arange(self.orbital_input_params['inclination_lims'][0],
                                         self.orbital_input_params['inclination_lims'][1],
                                         self.orbital_input_params['inclination_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(n_sats_list))

        fig = get_ax()
        plt.grid()
        # math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                    math_results_df['alpha'] == alpha), 'rx_power']
            plt.plot(inclination_list_math, y_values, color='r',
                     label=r'N={}'.format(n_sats), markersize=5, markevery=None)
        # simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                    sim_results_df['alpha'] == alpha), 'rx_power']
            plt.plot(inclination_list_sim, y_values, color='k',
                     label=r'Sim.: N={}'.format(n_sats), markersize=5, markevery=None)

        plt.suptitle('Shifted orbits: Rx power', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km'.format(h * 1e-3),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel(r'Inclination [$^\circ$]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Rx power [dBm]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_rx_power_vs_inclination' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def shifted_interference_vs_inclination_plot(self, file_ext='.jpg', n_sats_list=None,
                                                 show=True, alpha_list=30):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        h = self.orbital_input_params['h_lims'][0]

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                  as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                as_index=False).mean()
        adjust_db_columns(sim_results_df)

        inclination_list_math = np.arange(self.orbital_input_params['inclination_lims'][0],
                                          self.orbital_input_params['inclination_lims'][1],
                                          self.orbital_input_params['inclination_resolution_math'])
        inclination_list_sim = np.arange(self.orbital_input_params['inclination_lims'][0],
                                         self.orbital_input_params['inclination_lims'][1],
                                         self.orbital_input_params['inclination_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(n_sats_list))

        fig = get_ax()
        plt.grid()
        # math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                    math_results_df['alpha'] == alpha), 'interference']
            plt.plot(inclination_list_math, y_values, color='r',
                     label=r'N={}'.format(n_sats), markersize=5, markevery=None)
        # simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                    sim_results_df['alpha'] == alpha), 'interference']
            plt.plot(inclination_list_sim, y_values, color='k',
                     label=r'Sim.: N={}'.format(n_sats), markersize=5, markevery=None)

        plt.suptitle('Shifted orbits: Expected interference', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km'.format(h * 1e-3),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel(r'Inclination [$^\circ$]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('E[I] [dBm]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_interference_vs_inclination' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def shifted_sir_vs_inclination_plot(self, file_ext='.jpg', n_sats_list=None,
                                        show=True, alpha_list=30):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        h = self.orbital_input_params['h_lims'][0]

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                  as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                as_index=False).mean()
        adjust_db_columns(sim_results_df)

        inclination_list_math = np.arange(self.orbital_input_params['inclination_lims'][0],
                                          self.orbital_input_params['inclination_lims'][1],
                                          self.orbital_input_params['inclination_resolution_math'])
        inclination_list_sim = np.arange(self.orbital_input_params['inclination_lims'][0],
                                         self.orbital_input_params['inclination_lims'][1],
                                         self.orbital_input_params['inclination_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(n_sats_list))

        fig = get_ax()
        plt.grid()
        # math
        for i, (alpha, n_sats) in enumerate(pairs):
            interference = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                        math_results_df['alpha'] == alpha), 'interference'])
            rx_power = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                        math_results_df['alpha'] == alpha), 'rx_power'])
            y_values = [rx_power[i] - interference[i] if interference[i] else None for i in
                        range(len(rx_power))]
            plt.plot(inclination_list_math, y_values, color='r',
                     label=r'N={}'.format(n_sats), markersize=5, markevery=None)
        # simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            interference = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                        sim_results_df['alpha'] == alpha), 'interference'])
            rx_power = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                        sim_results_df['alpha'] == alpha), 'rx_power'])
            y_values = [rx_power[i] - interference[i] if interference[i] else None for i in
                        range(len(rx_power))]
            plt.plot(inclination_list_sim, y_values, color='k',
                     label=r'Sim.: N={}'.format(n_sats), markersize=5, markevery=None)

        plt.suptitle('Shifted orbits: Signal to Interference Ratio (SIR)', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km'.format(h * 1e-3),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel(r'Inclination [$^\circ$]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_sir_vs_inclination' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def shifted_n_interferers_vs_inclination_plot(self, file_ext='.jpg', n_sats_list=None,
                                                  show=True, alpha_list=3):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        h = self.orbital_input_params['h_lims'][0]

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                  as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                as_index=False).mean()
        adjust_db_columns(sim_results_df)

        inclination_list_math = np.arange(self.orbital_input_params['inclination_lims'][0],
                                          self.orbital_input_params['inclination_lims'][1],
                                          self.orbital_input_params['inclination_resolution_math'])
        inclination_list_sim = np.arange(self.orbital_input_params['inclination_lims'][0],
                                         self.orbital_input_params['inclination_lims'][1],
                                         self.orbital_input_params['inclination_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(n_sats_list))

        fig = get_ax()
        plt.grid()
        # math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                    math_results_df['alpha'] == alpha), 'n_interferers']
            plt.plot(inclination_list_math, y_values, color='r',
                     label=r'N={}'.format(n_sats), markersize=5, markevery=None)
        # simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                    sim_results_df['alpha'] == alpha), 'n_interferers']
            plt.plot(inclination_list_sim, y_values, color='k',
                     label=r'Sim.: N={}'.format(n_sats), markersize=5, markevery=None)

        plt.suptitle('Shifted orbits:\nNumber of satellites contributing to interference at the receiver',
                     fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km'.format(h * 1e-3),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel(r'Inclination [$^\circ$]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Number of interferers', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_n_interferers_vs_inclination' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    def shifted_distance_to_nearest_interferer_vs_inclination_plot(self, file_ext='.jpg', n_sats_list=None,
                                                                   show=True, alpha_list=3):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        h = self.orbital_input_params['h_lims'][0]

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                  as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                as_index=False).mean()
        adjust_db_columns(sim_results_df)

        inclination_list_math = np.arange(self.orbital_input_params['inclination_lims'][0],
                                          self.orbital_input_params['inclination_lims'][1],
                                          self.orbital_input_params['inclination_resolution_math'])
        inclination_list_sim = np.arange(self.orbital_input_params['inclination_lims'][0],
                                         self.orbital_input_params['inclination_lims'][1],
                                         self.orbital_input_params['inclination_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(n_sats_list))

        fig = get_ax()
        plt.grid()
        # math
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                    math_results_df['alpha'] == alpha), 'd_interferer']
            plt.plot(inclination_list_math, y_values, color='r',
                     label=r'N={}'.format(n_sats), markersize=5, markevery=None)
        # simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            y_values = sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                    sim_results_df['alpha'] == alpha), 'd_interferer']
            plt.plot(inclination_list_sim, y_values, color='k',
                     label=r'Sim.: N={}'.format(n_sats), markersize=5, markevery=None)

        plt.suptitle('Shifted orbits: Distance to closest interferer', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km'.format(h * 1e-3),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        plt.xlabel(r'Inclination [$^\circ$]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Distance [Km]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend()
        if show:
            plt.show()
        extension = 'shifted_d_closest_interferer_vs_inclination' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        plt.close()

    # Paper plots
    def shifted_sir_vs_time_plot_paper(self, file_ext='.jpg', n_sats_list=None, show=True, alpha_list=30,
                                       inclination=3):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        h = self.orbital_input_params['h_lims'][0]

        # Formatting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[
            (math_results_df['n_sats'] == n_sats_list[0]) & (math_results_df['h'] == h) & (
                    math_results_df['inclination'] == inclination) & (
                    math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time_math[i] - t_start for i in range(len(time_math))]
        time_math = [time_math[i].total_seconds() for i in range(len(time_math))]

        time_sim = sim_results_df.loc[
            (sim_results_df['n_sats'] == n_sats_list[0]) & (sim_results_df['h'] == h) & (
                    sim_results_df['inclination'] == inclination) & (sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time_sim[i] - t_start for i in range(len(time_sim))]
        time_sim = [time_sim[i].total_seconds() for i in range(len(time_sim))]

        _, _, colors = tk.templateColors(iterable_size=len(pairs))

        fig = get_ax(figsize=(10, 8))
        fig.xaxis.set_major_formatter(formatter)
        pairs.reverse()
        np.flip(colors)
        for i, (alpha, n_sats) in enumerate(pairs):
            color = colors[i]
            # MATH
            interference = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                        math_results_df['inclination'] == inclination) & (
                                            math_results_df['alpha'] == alpha), 'interference'])
            rx_power = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                        math_results_df['inclination'] == inclination) & (
                                            math_results_df['alpha'] == alpha), 'rx_power'])
            SIR = [rx_power[i] - interference[i] if interference[i] else None for i in
                   range(len(rx_power))]
            plt.plot(time_math, SIR, color=color,
                     label=r'N = {}'.format(n_sats), linewidth=3)

        for i, (alpha, n_sats) in enumerate(pairs):
            color = colors[i]
            # SIMULATION
            interference = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                        sim_results_df['inclination'] == inclination) & (
                                           sim_results_df['alpha'] == alpha), 'interference'])
            rx_power = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                        sim_results_df['inclination'] == inclination) & (
                                           sim_results_df['alpha'] == alpha), 'rx_power'])
            SIR = [rx_power[i] - interference[i] if interference[i] else None for i in
                   range(len(rx_power))]
            plt.scatter(time_sim, SIR, color=color,
                        label=r'Sim.: N = {}'.format(n_sats), marker=markers[i], edgecolors='k', zorder=3, s=sizes[i])

        # plt.suptitle('Shifted orbits: Signal to Interference Ratio (SIR)', fontsize=18)
        # plt.title('h = {} km, Ang. offset = {} deg, inc = {} deg'.format(h_plots * 1e-3, beta_plots, inc_plots), fontsize=15)
        plt.xlabel('Time [h]', fontsize=35, fontname="Times New Roman")
        plt.ylabel('SIR [dB]', fontsize=35, fontname="Times New Roman")
        plt.grid()
        plt.xticks(fontsize=27, fontname="Times New Roman")
        plt.yticks(fontsize=27, fontname="Times New Roman")
        plt.legend(ncol=2, framealpha=0.7, fontsize=20, loc='lower right')
        if show:
            plt.show()
        extension = 'shifted_sir_vs_time_paper' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        plt.close()

    def shifted_sir_vs_beamwidth_plot_paper(self, file_ext='.jpg', n_sats_list=None,
                                            show=True, inclination=3):
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        h = self.orbital_input_params['h_lims'][0]

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                  as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                as_index=False).mean()
        adjust_db_columns(sim_results_df)

        alpha_list_math = np.arange(self.orbital_input_params['alpha_lims'][0],
                                    self.orbital_input_params['alpha_lims'][1],
                                    self.orbital_input_params['alpha_resolution_math'])
        alpha_list_sim = np.arange(self.orbital_input_params['alpha_lims'][0],
                                   self.orbital_input_params['alpha_lims'][1],
                                   self.orbital_input_params['alpha_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(n_sats_list))

        fig = get_ax(figsize=(10, 8))
        n_sats_list.reverse()
        np.flip(colors)
        for i, n_sats in enumerate(n_sats_list):
            color = colors[i]
            # MATH
            interference = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                        math_results_df['inclination'] == inclination), 'interference'])
            rx_power = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                        math_results_df['inclination'] == inclination), 'rx_power'])
            SIR = [rx_power[i] - interference[i] if interference[i] else None for i in
                   range(len(rx_power))]
            plt.plot(alpha_list_math, SIR, color=color,
                     label=r'N = {}'.format(n_sats), linewidth=3)

        for i, n_sats in enumerate(n_sats_list):
            color = colors[i]
            # SIMULATION
            interference = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                        sim_results_df['inclination'] == inclination), 'interference'])
            rx_power = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                        sim_results_df['inclination'] == inclination), 'rx_power'])
            SIR = [rx_power[i] - interference[i] if interference[i] else None for i in
                   range(len(rx_power))]
            plt.scatter(alpha_list_sim, SIR, color=color,
                        label=r'Sim.: N = {}'.format(n_sats), marker=markers[i], edgecolors='k', zorder=3, s=sizes[i])
        '''
        plt.suptitle('Shifted orbits: Signal to Interference Ratio (SIR)', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km, inclination = {}$^\circ$'.format(h * 1e-3, inclination),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        '''
        plt.xlabel(r'Beamwidth [$^\circ$]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.grid()
        plt.xticks(fontsize=27, fontname="Times New Roman")
        plt.yticks(fontsize=27, fontname="Times New Roman")
        plt.legend(ncol=2, framealpha=0.7, fontsize=20, loc='lower right')
        if show:
            plt.show()
        extension = 'shifted_sir_vs_beamwidth_paper' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        plt.close()

    def shifted_sir_vs_inclination_plot_paper(self, file_ext='.jpg', n_sats_list=None,
                                              show=True, alpha_list=30):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        h = self.orbital_input_params['h_lims'][0]

        math_results_df = self.math_results_df
        math_results_df = math_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                  as_index=False).mean()
        adjust_db_columns(math_results_df)

        sim_results_df = self.sim_results_df
        sim_results_df = sim_results_df.groupby(['alpha', 'n_sats', 'h', 'inclination', 'beta'],
                                                as_index=False).mean()
        adjust_db_columns(sim_results_df)

        inclination_list_math = np.arange(self.orbital_input_params['inclination_lims'][0],
                                          self.orbital_input_params['inclination_lims'][1],
                                          self.orbital_input_params['inclination_resolution_math'])
        inclination_list_sim = np.arange(self.orbital_input_params['inclination_lims'][0],
                                         self.orbital_input_params['inclination_lims'][1],
                                         self.orbital_input_params['inclination_resolution_sim'])

        _, _, colors = tk.templateColors(iterable_size=len(n_sats_list))

        fig = get_ax(figsize=(10, 8))
        pairs.reverse()
        np.flip(colors)
        for i, (alpha, n_sats) in enumerate(pairs):
            color = colors[i]
            # MATH
            interference = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                        math_results_df['alpha'] == alpha), 'interference'])
            rx_power = list(
                math_results_df.loc[(math_results_df['n_sats'] == n_sats) & (math_results_df['h'] == h) & (
                        math_results_df['alpha'] == alpha), 'rx_power'])
            SIR = [rx_power[i] - interference[i] if interference[i] else None for i in
                   range(len(rx_power))]
            plt.plot(inclination_list_math, SIR, color=color,
                     label=r'N = {}'.format(n_sats), linewidth=3)

        for i, (alpha, n_sats) in enumerate(pairs):
            color = colors[i]
            # SIMULATION
            interference = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                        sim_results_df['alpha'] == alpha), 'interference'])
            rx_power = list(
                sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) & (sim_results_df['h'] == h) & (
                        sim_results_df['alpha'] == alpha), 'rx_power'])
            SIR = [rx_power[i] - interference[i] if interference[i] else None for i in
                   range(len(rx_power))]
            plt.scatter(inclination_list_sim, SIR, color=color,
                        label=r'Sim.: N = {}'.format(n_sats), marker=markers[i], edgecolors='k', zorder=3, s=sizes[i])
        '''
        plt.suptitle('Shifted orbits: Signal to Interference Ratio (SIR)', fontsize=self.font_line_sizes['titleSize'],
                     fontname=self.font_line_sizes['fontname'])
        plt.title(r'orbit altitude = {} km'.format(h * 1e-3),
                  fontsize=self.font_line_sizes['subTitleSize'], fontname=self.font_line_sizes['fontname'])
        '''
        plt.xlabel(r'Inclination [$^\circ$]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.grid()
        plt.xticks(fontsize=27, fontname="Times New Roman")
        plt.yticks(fontsize=27, fontname="Times New Roman")
        plt.legend(ncol=2, framealpha=0.7, fontsize=20, loc='lower right')
        if show:
            plt.show()
        extension = 'shifted_sir_vs_inclination_paper' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        plt.close()

    # Journal paper plots
    def journal_shifted_sir_vs_time(self, file_ext='.jpg', n_sats_list=None, show=True, alpha_list=30, inclination=3):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        math_results_df = self.math_results_df
        sim_results_df = self.sim_results_df
        h = self.orbital_input_params['h_lims'][0]

        # Formatting stuff for x axis
        formatter = FuncFormatter(format_func)

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time_math = math_results_df.loc[(math_results_df['n_sats'] == n_sats_list[0]) &
                                        (math_results_df['h'] == h) &
                                        (math_results_df['inclination'] == inclination) &
                                        (math_results_df['alpha'] == alpha_list[0]), 'time']
        time_math = time_math.to_list()
        time_math = [time_math[i] - t_start for i in range(len(time_math))]
        time_math = [time_math[i].total_seconds() for i in range(len(time_math))]

        time_sim = sim_results_df.loc[(sim_results_df['n_sats'] == n_sats_list[0]) &
                                      (sim_results_df['h'] == h) &
                                      (sim_results_df['inclination'] == inclination) &
                                      (sim_results_df['alpha'] == alpha_list[0]), 'time']
        time_sim = time_sim.to_list()
        time_sim = [time_sim[i] - t_start for i in range(len(time_sim))]
        time_sim = [time_sim[i].total_seconds() for i in range(len(time_sim))]

        _, _, colors = tk.templateColors(iterable_size=len(pairs))

        figsize = (10, 8)
        ticks_size = 35
        label_size = 45
        legend_size = 28

        fig = get_ax(figsize=figsize)
        plt.grid()
        fig.xaxis.set_major_formatter(formatter)

        time_shift_multiplier = int(
            self.orbital_input_params['n_timesteps_math'] / self.orbital_input_params['n_timesteps_sim'])
        time_shift = 1

        # Math
        for i, (alpha, n_sats) in enumerate(pairs):
            sir_linear = list(math_results_df.loc[(math_results_df['n_sats'] == n_sats) &
                                                  (math_results_df['h'] == h) &
                                                  (math_results_df['inclination'] == inclination) &
                                                  (math_results_df['alpha'] == alpha), 'sir_linear'])
            sir_linear = sir_linear[:len(time_math)]
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            y_values = np.roll(y_values, time_shift * time_shift_multiplier)
            plt.plot(time_math, y_values, color=colors[i], label=r'N = {}'.format(n_sats), linewidth=3)

        # Simulation
        for i, (alpha, n_sats) in enumerate(pairs):
            sir_linear = list(sim_results_df.loc[(sim_results_df['n_sats'] == n_sats) &
                                                 (sim_results_df['h'] == h) &
                                                 (sim_results_df['inclination'] == inclination) &
                                                 (sim_results_df['alpha'] == alpha), 'sir_linear'])
            sir_linear = sir_linear[:len(time_math)]
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            y_values = np.roll(y_values, time_shift)
            plt.scatter(time_sim, y_values, color=colors[i],
                        label=r'Sim: N = {}'.format(n_sats), marker=markers[i], edgecolors='k', zorder=3, s=sizes[i])

        plt.xlabel('Time [h]', fontsize=label_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SIR [dB]', fontsize=label_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=ticks_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=ticks_size,
                   fontname=self.font_line_sizes['fontname'])

        # Adjustemnt of bbox to better fit
        box = fig.get_position()
        fig.set_position([box.x0 * 1.05, box.y0 * 1.34, box.width * 1.11, box.height * 1.1])

        plt.legend(ncol=2, columnspacing=0, framealpha=0.7, fontsize=legend_size, loc='lower right')
        plt.ylim(-4, 6.5)
        # # Time windows to expand for the pdf
        # minutes = [7,
        #            30, 31 + 25,
        #            60 + 17, 60 + 18 + 25,
        #            120 + 5, 120 + 6 + 25,
        #            120 + 51]
        # plt.vlines([minutes[0] * 60,
        #             minutes[1] * 60, minutes[2] * 60,
        #             minutes[3] * 60, minutes[4] * 60,
        #             minutes[5] * 60, minutes[6] * 60,
        #             minutes[7] * 60], -5, 7, colors='k')
        # plt.vlines(time_math, -5, 7, colors='k', alpha=0.01)

        extension = 'journal_shifted_8_sir_vs_time' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def journal_shifted_pdf(self, file_ext='.jpg', n_sats_list=None, show=True, alpha_list=30, inclination=3):
        alpha_list = [alpha_list] if not isinstance(alpha_list, list) else alpha_list
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        pairs = list(product(alpha_list, n_sats_list))
        results_df = self.math_results_df
        h = self.orbital_input_params['h_lims'][0]

        _, _, colors = tk.templateColors(iterable_size=len(pairs))

        fig = get_ax()
        plt.grid()

        for i, (alpha, n_sats) in enumerate(pairs):
            x_values = results_df.loc[(results_df['n_sats'] == n_sats) &
                                      (results_df['h'] == h) &
                                      (results_df['inclination'] == inclination) &
                                      (results_df['alpha'] == alpha), 'interference']
            if not x_values.isnull().all():
                ###### Curve joining all histogram bar tops
                # create a histogram
                hist, edges = np.histogram(x_values, bins=100, density=True)

                # find the x and y coordinates of the tips of the histogram bars
                x = edges[:-1] + np.diff(edges) / 2
                y = hist

                # plot the original histogram and the smooth curve
                plt.plot(x, y, color=colors[i], label=r'$\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                         linewidth=self.font_line_sizes['linewidth'])

        extension = 'journal_shifted_9_interference_pdf' + file_ext
        plt.xlabel('E[I] [dBm]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Probability', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])

        plt.legend()
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def journal_shifted_sinr_vs_beamwidth(self, file_ext='.jpg', n_sats_list=None, show=True, inclination=3):
        thz_n_sats_list = self.thz_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        mmwave_n_sats_list = self.mmWave_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        alpha_list = np.arange(self.orbital_input_params['alpha_lims'][0],
                               self.orbital_input_params['alpha_lims'][1],
                               self.orbital_input_params['alpha_resolution_math'])
        h = self.orbital_input_params['h_lims'][0]

        # Results dataframes
        thz_math_results_df = self.thz_math_results_df
        mmwave_math_results_df = self.mmWave_math_results_df

        # Time averaging of results
        thz_math_results_df = thz_math_results_df.groupby(['n_sats', 'alpha', 'h', 'inclination', 'beta'],
                                                          as_index=False).mean()
        adjust_db_columns(thz_math_results_df)
        mmwave_math_results_df = mmwave_math_results_df.groupby(['n_sats', 'alpha', 'h', 'inclination', 'beta'],
                                                                as_index=False).mean()
        adjust_db_columns(mmwave_math_results_df)

        _, _, colors = tk.templateColors(iterable_size=2, id=2)
        linestyles = ['solid', 'dashed', 'dotted']

        figsize = (10, 8)
        ticks_size = 35
        label_size = 45
        legend_size = 28

        ax = get_ax(figsize=figsize)
        plt.grid()

        # Theoretical maximum (SNR) as a dashed line, from the math model
        for i, n_sats in enumerate([max(mmwave_n_sats_list)]):
            snr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['n_sats'] == n_sats) &
                                                         (mmwave_math_results_df['h'] == h) &
                                                         (mmwave_math_results_df['inclination'] == inclination),
                                                         'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.plot(alpha_list, y_values, color='dimgray',
                     label=None,
                     linewidth=self.font_line_sizes['linewidth'], linestyle='dashed')

        # THz
        for i, n_sats in enumerate(np.flip(thz_n_sats_list)):
            sinr_linear = list(thz_math_results_df.loc[(thz_math_results_df['n_sats'] == n_sats) &
                                                       (thz_math_results_df['h'] == h) &
                                                       (thz_math_results_df['inclination'] == inclination),
                                                       'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(alpha_list, y_values, color=colors[1],
                     label=r'THz: N={}'.format(n_sats),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i], marker=markers[i],
                     markersize=sizesplot[i], markeredgecolor='k', markevery=5)

        # mmWave
        for i, n_sats in enumerate(np.flip(mmwave_n_sats_list)):
            sinr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['n_sats'] == n_sats) &
                                                          (mmwave_math_results_df['h'] == h) &
                                                          (mmwave_math_results_df['inclination'] == inclination),
                                                          'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(alpha_list, y_values, color=colors[0],
                     label=r'mmWave: N={}'.format(n_sats),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i], marker=markers[i],
                     markersize=sizesplot[i], markeredgecolor='k', markevery=5)

        # plt.suptitle('Coplanar orbits: SINR', fontsize=self.font_line_sizes['titleSize'],
        #              fontname=self.font_line_sizes['fontname'])
        plt.xlabel('Beamwidth [$^\circ$]', fontsize=label_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SINR [dB]', fontsize=label_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks([1, 5, 10, 15, 20, 25, 30],
                   fontsize=ticks_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks([-80, -60, -40, -20, 0, 20, 40, 60], fontsize=ticks_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.ylim(-100, 70)
        plt.xlim(0.5, 30)

        # Version 2: Legend below
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.9])
        # plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='upper center',
        #            bbox_to_anchor=(0.5, -0.12), ncol=3, fancybox=True)

        # Adjustemnt of bbox to better fit
        box = ax.get_position()
        ax.set_position([box.x0 * 1.19, box.y0 * 1.36, box.width * 1.05, box.height * 1.1])
        plt.legend(ncol=2, columnspacing=0.4, handletextpad=0.2, framealpha=0.7, fontsize=legend_size, loc='lower right')

        extension = 'journal_shifted_9_sinr_vs_beamwidth' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def journal_shifted_capacity_vs_inclination(self, file_ext='.jpg', thz_alpha_list=None,
                                                mmwave_alpha_list=None, n_sats_list=None, show=True):
        n_sats_list = self.orbital_input_params['n_sats_list'] if n_sats_list is None else n_sats_list
        thz_alpha_list = self.thz_input_params['alpha_list'] if thz_alpha_list is None else thz_alpha_list
        mmwave_alpha_list = self.mmWave_input_params['alpha_list'] if mmwave_alpha_list is None else mmwave_alpha_list
        h = self.orbital_input_params['h_lims'][0]

        # Results dataframes
        thz_math_results_df = self.thz_math_results_df
        mmwave_math_results_df = self.mmWave_math_results_df

        # Time averaging of results
        thz_math_results_df = thz_math_results_df.groupby(['n_sats', 'alpha', 'h', 'inclination', 'beta'],
                                                          as_index=False).mean()
        adjust_db_columns(thz_math_results_df)
        mmwave_math_results_df = mmwave_math_results_df.groupby(['n_sats', 'alpha', 'h', 'inclination', 'beta'],
                                                                as_index=False).mean()
        adjust_db_columns(mmwave_math_results_df)

        # Horizontal axis list
        inclination_list = np.arange(self.orbital_input_params['inclination_lims'][0],
                                     self.orbital_input_params['inclination_lims'][1],
                                     self.orbital_input_params['inclination_resolution_math'])

        mmwave_colors = ['#e41a1c', '#ff7f00']  # Red, Orange
        thz_colors = ['#377eb8', '#984ea3']  # Blue, Purple
        linestyles = ['solid', 'dashed', 'dotted']

        figsize = (10, 8)
        ticks_size = 35
        label_size = 45
        legend_size = 28

        ax = get_ax(figsize=figsize)
        plt.grid()

        # mmWave
        for i, alpha in enumerate(mmwave_alpha_list):
            for j, n_sats in enumerate(n_sats_list):
                y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['n_sats'] == n_sats) &
                                                           (mmwave_math_results_df['h'] == h) &
                                                           (mmwave_math_results_df['alpha'] == alpha),
                                                           'capacity_gbps'])
                plt.plot(inclination_list, y_values, color=mmwave_colors[i],
                         label=r'mmWave: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                         linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[j])
        # THz
        for i, alpha in enumerate(thz_alpha_list):
            for j, n_sats in enumerate(n_sats_list):
                y_values = list(thz_math_results_df.loc[(thz_math_results_df['n_sats'] == n_sats) &
                                                        (thz_math_results_df['h'] == h) &
                                                        (thz_math_results_df['alpha'] == alpha),
                                                        'capacity_gbps'])
                plt.plot(inclination_list, y_values, color=thz_colors[i],
                         label=r'THz: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                         linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[j])


        # plt.suptitle('Coplanar orbits: SINR', fontsize=self.font_line_sizes['titleSize'],
        #              fontname=self.font_line_sizes['fontname'])
        plt.yscale('log')
        plt.xlabel('Inclination [$^\circ$]', fontsize=label_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Capacity [Gbps]', fontsize=label_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=ticks_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=ticks_size,
                   fontname=self.font_line_sizes['fontname'])

        # # Version 2: Legend below
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.9])
        # plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='upper center',
        #            bbox_to_anchor=(0.5, -0.12), ncol=2, fancybox=True)

        # Adjustemnt of bbox to better fit
        box = ax.get_position()
        ax.set_position([box.x0 * 1.19, box.y0 * 1.36, box.width * 1.05, box.height * 1.1])
        plt.legend(ncol=2, columnspacing=0.4, handletextpad=0.2, framealpha=0.7, fontsize=legend_size,
                   loc='lower right')

        extension = 'journal_coplanar_11_capcity_vs_inclination' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    def journal_shifted_capacity_vs_number_of_satellites(self, file_ext='.jpg', thz_alpha_list=None,
                                                         mmwave_alpha_list=None, show=True):
        thz_alpha_list = self.thz_input_params['alpha_list'] if thz_alpha_list is None else thz_alpha_list
        mmwave_alpha_list = self.mmWave_input_params['alpha_list'] if mmwave_alpha_list is None else mmwave_alpha_list
        thz_inc_list = self.thz_input_params['inclination_list']
        mmwave_inc_list = self.mmWave_input_params['inclination_list']
        h = self.orbital_input_params['h_lims'][0]

        # Results dataframes
        thz_math_results_df = self.thz_math_results_df
        mmwave_math_results_df = self.mmWave_math_results_df

        # Time averaging of results
        thz_math_results_df = thz_math_results_df.groupby(['n_sats', 'alpha', 'h', 'inclination', 'beta'],
                                                          as_index=False).mean()
        adjust_db_columns(thz_math_results_df)
        mmwave_math_results_df = mmwave_math_results_df.groupby(['n_sats', 'alpha', 'h', 'inclination', 'beta'],
                                                                as_index=False).mean()
        adjust_db_columns(mmwave_math_results_df)

        # Horizontal axis list
        n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                self.orbital_input_params['n_sats_lims'][1],
                                self.orbital_input_params['n_sats_resolution_math'])

        mmwave_colors = ['#e41a1c', '#ff7f00']  # Red, Orange
        thz_colors = ['#377eb8', '#984ea3']  # Blue, Purple
        linestyles = ['solid', 'dashdot', 'dotted']

        figsize = (10, 8)
        ticks_size = 35
        label_size = 45
        legend_size = 28

        ax = get_ax(figsize=figsize)
        plt.grid()


        # THz
        for i, alpha in enumerate(thz_alpha_list):
            for j, inclination in enumerate([min(thz_inc_list)]):
                y_values = list(thz_math_results_df.loc[(thz_math_results_df['inclination'] == inclination) &
                                                        (thz_math_results_df['h'] == h) &
                                                        (thz_math_results_df['alpha'] == alpha),
                                                        'capacity_gbps'])
                plt.plot(n_sats_list, y_values, color=thz_colors[i],
                         label=r'Sub-THz: $\alpha$={}$^\circ$, $\gamma$={}$^\circ$'.format(alpha, inclination),
                         linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[j], marker=markers[i],
                         markersize=sizesplot[i], markevery=15, markeredgecolor='k')

        # mmWave
        for i, alpha in enumerate(mmwave_alpha_list):
            for j, inclination in enumerate(mmwave_inc_list):
                y_values = list(
                    mmwave_math_results_df.loc[(mmwave_math_results_df['inclination'] == inclination) &
                                               (mmwave_math_results_df['h'] == h) &
                                               (mmwave_math_results_df['alpha'] == alpha),
                                               'capacity_gbps'])
                plt.plot(n_sats_list, y_values, color=mmwave_colors[i],
                         label=r'mmWave: $\alpha$={}$^\circ$, $\gamma$={}$^\circ$'.format(alpha, inclination),
                         linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[j],
                         marker=markers[j],
                         markersize=sizesplot[j], markevery=15, markeredgecolor='k')


        # plt.suptitle('Coplanar orbits: SINR', fontsize=self.font_line_sizes['titleSize'],
        #              fontname=self.font_line_sizes['fontname'])
        plt.yscale('log')
        plt.xlabel('Satellites per orbit', fontsize=label_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Capacity [Gbps]', fontsize=label_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks([10, 25, 50, 75, 100, 125, 150, 175, 200],
                   fontsize=ticks_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks([0.01, 0.1, 1, 10, 100],
                   [0.01, 0.1, 1, 10, 100],
                   fontsize=ticks_size,
                   fontname=self.font_line_sizes['fontname'])
        plt.xlim(8, 200)
        plt.ylim(0.001, 100)

        # # Version 2: Legend below
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.9])
        # plt.legend(framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='upper center',
        #            bbox_to_anchor=(0.5, -0.12), ncol=2, fancybox=True)

        # Adjustemnt of bbox to better fit
        box = ax.get_position()
        ax.set_position([box.x0 * 1.27, box.y0 * 1.36, box.width * 1.02, box.height * 1.07])
        plt.legend(ncol=1, columnspacing=0.4, handletextpad=0.2, framealpha=0.7, fontsize=legend_size,
                   loc='lower right', borderaxespad=0.2, labelspacing=0.3)

        extension = 'journal_shifted_10_capacity_vs_n_sats' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
        if show:
            plt.show()
        plt.close()

    # Complete constellation plots
    def complete_constellation_sir_snr_sinr_capacity_vs_time(self, file_ext='.jpg', n_sats_list=10, thz_alpha_list=None,
                                                             mmwave_alpha_list=None, show=True):
        thz_alpha_list = self.thz_input_params['alpha_list'] if thz_alpha_list is None else thz_alpha_list
        mmwave_alpha_list = self.mmWave_input_params['alpha_list'] if mmwave_alpha_list is None else mmwave_alpha_list
        n_sats_list = [n_sats_list] if not isinstance(n_sats_list, list) else n_sats_list

        # Results dataframes
        thz_math_results_df = self.thz_math_results_df
        mmwave_math_results_df = self.mmWave_math_results_df

        # Time vectors from results dataframes
        t_start = datetime(2022, 1, 1)
        time = thz_math_results_df.loc[(thz_math_results_df['n_sats'] == n_sats_list[0]) &
                                       (thz_math_results_df['alpha'] == thz_alpha_list[0]), 'time']
        time = time.to_list()
        time = [time[i] - t_start for i in range(len(time))]
        time = [time[i].total_seconds() for i in range(len(time))]

        mmwave_colors = ['#e31a1c', '#e41bb2']  # Dark red, Light red
        thz_colors = ['#3b9dde', '#1e1eb3']  # Dark blue, Light blue
        linestyles = ['solid', 'dashdot', 'dotted']

        # Formatting stuff for x axis
        formatter = FuncFormatter(format_func)
        fig = get_ax(figsize=(10 * 2, 10))

        # SIR
        ax1 = plt.subplot(221)
        plt.grid()
        ax1.xaxis.set_major_formatter(formatter)
        # mmWave
        for i, alpha in enumerate(mmwave_alpha_list):
            for j, n_sats in enumerate(n_sats_list):
                sir_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['n_sats'] == n_sats) &
                                                             (mmwave_math_results_df['alpha'] == alpha), 'sir_linear'])
                y_values = [10 * np.log10(value) if value else None for value in sir_linear]
                plt.plot(time, y_values, color=mmwave_colors[j],
                         label=r'mmWave: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                         linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        # THz
        for i, alpha in enumerate(thz_alpha_list):
            for j, n_sats in enumerate(n_sats_list):
                sir_linear = list(thz_math_results_df.loc[(thz_math_results_df['n_sats'] == n_sats) &
                                                          (thz_math_results_df['alpha'] == alpha), 'sir_linear'])
                y_values = [10 * np.log10(value) if value else None for value in sir_linear]
                plt.plot(time, y_values, color=thz_colors[j],
                         label=r'mmWave: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                         linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        # plt.xlim(0, 500)
        # plt.ylim(-25, 14)

        # SNR
        ax2 = plt.subplot(222)
        plt.grid()
        ax2.xaxis.set_major_formatter(formatter)
        # mmWave
        for i, alpha in enumerate(mmwave_alpha_list):
            for j, n_sats in enumerate(n_sats_list):
                snr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['n_sats'] == n_sats) &
                                                             (mmwave_math_results_df['alpha'] == alpha), 'snr_linear'])
                y_values = [10 * np.log10(value) if value else None for value in snr_linear]
                plt.plot(time, y_values, color=mmwave_colors[j],
                         label=r'mmWave: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                         linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        # THz
        for i, alpha in enumerate(thz_alpha_list):
            for j, n_sats in enumerate(n_sats_list):
                snr_linear = list(thz_math_results_df.loc[(thz_math_results_df['n_sats'] == n_sats) &
                                                          (thz_math_results_df['alpha'] == alpha), 'snr_linear'])
                y_values = [10 * np.log10(value) if value else None for value in snr_linear]
                plt.plot(time, y_values, color=thz_colors[j],
                         label=r'mmWave: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                         linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SNR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        # plt.xlim(0, 500)
        # plt.ylim(-25, 14)

        # SINR
        ax3 = plt.subplot(223)
        plt.grid()
        ax3.xaxis.set_major_formatter(formatter)
        # mmWave
        for i, alpha in enumerate(mmwave_alpha_list):
            for j, n_sats in enumerate(n_sats_list):
                sinr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['n_sats'] == n_sats) &
                                                              (mmwave_math_results_df[
                                                                   'alpha'] == alpha), 'sinr_linear'])
                y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
                plt.plot(time, y_values, color=mmwave_colors[j],
                         label=r'mmWave: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                         linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        # THz
        for i, alpha in enumerate(thz_alpha_list):
            for j, n_sats in enumerate(n_sats_list):
                sinr_linear = list(thz_math_results_df.loc[(thz_math_results_df['n_sats'] == n_sats) &
                                                           (thz_math_results_df['alpha'] == alpha), 'sinr_linear'])
                y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
                plt.plot(time, y_values, color=thz_colors[j],
                         label=r'mmWave: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                         linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SINR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        # plt.xlim(0, 500)
        # plt.ylim(-25, 14)

        # Channel capacity
        ax4 = plt.subplot(224)
        plt.grid()
        ax4.xaxis.set_major_formatter(formatter)
        # mmWave
        for i, alpha in enumerate(mmwave_alpha_list):
            for j, n_sats in enumerate(n_sats_list):
                y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['n_sats'] == n_sats) &
                                                           (mmwave_math_results_df['alpha'] == alpha), 'capacity_gbps'])
                plt.plot(time, y_values, color=mmwave_colors[j],
                         label=r'mmWave: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                         linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        # THz
        for i, alpha in enumerate(thz_alpha_list):
            for j, n_sats in enumerate(n_sats_list):
                y_values = list(thz_math_results_df.loc[(thz_math_results_df['n_sats'] == n_sats) &
                                                        (thz_math_results_df['alpha'] == alpha), 'capacity_gbps'])
                plt.plot(time, y_values, color=thz_colors[j],
                         label=r'mmWave: $\alpha$={}$^\circ$, N={}'.format(alpha, n_sats),
                         linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        plt.xlabel('Time [h]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Channel Capacity [Gbps]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yscale('log')
        plt.legend(ncol=2, framealpha=0.7, fontsize=self.font_line_sizes['legendSize'])
        # plt.xlim(0, 500)
        # plt.ylim(-25, 14)

        extension = 'complete_constellation_sir_snr_sinr_capacity_vs_time_' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def complete_constellation_sir_snr_sinr_capacity_n_sats(self, file_ext='.jpg', thz_alpha_list=None,
                                                            mmwave_alpha_list=None, show=True):
        thz_alpha_list = self.thz_input_params['alpha_list'] if thz_alpha_list is None else thz_alpha_list
        mmwave_alpha_list = self.mmWave_input_params['alpha_list'] if mmwave_alpha_list is None else mmwave_alpha_list
        n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                self.orbital_input_params['n_sats_lims'][1],
                                self.orbital_input_params['n_sats_resolution'])

        # Results dataframes
        thz_math_results_df = self.thz_math_results_df
        mmwave_math_results_df = self.mmWave_math_results_df

        # Time averaging of results
        thz_math_results_df = thz_math_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(thz_math_results_df)
        mmwave_math_results_df = mmwave_math_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(mmwave_math_results_df)

        mmwave_colors = ['#e31a1c', '#e41bb2']  # Dark red, Light red
        thz_colors = ['#3b9dde', '#1e1eb3']  # Dark blue, Light blue
        linestyles = ['solid', 'dashdot', 'dotted']

        fig = get_ax(figsize=(10 * 2, 10))

        # SIR
        ax1 = plt.subplot(221)
        plt.grid()
        # mmWave
        for i, alpha in enumerate(mmwave_alpha_list):
            sir_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(n_sats_list, y_values, color=mmwave_colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        # THz
        for i, alpha in enumerate(thz_alpha_list):
            sir_linear = list(thz_math_results_df.loc[(thz_math_results_df['alpha'] == alpha), 'sir_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sir_linear]
            plt.plot(n_sats_list, y_values, color=thz_colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        plt.xlabel('Satellites per orbit', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        # plt.xlim(0, 500)
        # plt.ylim(-25, 14)

        # SNR
        ax2 = plt.subplot(222)
        plt.grid()
        # mmWave
        for i, alpha in enumerate(mmwave_alpha_list):
            snr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha), 'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.plot(n_sats_list, y_values, color=mmwave_colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        # THz
        for i, alpha in enumerate(thz_alpha_list):
            snr_linear = list(thz_math_results_df.loc[(thz_math_results_df['alpha'] == alpha), 'snr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in snr_linear]
            plt.plot(n_sats_list, y_values, color=thz_colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        plt.xlabel('Satellites per orbit', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SNR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        # plt.xlim(0, 500)
        # plt.ylim(-25, 14)

        # SINR
        ax3 = plt.subplot(223)
        plt.grid()
        # mmWave
        for i, alpha in enumerate(mmwave_alpha_list):
            sinr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(n_sats_list, y_values, color=mmwave_colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        # THz
        for i, alpha in enumerate(thz_alpha_list):
            sinr_linear = list(thz_math_results_df.loc[(thz_math_results_df['alpha'] == alpha), 'sinr_linear'])
            y_values = [10 * np.log10(value) if value else None for value in sinr_linear]
            plt.plot(n_sats_list, y_values, color=thz_colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        plt.xlabel('Satellites per orbit', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('SINR [dB]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        # plt.xlim(0, 500)
        # plt.ylim(-25, 14)

        # Channel capacity
        ax4 = plt.subplot(224)
        plt.grid()
        # mmWave
        for i, alpha in enumerate(mmwave_alpha_list):
            y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha), 'capacity_gbps'])
            plt.plot(n_sats_list, y_values, color=mmwave_colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        # THz
        for i, alpha in enumerate(thz_alpha_list):
            y_values = list(thz_math_results_df.loc[(thz_math_results_df['alpha'] == alpha), 'capacity_gbps'])
            plt.plot(n_sats_list, y_values, color=thz_colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[i])

        plt.xlabel('Satellites per orbit', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Channel Capacity [Gbps]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yscale('log')
        plt.legend(ncol=2, framealpha=0.7, fontsize=self.font_line_sizes['legendSize'],
                   loc='upper right')
        # plt.xlim(0, 500)
        # plt.ylim(-25, 14)

        extension = 'complete_constellation_sir_snr_sinr_capacity_vs_n_sats_' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def complete_constellation_capacity_vs_n_sats(self, file_ext='.jpg', thz_alpha_list=None,
                                                  mmwave_alpha_list=None, show=True):
        thz_alpha_list = self.thz_input_params['alpha_list'] if thz_alpha_list is None else thz_alpha_list
        mmwave_alpha_list = self.mmWave_input_params['alpha_list'] if mmwave_alpha_list is None else mmwave_alpha_list
        n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                self.orbital_input_params['n_sats_lims'][1],
                                self.orbital_input_params['n_sats_resolution'])
        bw_mmwave = self.mmWave_input_params['bandwidth']
        bw_thz = self.thz_input_params['bandwidth']
        inclination_list = self.orbital_input_params['inclination']

        # Results dataframes
        thz_math_results_df = self.thz_math_results_df
        mmwave_math_results_df = self.mmWave_math_results_df

        # Time averaging of results
        thz_math_results_df = thz_math_results_df.groupby(['n_sats', 'alpha', 'inclination'], as_index=False).mean()
        adjust_db_columns(thz_math_results_df)
        mmwave_math_results_df = mmwave_math_results_df.groupby(['n_sats', 'alpha', 'inclination'], as_index=False).mean()
        adjust_db_columns(mmwave_math_results_df)

        mmwave_colors = ['#e31a1c', '#e41bb2']  # Dark red, Light red
        thz_colors = ['#3b9dde', '#1e1eb3']  # Dark blue, Light blue
        color_dict, _, colors_template = tk.templateColors(iterable_size=7, reverse=False)
        linestyles = ['solid', 'dashdot', 'dotted']

        figsize=(10,6)
        fig = get_ax(figsize=figsize)

        # Capacity with interference
        plt.grid()
        thz_inclination = inclination_list[-1]
        # THz
        # SNR capacity
        alpha = thz_alpha_list[0]
        blue = thz_colors[1]
        yellow = '#FCCB1A'
        snr_linear = list(thz_math_results_df.loc[(thz_math_results_df['alpha'] == alpha)
                                                  & (thz_math_results_df['inclination'] == thz_inclination),
                                                  'snr_linear'])
        y_values = [bw_thz * np.log2(1 + value) * 1e-9 if value != 0 else None for value in snr_linear]
        plt.plot(n_sats_list, y_values, color=blue,
                 label=None, linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[2])

        # Intermediate capacities THz
        # SNR + single
        y_values = list(thz_math_results_df.loc[(thz_math_results_df['alpha'] == alpha)
                                                & (thz_math_results_df['inclination'] == thz_inclination),
                                                'capacity_single'])
        plt.plot(n_sats_list, y_values,
                 label=r'Thz: $\alpha$={}$^\circ$, $\gamma$={}$^\circ$, '.format(alpha, thz_inclination) + 'single orbit',
                 linewidth=self.font_line_sizes['linewidth'], linestyle=(0, (5, 5)), color=blue, zorder=5)

        # SNR + single + shifted + coplanar
        y_values = list(thz_math_results_df.loc[(thz_math_results_df['alpha'] == alpha)
                                                & (thz_math_results_df[
                                                       'inclination'] == thz_inclination), 'capacity_gbps'])
        plt.plot(n_sats_list, y_values,
                 label=r'Thz: $\alpha$={}$^\circ$, $\gamma$={}$^\circ$'.format(alpha,
                                                                                 thz_inclination),
                 linewidth=self.font_line_sizes['linewidth'], linestyle='solid', color=yellow)

        # mmWave
        # SNR
        alpha = mmwave_alpha_list[0]
        inclination = inclination_list[1]
        red = '#FE2712'
        green = '#66B032'
        snr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha)
                                                     & (mmwave_math_results_df['inclination'] == inclination),
                                                     'snr_linear'])
        y_values = [bw_mmwave * np.log2(1 + value) * 1e-9 if value != 0 else None for value in snr_linear]
        plt.plot(n_sats_list, y_values, color=red,
                 label=None, linewidth=self.font_line_sizes['linewidth'], linestyle='dotted')

        # SNR + Single
        alpha = mmwave_alpha_list[0]
        linestyles = [(0, (5, 5)), 'solid']
        colors = [red, green]
        y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha)
                                                   & (mmwave_math_results_df['inclination'] == inclination),
                                                   'capacity_single'])
        plt.plot(n_sats_list, y_values, label=r'mmWave $\alpha$={}$^\circ$, $\gamma$={}$^\circ$, '.format(alpha,
                                                                                                          inclination) +
                                              'single orbit',
                 linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[0], color=colors[0], zorder=5)

        # SNR + single + shifted
        y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha)
                                                   & (mmwave_math_results_df['inclination'] == inclination),
                                                   'capacity_shifted'])
        plt.plot(n_sats_list, y_values,
                 label=r'mmWave $\alpha$={}$^\circ$, $\gamma$={}$^\circ$, '.format(alpha, inclination) + 'shifted orbits',
                 linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[1], color=colors[1])

        # SNR + single + shifted + low inclination
        magenta = '#AE0D7A'
        inclination = inclination_list[0]
        y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha)
                                                   & (mmwave_math_results_df['inclination'] == inclination),
                                                   'capacity_shifted'])
        plt.plot(n_sats_list, y_values,
                 label=r'mmWave $\alpha$={}$^\circ$, $\gamma$={}$^\circ$, '.format(alpha, inclination) + 'shifted orbits',
                 linewidth=self.font_line_sizes['linewidth'], linestyle='dashdot', color=magenta)

        # SNR + single + shifted + ONE coplanar
        inclination = inclination_list[1]
        y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha)
                                                   & (mmwave_math_results_df['inclination'] == inclination),
                                                   'capacity_shifted_1_coplanar'])
        plt.plot(n_sats_list, y_values,
                 label=r'mmWave $\alpha$={}$^\circ$, $\gamma$={}$^\circ$, 1 coplanar'.format(alpha, inclination),
                 linewidth=self.font_line_sizes['linewidth'], linestyle=(0, (5, 5)), color='orange', zorder=5)

        # SNR + single + shifted + coplanar
        color = color_dict['Blue']
        inclination = inclination_list[-1]
        y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha)
                                                   & (mmwave_math_results_df['inclination'] == inclination),
                                                   'capacity_gbps'])
        plt.plot(n_sats_list, y_values,
                 label=r'mmWave $\alpha$={}$^\circ$, $\gamma$={}$^\circ$'.format(alpha, inclination),
                 linewidth=self.font_line_sizes['linewidth'], linestyle='solid', color=thz_colors[0])

        # # SNR + single + shifted + coplanar low inclination
        # color = color_dict['Orange']
        # inclination = inclination_list[0]
        # y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha)
        #                                            & (mmwave_math_results_df['inclination'] == inclination),
        #                                            'capacity_gbps'])
        # plt.plot(n_sats_list, y_values,
        #          label=r'mmWave $\alpha$={}$^\circ$, $\gamma$={}$^\circ$'.format(alpha, inclination),
        #          linewidth=self.font_line_sizes['linewidth'], linestyle='solid', color=color)


        # # SNR + single + shifted + ONE coplanar low inclination
        # inclination = inclination_list[0]
        # y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha)
        #                                            & (mmwave_math_results_df['inclination'] == inclination),
        #                                            'capacity_shifted_1_coplanar'])
        # plt.plot(n_sats_list, y_values,
        #          label=r'mmWave $\alpha$={}$^\circ$, $\gamma$={}$^\circ$, 1 coplanar'.format(alpha, inclination),
        #          linewidth=self.font_line_sizes['linewidth'], linestyle='dotted', color='k')
        plt.yscale('log')
        plt.xlabel('Satellites per orbit', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Capacity [Gbps]', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks([10, 100, 200, 300, 400, 500],
                   fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks([0.1, 1, 10, 100],
                   [0.1, 1, 10, 100],
                   fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])

        plt.xlim(0, 500)
        plt.ylim(0.01, 100)

        # Adjustemnt of bbox to better fit figures 5b and 5c which have the legent outside of the bbox
        box = fig.get_position()
        fig.set_position([box.x0 * 0.9, box.y0, box.width * 1.1, box.height * 1.1])
        plt.legend(ncol=2, framealpha=1, fontsize=self.font_line_sizes['legendSize'], loc='lower left', columnspacing=0.2)

        extension = 'complete_constellation_capacity_vs_n_sats_' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def complete_constellation_capacity_vs_n_sats_v2(self, file_ext='.jpg', thz_alpha_list=None,
                                                  mmwave_alpha_list=None, show=True):
        thz_alpha_list = self.thz_input_params['alpha_list'] if thz_alpha_list is None else thz_alpha_list
        mmwave_alpha_list = self.mmWave_input_params['alpha_list'] if mmwave_alpha_list is None else mmwave_alpha_list
        n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                self.orbital_input_params['n_sats_lims'][1],
                                self.orbital_input_params['n_sats_resolution'])
        bw_mmwave = self.mmWave_input_params['bandwidth']
        bw_thz = self.thz_input_params['bandwidth']
        inclination_list = self.orbital_input_params['inclination']

        # Results dataframes
        thz_math_results_df = self.thz_math_results_df
        mmwave_math_results_df = self.mmWave_math_results_df

        # Time averaging of results
        thz_math_results_df = thz_math_results_df.groupby(['n_sats', 'alpha', 'inclination'], as_index=False).mean()
        adjust_db_columns(thz_math_results_df)
        mmwave_math_results_df = mmwave_math_results_df.groupby(['n_sats', 'alpha', 'inclination'], as_index=False).mean()
        adjust_db_columns(mmwave_math_results_df)

        color_dict, _, colors_template = tk.templateColors(iterable_size=7, reverse=False)
        linestyles = ['solid', 'dashdot', 'dotted']

        figsize = (10, 8)
        fig = get_ax(figsize=figsize)

        # Capacity with interference
        plt.grid()
        thz_inclination = inclination_list[-1]
        # THz
        # SNR capacity
        alpha = thz_alpha_list[0]
        alpha_plots=0.3
        line_width_mult=4
        snr_linear = list(thz_math_results_df.loc[(thz_math_results_df['alpha'] == alpha)
                                                  & (thz_math_results_df['inclination'] == thz_inclination),
                                                  'snr_linear'])
        y_values = [bw_thz * np.log2(1 + value) * 1e-9 if value != 0 else None for value in snr_linear]
        plt.plot(n_sats_list, y_values, color=colors_template[1], label=None,
                 linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[2])

        # Intermediate capacities THz
        # SNR + single
        y_values = list(thz_math_results_df.loc[(thz_math_results_df['alpha'] == alpha)
                                                & (thz_math_results_df['inclination'] == thz_inclination),
                                                'capacity_single'])
        # plt.plot(n_sats_list, y_values,
        #          label=r'$\alpha$={}$^\circ$, $\gamma$={}$^\circ$,'.format(alpha, thz_inclination) + '\nsingle orbit',
        #          linewidth=self.font_line_sizes['linewidth'], linestyle=(0, (5, 5)), color=blue, zorder=5)

        plt.plot(n_sats_list, y_values,
                 label=r'$\alpha$={}$^\circ$, $\gamma$={}$^\circ$,'.format(alpha, thz_inclination) + ' 1 orbit',
                 linewidth=self.font_line_sizes['linewidth'] * line_width_mult, alpha=alpha_plots, linestyle='solid',
                 color=colors_template[1], zorder=5,
                 marker='', markersize=sizesplot[0], markeredgecolor='k', markevery=2)

        # all orbits
        y_values = list(thz_math_results_df.loc[(thz_math_results_df['alpha'] == alpha)
                                                & (thz_math_results_df['inclination'] == thz_inclination),
                                                'capacity_gbps'])
        plt.plot(n_sats_list, y_values,
                 label=r'$\alpha$={}$^\circ$, $\gamma$={}$^\circ$'.format(alpha, thz_inclination) + ', all orbits',
                 linewidth=self.font_line_sizes['linewidth'], linestyle='solid', color=colors_template[1],
                 marker='', markersize=sizesplot[1], markeredgecolor='k', markevery=(1, 2))

        # mmWave
        # SNR
        alpha = mmwave_alpha_list[0]
        inclination = inclination_list[1]
        red = '#FE2712'
        green = '#66B032'
        snr_linear = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha)
                                                     & (mmwave_math_results_df['inclination'] == inclination),
                                                     'snr_linear'])
        y_values = [bw_mmwave * np.log2(1 + value) * 1e-9 if value != 0 else None for value in snr_linear]
        plt.plot(n_sats_list, y_values, color=colors_template[0],
                 label=None, linewidth=self.font_line_sizes['linewidth'], linestyle='dotted')

        # SNR + Single
        alpha = mmwave_alpha_list[0]
        linestyles = [(0, (5, 5)), 'solid']
        colors = [red, green]
        y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha)
                                                   & (mmwave_math_results_df['inclination'] == inclination),
                                                   'capacity_single'])
        # plt.plot(n_sats_list, y_values, label=r'mmWave $\alpha$={}$^\circ$, $\gamma$={}$^\circ$,'.format(alpha,
        #                                                                                                   inclination) +
        #                                       '\nsingle orbit',
        #          linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[0], color=colors[0], zorder=5)

        plt.plot(n_sats_list, y_values, label=r'$\alpha$={}$^\circ$, $\gamma$={}$^\circ$, '.format(alpha,inclination) +
                                              '1 orbit',
                 linewidth=self.font_line_sizes['linewidth'] * line_width_mult, linestyle='solid', color=colors_template[0],
                 alpha=alpha_plots, zorder=5)

        # SNR + single + shifted
        y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha)
                                                   & (mmwave_math_results_df['inclination'] == inclination),
                                                   'capacity_shifted'])
        # plt.plot(n_sats_list, y_values,
        #          label=r'$\alpha$={}$^\circ$, $\gamma$={}$^\circ$,'.format(alpha, inclination) + '\nshifted orbits',
        #          linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[1], color=colors[1])

        plt.plot(n_sats_list, y_values,
                 label=r'$\alpha$={}$^\circ$, $\gamma$={}$^\circ$'.format(alpha, inclination) + ', 10 orbits',
                 linewidth=self.font_line_sizes['linewidth'], linestyle=linestyles[1], color=colors_template[0])

        # SNR + single + shifted + low inclination
        magenta = '#AE0D7A'
        inclination = inclination_list[0]
        y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha)
                                                   & (mmwave_math_results_df['inclination'] == inclination),
                                                   'capacity_shifted'])
        # plt.plot(n_sats_list, y_values,
        #          label=r'$\alpha$={}$^\circ$, $\gamma$={}$^\circ$,'.format(alpha, inclination) + '\nshifted orbits',
        #          linewidth=self.font_line_sizes['linewidth'], linestyle='dashdot', color=magenta)

        plt.plot(n_sats_list, y_values,
                 label=r'$\alpha$={}$^\circ$, $\gamma$={}$^\circ$'.format(alpha, inclination) + ', 10 orbits',
                 linewidth=self.font_line_sizes['linewidth'], linestyle='dashdot', color=colors_template[3], zorder=5)

        # SNR + single + shifted + ONE coplanar
        inclination = inclination_list[1]
        y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha)
                                                   & (mmwave_math_results_df['inclination'] == inclination),
                                                   'capacity_shifted_1_coplanar'])
        # plt.plot(n_sats_list, y_values,
        #          label=r'$\alpha$={}$^\circ$, $\gamma$={}$^\circ$,'.format(alpha, inclination) + '\n1 coplanar',
        #          linewidth=self.font_line_sizes['linewidth'], linestyle=(0, (5, 5)), color='orange', zorder=5)

        plt.plot(n_sats_list, y_values,
                 label=r'$\alpha$={}$^\circ$, $\gamma$={}$^\circ$,'.format(alpha, inclination) + ' 10 + 1 orbit',
                 linewidth=self.font_line_sizes['linewidth'] * line_width_mult, linestyle='solid', color=colors_template[3], alpha=alpha_plots)

        # SNR + single + shifted + coplanar
        inclination = inclination_list[-1]
        y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha)
                                                   & (mmwave_math_results_df['inclination'] == inclination),
                                                   'capacity_gbps'])
        plt.plot(n_sats_list, y_values,
                 label=r'$\alpha$={}$^\circ$, $\gamma$={}$^\circ$'.format(alpha, inclination) + ', all orbits',
                 linewidth=self.font_line_sizes['linewidth'], linestyle='solid', color=colors_template[3])

        # # SNR + single + shifted + coplanar low inclination
        # color = color_dict['Orange']
        # inclination = inclination_list[0]
        # y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha)
        #                                            & (mmwave_math_results_df['inclination'] == inclination),
        #                                            'capacity_gbps'])
        # plt.plot(n_sats_list, y_values,
        #          label=r'mmWave $\alpha$={}$^\circ$, $\gamma$={}$^\circ$'.format(alpha, inclination),
        #          linewidth=self.font_line_sizes['linewidth'], linestyle='solid', color=color)


        # # SNR + single + shifted + ONE coplanar low inclination
        # inclination = inclination_list[0]
        # y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha)
        #                                            & (mmwave_math_results_df['inclination'] == inclination),
        #                                            'capacity_shifted_1_coplanar'])
        # plt.plot(n_sats_list, y_values,
        #          label=r'mmWave $\alpha$={}$^\circ$, $\gamma$={}$^\circ$, 1 coplanar'.format(alpha, inclination),
        #          linewidth=self.font_line_sizes['linewidth'], linestyle='dotted', color='k')

        text_scale_factor = 1.2
        plt.yscale('log')
        plt.xlabel('Satellites per orbit', fontsize=self.font_line_sizes['axisLabelSize'] * text_scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('Capacity [Gbps]', fontsize=self.font_line_sizes['axisLabelSize'] * text_scale_factor,
                   fontname=self.font_line_sizes['fontname'], labelpad=-4)
        plt.xticks([10, 100, 200, 300, 400, 500],
                   fontsize=self.font_line_sizes['ticksLabelSize'] * text_scale_factor,
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks([0.1, 1, 10, 100],
                   [0.1, 1, 10, 100],
                   fontsize=self.font_line_sizes['ticksLabelSize'] * text_scale_factor,
                   fontname=self.font_line_sizes['fontname'])

        # plt.vlines(350, 0.01, 100)

        plt.xlim(0, 500)
        plt.ylim(0.1, 100)

        # Adjustemnt of bbox to better fit figures 5b and 5c which have the legent outside of the bbox
        h, l = fig.get_legend_handles_labels()
        ph = [plt.scatter([], [], marker="", color=colors[1], s=100),
              plt.scatter([], [], marker=""),
              plt.scatter([], [], marker="", color=colors[0], s=100),
              plt.scatter([], [], marker="")]
        handles = ph[:1] + h[:2] + ph[1:2] + h[2:]
        labels = ["Sub-THz:"] + l[:2] + ["mmWave:"] + l[2:]
        box = fig.get_position()

        # fig.set_position([box.x0 * 0.75, box.y0, box.width * 0.8, box.height * 1.12])
        # leg = plt.legend(handles, labels, ncol=1, framealpha=1, fontsize=self.font_line_sizes['legendSize'],
        #                  loc='center right', columnspacing=0.1, bbox_to_anchor=(1.45, 0.5))

        fig.set_position([box.x0 * 0.75, box.y0 * 2.6, box.width * 1.14, box.height * 0.9])
        leg = fig.legend(handles, labels, ncol=3, framealpha=1, fontsize=self.font_line_sizes['legendSize'],
                         loc='center right', columnspacing=0.5, bbox_to_anchor=(1.03, -0.27), handletextpad=0.4,
                         handlelength=1.7)

        for vpack in leg._legend_handle_box.get_children()[:2]:
            for hpack in vpack.get_children()[:1]:
                hpack.get_children()[0].set_width(-8)


        extension = 'complete_constellation_capacity_vs_n_sats_' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def complete_constellation_n_interferers_vs_n_sats(self, file_ext='.jpg', thz_alpha_list=None,
                                                       mmwave_alpha_list=None, show=True):
        thz_alpha_list = self.thz_input_params['alpha_list'] if thz_alpha_list is None else thz_alpha_list
        mmwave_alpha_list = self.mmWave_input_params['alpha_list'] if mmwave_alpha_list is None else mmwave_alpha_list
        n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                self.orbital_input_params['n_sats_lims'][1],
                                self.orbital_input_params['n_sats_resolution'])
        bw_mmwave = self.mmWave_input_params['bandwidth']
        bw_thz = self.thz_input_params['bandwidth']

        # Results dataframes
        thz_math_results_df = self.thz_math_results_df
        mmwave_math_results_df = self.mmWave_math_results_df

        # Time averaging of results
        thz_math_results_df = thz_math_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(thz_math_results_df)
        mmwave_math_results_df = mmwave_math_results_df.groupby(['n_sats', 'alpha'], as_index=False).mean()
        adjust_db_columns(mmwave_math_results_df)

        _, _, colors = tk.templateColors(iterable_size=2, reverse=False)

        fig = get_ax(figsize=(10, 6))

        # Capacity with interference
        plt.grid()
        # THz
        for i, alpha in enumerate(thz_alpha_list):
            y_values = list(thz_math_results_df.loc[(thz_math_results_df['alpha'] == alpha), 'n_interferers'])
            plt.plot(n_sats_list, y_values, color=colors[1],
                     label=r'Thz: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle='solid')

        # mmWave
        for i, alpha in enumerate(mmwave_alpha_list):
            y_values = list(mmwave_math_results_df.loc[(mmwave_math_results_df['alpha'] == alpha), 'n_interferers'])
            plt.plot(n_sats_list, y_values, color=colors[0],
                     label=r'mmWave: $\alpha$={}$^\circ$'.format(alpha),
                     linewidth=self.font_line_sizes['linewidth'], linestyle='dotted')

        plt.xlabel('Satellites per orbit', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.ylabel('# interferers', fontsize=self.font_line_sizes['axisLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.xticks([10, 25, 50, 75, 100, 125, 150, 175, 200], fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.yticks(fontsize=self.font_line_sizes['ticksLabelSize'],
                   fontname=self.font_line_sizes['fontname'])
        plt.legend(ncol=1, framealpha=0.7, fontsize=self.font_line_sizes['legendSize'])
        # plt.xlim(0, 500)
        # plt.ylim(-25, 14)

        extension = 'complete_constellation_n_interferers_vs_n_sats_' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    # Miscellaneous plots
    def lim_n_satellites_sir(self, file_ext='.jpg', show=True):
        n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                self.orbital_input_params['n_sats_lims'][1],
                                self.orbital_input_params['n_sats_resolution_math'])

        # Numerator of SIR expression
        numerator = 1 / (1 - np.cos(2 * np.pi / n_sats_list))

        # Denominator of SIR expression
        h = 500e3
        alpha_rad = np.deg2rad(30)
        denominator = np.zeros(len(n_sats_list))
        for j, n_sats in enumerate(n_sats_list):
            cond1 = n_sats / np.pi * np.arccos(Re / (Re + h))
            cond2 = 1 + n_sats / (2 * np.pi) * alpha_rad
            j_max = int(min(cond1, cond2))
            den = 0
            for i in range(2, j_max):
                den += 1 / (1 - np.cos(2 * np.pi * i / n_sats))
            denominator[j] = den

        # Plotting
        fig = get_ax(figsize=(10 * 2, 10))
        scale_factor = 0.8

        ax1 = plt.subplot(221)
        plt.grid()
        plt.plot(n_sats_list, numerator, color='b', linewidth=self.font_line_sizes['linewidth'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel('Numerator SIR linear', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)

        ax2 = plt.subplot(222)
        plt.grid()
        plt.plot(n_sats_list, denominator, color='b', linewidth=self.font_line_sizes['linewidth'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel('Denominator SIR linear', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)

        ax3 = plt.subplot(223)
        plt.grid()
        plt.plot(n_sats_list, 10 * np.log10(numerator / denominator), color='b',
                 linewidth=self.font_line_sizes['linewidth'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel('SIR [dB]', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)

        ax4 = plt.subplot(224)
        plt.grid()
        plt.plot(n_sats_list, numerator / denominator, color='b', linewidth=self.font_line_sizes['linewidth'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)
        plt.ylabel('SIR linear', fontsize=self.font_line_sizes['axisLabelSize'] * scale_factor)

        extension = 'single_sir_limit_analysis' + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def n_interferers_thresholds(self, file_ext='.jpg', show=True, h=500e3, alpha=30):
        n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                self.orbital_input_params['n_sats_lims'][1],
                                self.orbital_input_params['n_sats_resolution_math'])
        h_list = np.arange(self.orbital_input_params['h_lims'][0],
                           self.orbital_input_params['h_lims'][1],
                           self.orbital_input_params['h_resolution_math'])
        alpha_list = self.link_budget_input_params['alpha_list']

        alpha_rad = np.deg2rad(alpha)

        cond1 = n_sats_list / np.pi * np.arccos(Re / (Re + h))
        cond2 = 1 + n_sats_list / (2 * np.pi) * alpha_rad
        ratio = cond1 / cond2

        N_th = np.pi / (np.arccos(Re / (Re + h_list)) - alpha_rad / 2)
        N_th_denominator_th = 2 * np.arccos(Re / (Re + h_list))

        n_sats_min_list = [6, 8, 10]
        alpha_low_th = []
        for n_sats_min in n_sats_min_list:
            alpha_max = 2 * np.arccos(Re / (Re + h_list)) - 2 * np.pi / n_sats_min
            alpha_low_th.append(alpha_max)

        # Plotting
        fig = get_ax(figsize=(10 * 2, 6 * 2))

        ax1 = plt.subplot(221)
        plt.grid()
        plt.plot(n_sats_list, cond1, color='b', linewidth=self.font_line_sizes['linewidth'], label='cond1 blockage')
        plt.plot(n_sats_list, cond2, color='r', linewidth=self.font_line_sizes['linewidth'], label='cond2 beamwidth')
        plt.title(r'h={:.0f}Km, $\alpha$={}$^\circ$'.format(h * 1e-3, alpha),
                  fontsize=self.font_line_sizes['titleSize'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('Number of interfering satellites', fontsize=self.font_line_sizes['axisLabelSize'])

        ax2 = plt.subplot(222)
        plt.grid()
        plt.plot(n_sats_list, ratio, color='k', linewidth=self.font_line_sizes['linewidth'])
        plt.title(r'h={:.0f}Km, $\alpha$={}$^\circ$'.format(h * 1e-3, alpha),
                  fontsize=self.font_line_sizes['titleSize'])
        plt.xlabel('Number of satellites', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('cond1/cond2', fontsize=self.font_line_sizes['axisLabelSize'])

        ax3 = plt.subplot(223)
        plt.grid()
        plt.plot(h_list * 1e-3, N_th, color='k', linewidth=self.font_line_sizes['linewidth'])
        plt.title(r'$\alpha$={}$^\circ$'.format(alpha),
                  fontsize=self.font_line_sizes['titleSize'])
        plt.xlabel('Orbit altitude [Km]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('Number of interferers threshold', fontsize=self.font_line_sizes['axisLabelSize'])

        ax4 = plt.subplot(224)
        plt.grid()
        plt.plot(h_list * 1e-3, np.rad2deg(N_th_denominator_th), color='k', linewidth=self.font_line_sizes['linewidth'])
        plt.title('Regions where # interferers is clearly defined',
                  fontsize=self.font_line_sizes['titleSize'])
        plt.xlabel('Orbit altitude [Km]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.ylabel('Beamwidth [deg]', fontsize=self.font_line_sizes['axisLabelSize'])
        plt.vlines(h * 1e-3, 0, 90, color='k', linestyles='dotted')
        plt.xlim([h_list[0] * 1e-3, h_list[-1] * 1e-3])
        plt.ylim([0, 90])
        x = np.concatenate([h_list * 1e-3, np.flip(h_list * 1e-3)])
        y = np.concatenate([np.rad2deg(N_th_denominator_th), 90 * np.ones(len(h_list))])
        plt.fill(x, y, alpha=0.2, color='b', edgecolor='k')
        plt.annotate(r'$N_i = \frac{N}{\pi}arcos(\frac{R_e}{R_e+h})-1$', [400, 70],
                     fontsize=self.font_line_sizes['axisLabelSize'])
        for i, alpha_max in enumerate(alpha_low_th):
            plt.plot(h_list * 1e-3, np.rad2deg(alpha_max), color='k',
                     linewidth=self.font_line_sizes['linewidth'])
            values = abs(h_list * 1e-3 - 1000)
            ind = min(range(len(values)), key=values.__getitem__)
            plt.annotate(r'$N_{{min}}$ = {}'.format(n_sats_min_list[i]), [1000, np.rad2deg(alpha_max[ind]) + 2],
                         rotation=15,
                         fontsize=self.font_line_sizes['axisLabelSize'] * 0.7)
        y = np.concatenate([np.rad2deg(alpha_low_th[-1]), np.zeros(len(h_list))])
        plt.fill(x, y, alpha=0.2, color='r', edgecolor='k')
        plt.annotate(r'$N_i = \frac{N}{2\pi}\alpha$', [1600, 5],
                     fontsize=self.font_line_sizes['axisLabelSize'])
        plt.scatter(h * 1e-3 * np.ones(len(alpha_list)), alpha_list, s=150, marker='X', color='k', zorder=10)

        extension = 'single_n_interferers_thresholds_analysis_h_{:.0f}_alpha_{}'.format(h * 1e-3, alpha) + file_ext
        plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
