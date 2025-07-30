import sys
import os
import numpy as np
import pandas as pd
import simulator.constants as c
import simulator.toolkit as tk
import simulator.astrodynamics as ad
from tqdm import tqdm
from datetime import datetime, timedelta
from itertools import product
from simulator.simulation import Simulation
from simulator.file_utils import load_or_recompute_and_cache
from simulator.satellite import Orbit, Constellation
from simulator.link_budget import rx_power
from sergi_orbits_visualization import plot_from_above
import imageio
import matplotlib.pyplot as plt

# Simulator path
sys.path.append('../simulator')

# Constants
Re = c.EARTH_RADIUS_M
cache_path = '../interference_analysis/cache'

tqdm.pandas()


class Analysis:
    """Class containing all the methods for analysis

    This class contains all the methods to generate the results (in a pandas.Dataframe format)
    for all of the analyzed scenarios (Single orbit, Co-planar orbits and Shifted Orbits)
    and with the different approaches (math, simulation).

    Args:
        common_input_params (dict): Input parameters common to all analysis
        especial_input_params(dict): Input parameters specific to the analysis
    """

    def __init__(self, link_budget_input_params, orbital_input_params, results_columns):
        self.link_budget_input_params = link_budget_input_params
        self.orbital_input_params = orbital_input_params
        self.results_columns = results_columns

    def single_orbit_math(self):
        """Mathematical analysis of the Single orbit scenario

        Returns:
            Dataframe: Results in a Dataframe format. Input columns: ['alpha', 'n_sats', 'h']

        """
        # ------Inputs------
        alpha_list = self.link_budget_input_params['alpha_list']
        n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                self.orbital_input_params['n_sats_lims'][1],
                                self.orbital_input_params['n_sats_resolution_math'])
        h_list = np.arange(self.orbital_input_params['h_lims'][0],
                           self.orbital_input_params['h_lims'][1],
                           self.orbital_input_params[
                               'h_resolution_math'])  # altitude difference between low and upper orbit
        input_columns = ['alpha', 'n_sats', 'h']
        replace = self.orbital_input_params['math_replace']

        # Link budget inputs
        p_tx_dbm = self.link_budget_input_params['p_tx_dbm']  # W = 10 ** ((P(dBm)-30)/10)
        p_tx = 10 ** (p_tx_dbm / 10)
        fc = self.link_budget_input_params['center_frequency']
        bw = self.link_budget_input_params['bandwidth']
        t_system = self.link_budget_input_params['T_system']
        # ------------------

        # Object Simulation creation, with the corresponding results folder
        t_start = datetime(2022, 1, 1)
        sim = Simulation(t_start=t_start)
        np.random.seed(0)

        # sim params on a .txt at the simulation results folder
        with open(os.path.join(sim.results_folder, 'sim_params.txt'), 'w') as file:
            file.write('SINGLE ORBIT MATH ANALYSIS\n')
            file.write('\nCommon Input parameters\n')
            for k, v in self.link_budget_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
            file.write('\nSingle orbit Input parameters\n')
            for k, v in self.orbital_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
        file.close()

        def generate_results():
            # Results dataframe creation
            results_df = pd.DataFrame(list(product(alpha_list, n_sats_list, h_list)), columns=input_columns)

            def compute_row_results(row):
                n_sats = row['n_sats']
                alpha = row['alpha']
                h = row['h']
                results = []
                p_i_mw = 0

                alpha_rad = np.deg2rad(alpha)

                d_i_min = None

                cond1 = n_sats / np.pi * np.arccos(Re / (Re + h))
                cond2 = 1 + n_sats / (2 * np.pi) * alpha_rad
                i_max = int(min(cond1, cond2))
                n_i = i_max - 1

                lambda_fc = c.SPEED_OF_LIGHT / fc
                num = lambda_fc ** 2 * p_tx
                den_tx = 8 * (np.pi ** 2) * (1 - np.cos(alpha_rad / 2)) ** 2 * (Re + h) ** 2 * (
                        1 - np.cos(2 * np.pi / n_sats))

                p_rx_dbm = (10 * np.log10(num / den_tx))
                d_tx_rx = np.sqrt(2 * (Re + h) ** 2 * (1 - np.cos(2 * np.pi / n_sats)))

                for i in range(2, i_max + 1):
                    den = 8 * (np.pi ** 2) * (1 - np.cos(alpha_rad / 2)) ** 2 * (Re + h) ** 2 * (
                            1 - np.cos(2 * np.pi * i / n_sats))

                    p_i_mw += num / den
                    d_i = np.sqrt(2 * (Re + h) ** 2 * (1 - np.cos(2 * np.pi * i / n_sats)))

                    # Distance to nearest interferer
                    if not d_i_min:
                        d_i_min = d_i
                    elif d_i < d_i_min:
                        d_i_min = d_i

                results.append(d_tx_rx)  # Tx-Rx distance
                results.append(p_rx_dbm)
                p_rx_linear = 10 ** (p_rx_dbm / 10)
                results.append(p_rx_linear)
                results.append(d_i_min)
                results.append(p_i_mw)
                if p_i_mw == 0:
                    results.append(None)
                else:
                    results.append(10 * np.log10(p_i_mw))
                results.append(n_i)

                # SIR
                results.append(p_rx_linear / p_i_mw if p_i_mw else None)

                # SNR
                system_t = self.link_budget_input_params['T_system']
                bw = self.link_budget_input_params['bandwidth']
                p_noise_mw = c.BOLTZMAN * system_t * bw * 1e3
                results.append(p_rx_linear / p_noise_mw)

                # SINR
                sinr_linear = p_rx_linear / (p_i_mw + p_noise_mw)
                results.append(sinr_linear)

                # Capacity (Gbps)
                results.append(bw * np.log2(1 + sinr_linear) * 1e-9 if sinr_linear != 0 else None)

                '''
                # Link budget
                g1 = 2 / (1 - np.cos(alpha_rad / 2))  # Gain in the direction of the main beam
                g1_db = 10 * np.log10(g1)
                rx_snr = snr(p_tx_dbm, d_tx_rx, fc, g1_db, bw, antenna_T, l_abs=0, g_rx=None, nf=NF_db)
                results.append(rx_snr)
                '''

                return results

            results_df[self.results_columns] = results_df.progress_apply(compute_row_results, axis=1,
                                                                         result_type='expand')

            return results_df

        # Dataframe cache file format
        df_extension = self.orbital_input_params['math_results_extension'] + self.link_budget_input_params['extension']
        df_path = os.path.join(cache_path, df_extension)
        rep = True if not os.path.isfile(df_path) or replace else False

        load_or_recompute_and_cache(df_path, generate_results, replace=rep, verbose=True)

    def single_orbit_simulation(self):
        """Simulation analysis of the Single orbit scenario

        Returns:
            Dataframe: Results in a Dataframe format. Input columns: ['alpha', 'n_sats', 'h']

        """
        # ------Inputs------
        alpha_list = self.link_budget_input_params['alpha_list']
        n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                self.orbital_input_params['n_sats_lims'][1],
                                self.orbital_input_params['n_sats_resolution_sim'])
        h_list = np.arange(self.orbital_input_params['h_lims'][0],
                           self.orbital_input_params['h_lims'][1],
                           self.orbital_input_params[
                               'h_resolution_sim'])  # altitude difference between low and upper orbit
        input_columns = ['alpha', 'n_sats', 'h']
        replace = self.orbital_input_params['sim_replace']

        # Link budget inputs
        p_tx_dbm = self.link_budget_input_params['p_tx_dbm']  # W = 10 ** ((P(dBm)-30)/10)
        p_tx = 10 ** (p_tx_dbm / 10)
        fc = self.link_budget_input_params['center_frequency']
        bw = self.link_budget_input_params['bandwidth']
        t_system = self.link_budget_input_params['T_system']
        # ------------------

        # Object Simulation creation, with the corresponding results folder
        t_start = datetime(2022, 1, 1)
        sim = Simulation(t_start=t_start)
        np.random.seed(0)

        # sim params on a .txt at the simulation results folder
        with open(os.path.join(sim.results_folder, 'sim_params.txt'), 'w') as file:
            file.write('SINGLE ORBIT SIMULATION ANALYSIS\n')
            file.write('\nCommon Input parameters\n')
            for k, v in self.link_budget_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
            file.write('\nSingle orbit Input parameters\n')
            for k, v in self.orbital_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
        file.close()

        def generate_results():
            # Results dataframe creation
            results_df = pd.DataFrame(list(product(alpha_list, n_sats_list, h_list)), columns=['alpha', 'n_sats', 'h'])

            # Fixed orbital parameters
            e = 0.0  # eccentricity
            omega = 0  # Right ascension of ascending node in degrees
            inc = 0  # Inclination in degrees
            w = 0  # Argument of perigee in degrees

            rx_id = 0
            tx_id = 1

            def compute_row_results(row):
                n_sats = int(row['n_sats'])
                alpha = row['alpha']
                h = row['h']
                results = []

                alpha_rad = np.deg2rad(alpha)
                half_cone_angle = alpha / 2
                g1 = 2 / (1 - np.cos(alpha_rad / 2))  # Gain in the direction of the main beam
                g1_db = 10 * np.log10(g1)

                a = Re + h
                orbit = Orbit(sim, e, a, omega, inc, w)
                orbit.add_satellites(n_sats=n_sats)
                const = Constellation(sim, orbits=[orbit])
                const.update_SSPs()  # Required before computing distances between satellites

                tx_sat = const.satellites[tx_id]
                rx_sat = const.satellites[rx_id]

                # Distance
                distance = tk.sat_to_sat_disance(tx_sat.xyz_r, rx_sat.xyz_r)
                results.append(distance)

                # Rx power
                p_rx_dbm = rx_power(p_tx_dbm, distance, fc, l_abs=0, g_tx=g1_db)
                results.append(p_rx_dbm)
                p_rx_linear = 10 ** (p_rx_dbm / 10)
                results.append(p_rx_linear)

                # Interference
                p_i_mw = 0
                n_i = 0
                d_i_min = None
                for i, sat in enumerate(const.satellites[:-1]):
                    if i == 0:
                        continue  # If not, we would be considering the link of interest as interference
                    else:
                        i_rx_sat = const.satellites[i]
                        i_tx_sat = const.satellites[(i + 1) % n_sats]
                        if tk.check_interference(rx_sat, tx_sat, i_rx_sat, i_tx_sat, half_cone_angle=half_cone_angle):
                            n_i += 1
                            distance_interferer = tk.sat_to_sat_disance(i_tx_sat.xyz_r, rx_sat.xyz_r)
                            g_i_tx = g1_db
                            p_i_dbm = rx_power(p_tx_dbm, distance_interferer, fc, l_abs=0, g_tx=g_i_tx)
                            p_i_mw += (10 ** (p_i_dbm / 10))
                            if not d_i_min:
                                d_i_min = distance_interferer
                            elif distance_interferer < d_i_min:
                                d_i_min = distance_interferer
                results.append(d_i_min)
                results.append(p_i_mw)
                if p_i_mw == 0:
                    results.append(None)
                else:
                    results.append(10 * np.log10(p_i_mw))
                results.append(n_i)

                # SIR
                results.append(p_rx_linear / p_i_mw if p_i_mw else None)

                # SNR
                system_t = self.link_budget_input_params['T_system']
                bw = self.link_budget_input_params['bandwidth']
                p_noise_mw = c.BOLTZMAN * system_t * bw * 1e3
                results.append(p_rx_linear / p_noise_mw)

                # SINR
                sinr_linear = p_rx_linear / (p_i_mw + p_noise_mw)
                results.append(sinr_linear)

                # Capacity (Gbps)
                results.append(bw * np.log2(1 + sinr_linear) * 1e-9 if sinr_linear != 0 else None)

                return results

            results_df[self.results_columns] = results_df.progress_apply(compute_row_results, axis=1,
                                                                         result_type='expand')

            return results_df

        # Dataframe cache file format
        df_extension = self.orbital_input_params['sim_results_extension'] + self.link_budget_input_params['extension']
        df_path = os.path.join(cache_path, df_extension)
        rep = True if not os.path.isfile(df_path) or replace else False

        load_or_recompute_and_cache(df_path, generate_results, replace=rep, verbose=True)

    def coplanar_orbits_math(self):
        """Mathematical analysis of the Co-planar orbits scenario

        Returns:
            Dataframe: Results in a Dataframe format. Input columns: ['alpha', 'n_sats', 'h_high', 'time']

        """
        # ------Inputs------
        alpha_list = self.link_budget_input_params['alpha_list']
        n_sats_list = self.orbital_input_params['n_sats_list']
        # Altitude vector for the upper orbit
        h_high_list = np.arange(self.orbital_input_params['h_lims'][0],
                                self.orbital_input_params['h_lims'][1],
                                self.orbital_input_params['h_resolution_math'])
        h_low = self.orbital_input_params['h_low']

        beta_max = 2 * np.pi * 1 / min(n_sats_list)
        beta_steps = self.orbital_input_params['beta_steps_math']

        # Link budget inputs
        p_tx_dbm = self.link_budget_input_params['p_tx_dbm']  # W = 10 ** ((P(dBm)-30)/10)
        p_tx = 10 ** (p_tx_dbm / 10)
        fc = self.link_budget_input_params['center_frequency']

        replace = self.orbital_input_params['math_replace']
        higher_orbit_only = self.orbital_input_params['higher_orbit_only']
        # ------------------

        # Object Simulation creation, with the corresponding results folder
        t_start = datetime(2022, 1, 1)
        sim = Simulation(t_start=t_start)
        np.random.seed(0)

        # sim params on a .txt at the simulation results folder
        with open(os.path.join(sim.results_folder, 'sim_params.txt'), 'w') as file:
            file.write('COPLANAR ORBITS MATH ANALYSIS\n')
            file.write('\nCommon Input parameters\n')
            for k, v in self.link_budget_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
            file.write('\nCoplanar orbit Input parameters\n')
            for k, v in self.orbital_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
        file.close()

        def generate_results():
            # Results dataframe creation
            results_df = pd.DataFrame(list(product(alpha_list, n_sats_list, h_high_list)),
                                      columns=['alpha', 'n_sats', 'h_high'])

            # Compute angular offset and time vector for each number of satellites and altitudes, respectively
            ts = []
            for n_sats in n_sats_list:
                for _ in alpha_list:
                    for h_high in h_high_list:
                        t1 = ad.period_from_semi_major_axis(Re + h_low)
                        # T2 = ad.period_from_semi_major_axis(Re + h_high)
                        # We simulate the same interval of time for all h of the upper orbit
                        # The simulated time corresponds to the largest period of the interference pattern, i.e.,
                        # when the two orbits are the closest together (smaller relative angular speed)
                        t2 = ad.period_from_semi_major_axis(Re + min(h_high_list))
                        # beta_max = 2 * np.pi * 1 / n_sats

                        t_beta_max = beta_max / (2 * np.pi * (1 / t1 - 1 / t2))

                        time_steps = [sim.t_start + timedelta(seconds=t_beta_max) / beta_steps * i for i in
                                      range(beta_steps)]
                        ts.append(time_steps)
            results_df['time'] = ts
            results_df = results_df.explode('time', ignore_index=True)

            def compute_row_results(row):
                n_sats = row['n_sats']
                alpha = row['alpha']
                h_high = row['h_high']
                t = row['time']
                results = []
                p_i_mw = 0
                n_i = 0

                alpha_rad = np.deg2rad(alpha)
                g1 = 2 / (1 - np.cos(alpha_rad / 2))  # Gain in the direction of the main beam
                g1_db = 10 * np.log10(g1)
                half_cone_angle = alpha / 2

                T1 = ad.period_from_semi_major_axis(Re + h_low)
                T2 = ad.period_from_semi_major_axis(Re + h_high)

                T_beta_max = beta_max / (2 * np.pi * (1 / T1 - 1 / T2))
                beta = ((t - t_start).total_seconds() / T_beta_max) * beta_max

                d_i_min = None

                # If we consider same orbit as interference too
                if not higher_orbit_only:
                    cond1 = n_sats / np.pi * np.arccos(Re / (Re + h_low))
                    cond2 = 1 + n_sats / (2 * np.pi) * alpha_rad
                    i_max = int(min(cond1, cond2))
                    n_i = i_max - 1

                    lambda_fc = c.SPEED_OF_LIGHT / fc
                    num = lambda_fc ** 2 * p_tx

                    for i in range(2, i_max + 1):
                        den = 8 * (np.pi ** 2) * (1 - np.cos(alpha_rad / 2)) ** 2 * (Re + h_low) ** 2 * (
                                1 - np.cos(2 * np.pi * i / n_sats))

                        p_i_mw += num / den
                        d_i = np.sqrt(2 * (Re + h_low) ** 2 * (1 - np.cos(2 * np.pi * i / n_sats)))

                        # Distance to nearest interferer
                        if not d_i_min:
                            d_i_min = d_i
                        elif d_i < d_i_min:
                            d_i_min = d_i

                # Higher orbit interference
                A_angle = np.pi / 2 - np.pi / n_sats
                AC_unit = np.array([np.sin(A_angle), -np.cos(A_angle)])
                for j in range(n_sats):
                    ############ Psi j ############
                    A_vec = np.array([0, Re + h_low])

                    O_angle = -beta + 2 * np.pi * j / n_sats

                    B_vec = (Re + h_high) * np.array([np.sin(O_angle), np.cos(O_angle)])
                    AB_vec = B_vec - A_vec

                    # Dot product
                    dot = AC_unit @ AB_vec
                    # Norm
                    AB_norm = np.linalg.norm(AB_vec)
                    # Arc cosine
                    # psi_j = np.arccos(dot / AB_norm)  # WRONG psi_j has to be signed!
                    cross = np.cross(AC_unit, AB_vec)
                    psi_j = np.arctan2(cross, dot)
                    psi_j_deg = psi_j * 180 / np.pi

                    ############ Psi j interferer ############
                    O_angle_t = O_angle - 2 * np.pi / n_sats
                    D_vec = (Re + h_high) * np.array([np.sin(O_angle_t), np.cos(O_angle_t)])
                    BD_vec = D_vec - B_vec
                    BA_vec = A_vec - B_vec

                    # Dot product
                    dot = BD_vec @ BA_vec
                    # Norm
                    BD_norm = np.linalg.norm(BD_vec)
                    BA_norm = np.linalg.norm(BA_vec)
                    # Arc cosine
                    # psi_j_i = np.arccos(dot / (BD_norm * BA_norm))  # WRONG psi_j_i has to be signed!
                    cross = np.cross(BD_vec, BA_vec)
                    psi_j_i = np.arctan2(cross, dot)
                    psi_j_i_deg = psi_j_i * 180 / np.pi

                    # Visibility with interferer (not blocked by Earth)
                    j_visible = psi_j > np.pi / n_sats - np.arccos(Re / (Re + h_low))

                    d_i = np.linalg.norm(AB_vec)

                    if j_visible and abs(psi_j_deg) <= half_cone_angle and abs(psi_j_i_deg) <= half_cone_angle:
                        # Distance to nearest interferer
                        if not d_i_min:
                            d_i_min = d_i
                        elif d_i < d_i_min:
                            d_i_min = d_i
                        # Number of interferers
                        n_i += 1
                        # Interference linear
                        p_i_dbm = rx_power(p_tx_dbm, d_i, fc, l_abs=0, g_tx=g1_db)
                        p_i_mw += (10 ** (p_i_dbm / 10))

                d_tx_rx = np.sqrt(2 * (Re + h_low) ** 2 * (1 - np.cos(2 * np.pi / n_sats)))
                results.append(d_tx_rx)  # Tx-Rx distance
                p_rx_dbm = rx_power(p_tx_dbm, d_tx_rx, fc, l_abs=0, g_tx=g1_db)  # Rx power
                results.append(p_rx_dbm)
                p_rx_linear = 10 ** (p_rx_dbm / 10)
                results.append(p_rx_linear)
                results.append(d_i_min)
                results.append(p_i_mw)
                if p_i_mw == 0:
                    results.append(None)
                else:
                    results.append(10 * np.log10(p_i_mw))
                results.append(n_i)

                # SIR
                results.append(p_rx_linear / p_i_mw if p_i_mw else None)

                # SNR
                system_t = self.link_budget_input_params['T_system']
                bw = self.link_budget_input_params['bandwidth']
                p_noise_mw = c.BOLTZMAN * system_t * bw * 1e3
                results.append(p_rx_linear / p_noise_mw)

                # SINR
                sinr_linear = p_rx_linear / (p_i_mw + p_noise_mw)
                results.append(sinr_linear)

                # Capacity (Gbps)
                results.append(bw * np.log2(1 + sinr_linear) * 1e-9 if sinr_linear != 0 else None)

                return results

            results_df[self.results_columns] = results_df.progress_apply(compute_row_results, axis=1,
                                                                         result_type='expand')

            return results_df

        # Dataframe cache file format
        df_extension = self.orbital_input_params['math_results_extension'] + '_higher_orbit_only_' + str(
            higher_orbit_only) \
                       + self.link_budget_input_params['extension']
        df_path = os.path.join(cache_path, df_extension)
        rep = True if not os.path.isfile(df_path) or replace else False

        load_or_recompute_and_cache(df_path, generate_results, replace=rep, verbose=True)

    def coplanar_orbits_simulation(self):
        """Simulation analysis of the Co-planar orbits scenario

        Returns:
            Dataframe: Results in a Dataframe format. Input columns: ['alpha', 'n_sats', 'h_high', 'time']

        """
        # ------Inputs------
        alpha_list = self.link_budget_input_params['alpha_list']
        n_sats_list = self.orbital_input_params['n_sats_list']
        # Altitude vector for the upper orbit
        h_high_list = np.arange(self.orbital_input_params['h_lims'][0],
                                self.orbital_input_params['h_lims'][1],
                                self.orbital_input_params['h_resolution_sim'])
        h_low = self.orbital_input_params['h_low']

        beta_max = 2 * np.pi * 1 / min(n_sats_list)
        beta_steps = self.orbital_input_params['beta_steps_sim']

        # Link budget inputs
        p_tx_dbm = self.link_budget_input_params['p_tx_dbm']  # W = 10 ** ((P(dBm)-30)/10)
        fc = self.link_budget_input_params['center_frequency']

        replace = self.orbital_input_params['sim_replace']
        higher_orbit_only = self.orbital_input_params['higher_orbit_only']
        # ------------------

        # Object Simulation creation, with the corresponding results folder
        t_start = datetime(2022, 1, 1)
        sim = Simulation(t_start=t_start)
        np.random.seed(0)

        # sim params on a .txt at the simulation results folder
        with open(os.path.join(sim.results_folder, 'sim_params.txt'), 'w') as file:
            file.write('COPLANAR ORBITS MATH ANALYSIS\n')
            file.write('\nCommon Input parameters\n')
            for k, v in self.link_budget_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
            file.write('\nCoplanar orbit Input parameters\n')
            for k, v in self.orbital_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
        file.close()

        def generate_results():
            # Results dataframe creation
            results_df = pd.DataFrame(list(product(alpha_list, n_sats_list, h_high_list)),
                                      columns=['alpha', 'n_sats', 'h_high'])

            # Compute angular offset and time vector for each number of satellites and altitudes, respectively
            ts = []
            for _ in n_sats_list:
                for _ in alpha_list:
                    for h_high in h_high_list:
                        T1 = ad.period_from_semi_major_axis(Re + h_low)
                        # T2 = ad.period_from_semi_major_axis(Re + h_high)
                        T2 = ad.period_from_semi_major_axis(Re + min(h_high_list))

                        T_beta_max = beta_max / (2 * np.pi * (1 / T1 - 1 / T2))

                        time_steps = [sim.t_start + timedelta(seconds=T_beta_max) / beta_steps * i for i in
                                      range(beta_steps)]
                        ts.append(time_steps)
            results_df['time'] = ts
            results_df = results_df.explode('time', ignore_index=True)

            # Fixed orbital parameters
            e = 0.0  # eccentricity
            omega = 0  # Right ascension of ascending node in degrees
            inc = 0  # Inclination in degrees
            w = 0  # Argument of perigee in degrees

            rx_id = 0
            tx_id = 1

            def compute_row_results(row):
                n_sats = row['n_sats']
                h_high = row['h_high']
                alpha = row['alpha']
                t = row['time']
                results = []

                g1 = 2 / (1 - np.cos(np.deg2rad(alpha) / 2))  # Gain in the direction of the main beam
                g1_db = 10 * np.log10(g1)
                half_cone_angle = alpha / 2

                d_i_min = None

                # Orbit objects creation
                # Lower orbit
                a = Re + h_low  # Semimajor axis in meters
                orbit_low = Orbit(sim, e, a, omega, inc, w, initial_anomaly=0)
                orbit_low.add_satellites(n_sats=n_sats)

                # Higher orbit
                a = Re + h_high  # Semimajor axis in meters
                orbit_high = Orbit(sim, e, a, omega, inc, w, initial_anomaly=0)
                orbit_high.add_satellites(n_sats=n_sats)

                # Constellation creation and propagation
                const = Constellation(sim, orbits=[orbit_low, orbit_high])
                const.update_SSPs(time=t)  # Propagate orbit to corresponding timestep

                # Link of interest in the lower orbit
                rx_sat = const.orbits[0].satellites[rx_id]
                tx_sat = const.orbits[0].satellites[tx_id]

                # Plot antenna diagrams

                # Distance
                distance = tk.sat_to_sat_disance(tx_sat.xyz_r, rx_sat.xyz_r)
                results.append(distance)

                # Rx power
                p_rx_dbm = rx_power(p_tx_dbm, distance, fc, l_abs=0, g_tx=g1_db)
                results.append(p_rx_dbm)
                p_rx_linear = 10 ** (p_rx_dbm / 10)
                results.append(p_rx_linear)

                # Interference
                p_i_mw = 0
                n_i = 0

                if higher_orbit_only:
                    orbits = const.orbits[1:]
                else:
                    orbits = const.orbits

                for o, orbit in enumerate(orbits):
                    for i, sat in enumerate(orbit.satellites):
                        if orbit == const.orbits[0] and (i == rx_id or (i + 1) % len(
                                orbit.satellites) == rx_id):  # Link of interest is not considered interference
                            continue
                        i_rx_sat = orbit.satellites[i]
                        i_tx_sat = orbit.satellites[(i + 1) % len(orbit.satellites)]
                        if tk.check_interference(rx_sat, tx_sat, i_rx_sat, i_tx_sat,
                                                 half_cone_angle=half_cone_angle):
                            n_i += 1
                            distance_interferer = tk.sat_to_sat_disance(i_tx_sat.xyz_r, rx_sat.xyz_r)
                            g_i_tx = g1_db
                            p_i_dbm = rx_power(p_tx_dbm, distance_interferer, fc, l_abs=0, g_tx=g_i_tx)
                            p_i_mw += (10 ** (p_i_dbm / 10))
                            # Distance to nearest interferer
                            if not d_i_min:
                                d_i_min = distance_interferer
                            elif distance_interferer < d_i_min:
                                d_i_min = distance_interferer
                results.append(d_i_min)
                results.append(p_i_mw)
                if p_i_mw == 0:
                    results.append(None)
                else:
                    results.append(10 * np.log10(p_i_mw))
                results.append(n_i)

                # SIR
                results.append(p_rx_linear / p_i_mw if p_i_mw else None)

                # SNR
                system_t = self.link_budget_input_params['T_system']
                bw = self.link_budget_input_params['bandwidth']
                p_noise_mw = c.BOLTZMAN * system_t * bw * 1e3
                results.append(p_rx_linear / p_noise_mw)

                # SINR
                sinr_linear = p_rx_linear / (p_i_mw + p_noise_mw)
                results.append(sinr_linear)

                # Capacity (Gbps)
                results.append(bw * np.log2(1 + sinr_linear) * 1e-9 if sinr_linear != 0 else None)

                return results

            results_df[self.results_columns] = results_df.progress_apply(compute_row_results, axis=1,
                                                                         result_type='expand')

            return results_df

        # Dataframe cache file format
        df_extension = self.orbital_input_params['sim_results_extension'] + '_higher_orbit_only_' + str(
            higher_orbit_only) \
                       + self.link_budget_input_params['extension']
        df_path = os.path.join(cache_path, df_extension)
        rep = True if not os.path.isfile(df_path) or replace else False

        load_or_recompute_and_cache(df_path, generate_results, replace=rep, verbose=True)

    def shifted_orbits_math(self):
        """Mathematical analysis of the Shifted orbits scenario

        Returns:
            Dataframe: Results in a Dataframe format. Input columns:
            ['alpha', 'n_sats', 'h', 'inclination', 'beta', 'time']

        """
        # ------Inputs------
        try:
            alpha_list = np.arange(self.orbital_input_params['alpha_lims'][0],
                                   self.orbital_input_params['alpha_lims'][1],
                                   self.orbital_input_params['alpha_resolution_math'])
        except:
            alpha_list = self.link_budget_input_params['alpha_list']

        try:
            n_sats_list = self.orbital_input_params['n_sats_list']
        except:
            try:
                n_sats_list = self.link_budget_input_params['n_sats_list']
            except:
                n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                        self.orbital_input_params['n_sats_lims'][1],
                                        self.orbital_input_params['n_sats_resolution_math'])

        try:
            inclination_list = np.arange(self.orbital_input_params['inclination_lims'][0],
                                         self.orbital_input_params['inclination_lims'][1],
                                         self.orbital_input_params['inclination_resolution_math'])
        except:
            inclination_list = self.link_budget_input_params['inclination_list']

        h_list = np.arange(self.orbital_input_params['h_lims'][0],
                           self.orbital_input_params['h_lims'][1],
                           self.orbital_input_params['h_resolution_math'])

        n_timesteps = self.orbital_input_params['n_timesteps_math']

        # Fixed orbital parameters
        omega = self.orbital_input_params['RAA difference']  # Right ascension of ascending node in degrees
        w = 0  # Argument of perigee in degrees

        # Link budget inputs
        p_tx_dbm = self.link_budget_input_params['p_tx_dbm']  # W = 10 ** ((P(dBm)-30)/10)
        p_tx = 10 ** (p_tx_dbm / 10)
        fc = self.link_budget_input_params['center_frequency']

        replace = self.orbital_input_params['replace_math']
        other_orbit_only = self.orbital_input_params['other_orbit_only']
        # ------------------

        # Object Simulation creation, with the corresponding results folder
        t_start = datetime(2022, 1, 1)
        sim = Simulation(t_start=t_start)
        np.random.seed(0)

        # sim params on a .txt at the simulation results folder
        with open(os.path.join(sim.results_folder, 'sim_params.txt'), 'w') as file:
            file.write('SHIFTED ORBITS MATH ANALYSIS\n')
            file.write('\nCommon Input parameters\n')
            for k, v in self.link_budget_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
            file.write('\nShifted orbit Input parameters\n')
            for k, v in self.orbital_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
        file.close()

        def generate_results():
            # Results dataframe creation
            results_df = pd.DataFrame(list(product(alpha_list, n_sats_list, h_list, inclination_list)),
                                      columns=['alpha', 'n_sats', 'h', 'inclination'])

            # Compute angular offset and time vector for each number of satellites and altitudes, respectively
            betas = []
            ts = []
            for _ in alpha_list:
                for n_sats in n_sats_list:
                    max_angular_offset = 2 * np.pi / n_sats
                    for h in h_list:
                        for _ in inclination_list:
                            beta_vec = np.linspace(0, max_angular_offset, self.orbital_input_params['beta_steps_math'])
                            betas.append(beta_vec)

                            T = ad.period_from_semi_major_axis(Re + h)
                            time_steps = [sim.t_start + 2 * timedelta(seconds=T) / n_timesteps * i for i in
                                          range(n_timesteps)]
                            for _ in beta_vec:
                                ts.append(time_steps)
            results_df['beta'] = betas
            results_df = results_df.explode('beta', ignore_index=True)
            results_df['time'] = ts
            results_df = results_df.explode('time', ignore_index=True)

            def compute_row_results(row):
                alpha = row['alpha']
                n_sats = row['n_sats']
                h = row['h']
                beta = row['beta']
                t = row['time']
                inc = row['inclination']
                results = []
                p_i_mw = 0
                n_i = 0

                alpha_rad = np.deg2rad(alpha)
                g1 = 2 / (1 - np.cos(np.deg2rad(alpha) / 2))  # Gain in the direction of the main beam
                g1_db = 10 * np.log10(g1)
                half_cone_angle = alpha / 2

                T = ad.period_from_semi_major_axis(Re + h)
                progress = (t - t_start).total_seconds() / T

                true_anomaly_rx = 0 + 2 * np.pi * progress
                true_anomaly_tx = true_anomaly_rx + 2 * np.pi / n_sats

                r0_rx = np.array([(Re + h) * np.cos(true_anomaly_rx), (Re + h) * np.sin(true_anomaly_rx), 0])
                r0_tx = np.array([(Re + h) * np.cos(true_anomaly_tx), (Re + h) * np.sin(true_anomaly_tx), 0])

                M1 = ad.orbital_to_GEC_transformation_matrix(0, w, inc)
                M2 = ad.orbital_to_GEC_transformation_matrix(omega, w, inc)

                r_rx = np.matmul(M1, np.transpose(r0_rx))
                r_tx = np.matmul(M1, np.transpose(r0_tx))
                r_rx_tx = r_tx - r_rx
                r_rx_tx_norm = np.linalg.norm(r_rx_tx)

                d_i_min = None

                # If we consider same orbit as interference too
                if not other_orbit_only:
                    cond1 = n_sats / np.pi * np.arccos(Re / (Re + h))
                    cond2 = 1 + n_sats / (2 * np.pi) * alpha_rad
                    i_max = int(min(cond1, cond2))
                    n_i = i_max - 1

                    lambda_fc = c.SPEED_OF_LIGHT / fc
                    num = lambda_fc ** 2 * p_tx

                    for i in range(2, i_max + 1):
                        den = 8 * (np.pi ** 2) * (1 - np.cos(alpha_rad / 2)) ** 2 * (Re + h) ** 2 * (
                                1 - np.cos(2 * np.pi * i / n_sats))

                        p_i_mw += num / den
                        d_i = np.sqrt(2 * (Re + h) ** 2 * (1 - np.cos(2 * np.pi * i / n_sats)))

                        # Distance to nearest interferer
                        if not d_i_min:
                            d_i_min = d_i
                        elif d_i < d_i_min:
                            d_i_min = d_i

                # Shifted orbit interference
                for j in range(n_sats):
                    true_anomaly_j_rx = (beta + 2 * np.pi * progress + 2 * np.pi * j / n_sats) % (
                            2 * np.pi)  # Where the interferer is pointing
                    true_anomaly_j1_tx = (beta + 2 * np.pi * progress + 2 * np.pi * ((j + 1) % n_sats) / n_sats) % (
                            2 * np.pi)  # interferer

                    r0_jrx = np.array([(Re + h) * np.cos(true_anomaly_j_rx), (Re + h) * np.sin(true_anomaly_j_rx), 0])
                    r0_j1tx = np.array(
                        [(Re + h) * np.cos(true_anomaly_j1_tx), (Re + h) * np.sin(true_anomaly_j1_tx), 0])

                    r_jrx = np.matmul(M2, np.transpose(r0_jrx))
                    r_j1tx = np.matmul(M2, np.transpose(r0_j1tx))

                    r_rx_j1tx = r_j1tx - r_rx

                    r_j1tx_jrx = r_jrx - r_j1tx
                    r_j1tx_rx = -r_rx_j1tx

                    # From receiver perspective
                    dot_prod = np.dot(r_rx_tx, r_rx_j1tx)
                    r_rx_j1tx_norm = np.linalg.norm(r_rx_j1tx)
                    theta_j = np.rad2deg(
                        np.arccos(dot_prod / (r_rx_tx_norm * r_rx_j1tx_norm)))  # Angle Interferer-Rx-Tx

                    # From interferer perspective
                    dot_prod = np.dot(r_j1tx_jrx, r_j1tx_rx)
                    d_i = r_rx_j1tx_norm
                    theta_j1 = np.rad2deg(
                        np.arccos(dot_prod / (np.linalg.norm(r_j1tx_jrx) * d_i)))  # Angle Interferer+1-Interferer-Rx

                    # Check no blockage from the Earth
                    dot_prod = np.dot(r_rx, r_jrx)
                    V = (Re ** 2 * (((Re + h) ** 2 + (Re + h) ** 2) - 2 * (dot_prod))) - (
                            ((Re + h) ** 2) * ((Re + h) ** 2)) + (dot_prod) ** 2

                    if V <= 0 and abs(theta_j) <= half_cone_angle and abs(theta_j1) <= half_cone_angle:
                        # Distance to nearest interferer
                        if not d_i_min:
                            d_i_min = d_i
                        elif d_i < d_i_min:
                            d_i_min = d_i
                        # Interference linear
                        p_i_dbm = rx_power(p_tx_dbm, d_i, fc, l_abs=0, g_tx=g1_db)
                        p_i_mw += (10 ** (p_i_dbm / 10))
                        # Number of interferers
                        n_i += 1

                results.append(r_rx_tx_norm)  # Tx-Rx distance
                p_rx_dbm = rx_power(p_tx_dbm, r_rx_tx_norm, fc, l_abs=0, g_tx=g1_db)  # Rx power
                results.append(p_rx_dbm)
                p_rx_linear = 10 ** (p_rx_dbm / 10)
                results.append(p_rx_linear)
                results.append(d_i_min)
                if p_i_mw == 0:
                    results.append(0)
                    results.append(None)
                else:
                    results.append(p_i_mw)
                    results.append(10 * np.log10(p_i_mw))
                results.append(n_i)

                # SIR
                results.append(p_rx_linear / p_i_mw if p_i_mw else None)

                # SNR
                system_t = self.link_budget_input_params['T_system']
                bw = self.link_budget_input_params['bandwidth']
                p_noise_mw = c.BOLTZMAN * system_t * bw * 1e3
                results.append(p_rx_linear / p_noise_mw)

                # SINR
                sinr_linear = p_rx_linear / (p_i_mw + p_noise_mw)
                results.append(sinr_linear)

                # Capacity (Gbps)
                results.append(bw * np.log2(1 + sinr_linear) * 1e-9 if sinr_linear != 0 else None)

                return results

            results_df[self.results_columns] = results_df.progress_apply(compute_row_results, axis=1,
                                                                         result_type='expand')

            return results_df

        # Dataframe cache file format
        df_extension = self.orbital_input_params['math_results_extension'] + '_other_orbit_only_' + str(
            other_orbit_only) \
                       + self.link_budget_input_params['extension']
        df_path = os.path.join(cache_path, df_extension)
        rep = True if not os.path.isfile(df_path) or replace else False

        load_or_recompute_and_cache(df_path, generate_results, replace=rep, verbose=True)

    def shifted_orbits_simulation(self):
        """Simulation analysis of the Shifted orbits scenario

        Returns:
            Dataframe: Results in a Dataframe format. Input columns:
            ['alpha', 'n_sats', 'h', 'inclination', 'beta', 'time']

        """
        # ------Inputs------
        alpha_list = np.arange(self.orbital_input_params['alpha_lims'][0],
                               self.orbital_input_params['alpha_lims'][1],
                               self.orbital_input_params['alpha_resolution_sim'])
        n_sats_list = self.orbital_input_params['n_sats_list']
        h_list = np.arange(self.orbital_input_params['h_lims'][0],
                           self.orbital_input_params['h_lims'][1],
                           self.orbital_input_params['h_resolution_sim'])
        inclination_list = np.arange(self.orbital_input_params['inclination_lims'][0],
                                     self.orbital_input_params['inclination_lims'][1],
                                     self.orbital_input_params['inclination_resolution_sim'])
        n_timesteps = self.orbital_input_params['n_timesteps_sim']

        # Link budget inputs
        p_tx_dbm = self.link_budget_input_params['p_tx_dbm']  # W = 10 ** ((P(dBm)-30)/10)
        p_tx = 10 ** (p_tx_dbm / 10)
        fc = self.link_budget_input_params['center_frequency']

        replace = self.orbital_input_params['replace_sim']
        other_orbit_only = self.orbital_input_params['other_orbit_only']
        # ------------------

        # Object Simulation creation, with the corresponding results folder
        t_start = datetime(2022, 1, 1)
        sim = Simulation(t_start=t_start)
        np.random.seed(0)

        # sim params on a .txt at the simulation results folder
        with open(os.path.join(sim.results_folder, 'sim_params.txt'), 'w') as file:
            file.write('SHIFTED ORBITS MATH ANALYSIS\n')
            file.write('\nCommon Input parameters\n')
            for k, v in self.link_budget_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
            file.write('\nShifted orbit Input parameters\n')
            for k, v in self.orbital_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
        file.close()

        def generate_results():
            # Results dataframe creation
            results_df = pd.DataFrame(list(product(alpha_list, n_sats_list, h_list, inclination_list)),
                                      columns=['alpha', 'n_sats', 'h', 'inclination'])

            # Compute angular offset and time vector for each number of satellites and altitudes, respectively
            betas = []
            ts = []
            for _ in alpha_list:
                for n_sats in n_sats_list:
                    max_angular_offset = 2 * np.pi / n_sats
                    for h in h_list:
                        for _ in inclination_list:
                            beta_vec = np.linspace(0, max_angular_offset, self.orbital_input_params['beta_steps_sim'])
                            betas.append(beta_vec)

                            T = ad.period_from_semi_major_axis(Re + h)
                            time_steps = [sim.t_start + 2 * timedelta(seconds=T) / n_timesteps * i for i in
                                          range(n_timesteps)]
                            for _ in beta_vec:
                                ts.append(time_steps)
            results_df['beta'] = betas
            results_df = results_df.explode('beta', ignore_index=True)
            results_df['time'] = ts
            results_df = results_df.explode('time', ignore_index=True)

            # Fixed orbital parameters
            e = 0.0  # eccentricity
            omega = self.orbital_input_params['RAA difference']  # Right ascension of ascending node in degrees
            w = 0  # Argument of perigee in degrees

            rx_id = 0
            tx_id = 1

            def compute_row_results(row):
                alpha = row['alpha']
                n_sats = row['n_sats']
                h = row['h']
                beta = row['beta']
                t = row['time']
                inc = row['inclination']
                results = []

                alpha_rad = np.deg2rad(alpha)
                g1 = 2 / (1 - np.cos(np.deg2rad(alpha) / 2))  # Gain in the direction of the main beam
                g1_db = 10 * np.log10(g1)
                half_cone_angle = alpha / 2

                # Orbit objects creation
                # Orbit 1
                a = c.EARTH_RADIUS_M + h  # Semimajor axis in meters
                orbit_low = Orbit(sim, e, a, 0, inc, w, initial_anomaly=0)
                orbit_low.add_satellites(n_sats=n_sats)

                # Orbit 2
                orbit_high = Orbit(sim, e, a, omega, inc, w, initial_anomaly=beta)
                orbit_high.add_satellites(n_sats=n_sats)

                # Constellation creation and propagation
                const = Constellation(sim, orbits=[orbit_low, orbit_high])
                const.update_SSPs(time=t)  # Propagate orbit to corresponding timestep

                # Link of interest in the lower orbit
                rx_sat = const.orbits[0].satellites[rx_id]
                tx_sat = const.orbits[0].satellites[tx_id]

                # Plot antenna diagrams

                # Distance
                distance = tk.sat_to_sat_disance(tx_sat.xyz_r, rx_sat.xyz_r)
                results.append(distance)

                # Rx power
                p_rx_dbm = rx_power(p_tx_dbm, distance, fc, l_abs=0, g_tx=g1_db)
                results.append(p_rx_dbm)
                p_rx_linear = 10 ** (p_rx_dbm / 10)
                results.append(p_rx_linear)

                # Interference
                p_i_mw = 0
                n_i = 0
                d_i_min = None

                if other_orbit_only:
                    orbits = const.orbits[1:]
                else:
                    orbits = const.orbits

                for o, orbit in enumerate(orbits):
                    for i, sat in enumerate(orbit.satellites):
                        # Link of interest is not considered interference
                        if orbit == const.orbits[0] and (i == rx_id or (i + 1) % len(orbit.satellites) == rx_id):
                            continue
                        elif i < len(orbit.satellites):
                            i_rx_sat = orbit.satellites[i]
                            i_tx_sat = orbit.satellites[(i + 1) % len(orbit.satellites)]
                            if tk.check_interference(rx_sat, tx_sat, i_rx_sat, i_tx_sat,
                                                     half_cone_angle=half_cone_angle):
                                n_i += 1
                                distance_interferer = tk.sat_to_sat_disance(i_tx_sat.xyz_r, rx_sat.xyz_r)
                                g_i_tx = g1_db
                                p_i_dbm = rx_power(p_tx_dbm, distance_interferer, fc, l_abs=0, g_tx=g_i_tx)
                                p_i_mw += (10 ** (p_i_dbm / 10))
                                if not d_i_min:
                                    d_i_min = distance_interferer
                                elif distance_interferer < d_i_min:
                                    d_i_min = distance_interferer
                results.append(d_i_min)
                results.append(p_i_mw)
                if p_i_mw == 0:
                    results.append(None)
                else:
                    results.append(10 * np.log10(p_i_mw))
                results.append(n_i)

                '''
                path = os.path.join(sim.figures_folder, 'const_plot_inc{:.2f}.jpg'.format(inc))
                if not os.path.isfile(path):
                    const.plot_constellation(filename='const_plot_inc{:.2f}'.format(inc),
                                             show=False, orbits=True, color_by_orbit=True, annotate=False)
                '''

                # SIR
                results.append(p_rx_linear / p_i_mw if p_i_mw else None)

                # SNR
                system_t = self.link_budget_input_params['T_system']
                bw = self.link_budget_input_params['bandwidth']
                p_noise_mw = c.BOLTZMAN * system_t * bw * 1e3
                results.append(p_rx_linear / p_noise_mw)

                # SINR
                sinr_linear = p_rx_linear / (p_i_mw + p_noise_mw)
                results.append(sinr_linear)

                # Capacity (Gbps)
                results.append(bw * np.log2(1 + sinr_linear) * 1e-9 if sinr_linear != 0 else None)

                return results

            results_df[self.results_columns] = results_df.progress_apply(compute_row_results, axis=1,
                                                                         result_type='expand')

            return results_df

        # Dataframe cache file format
        df_extension = self.orbital_input_params['sim_results_extension'] + '_other_orbit_only_' + str(other_orbit_only) \
                       + self.link_budget_input_params['extension']
        df_path = os.path.join(cache_path, df_extension)
        rep = True if not os.path.isfile(df_path) or replace else False

        load_or_recompute_and_cache(df_path, generate_results, replace=rep, verbose=True)

    def complete_constellation_math(self):
        # ------Inputs------
        # Orbital params
        n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                self.orbital_input_params['n_sats_lims'][1],
                                self.orbital_input_params['n_sats_resolution'])
        h_shell = self.orbital_input_params['h']
        h_coplanar = self.orbital_input_params['h'] + self.orbital_input_params['h_diff']
        n_orbits = self.orbital_input_params['n_orbits']
        inclination_list = self.orbital_input_params['inclination']
        w = 0  # Argument of perigee in degrees
        n_timesteps = self.orbital_input_params['n_timesteps_math']
        max_angular_offset = 2 * np.pi / min(n_sats_list)

        replace = self.orbital_input_params['math_replace']

        # Link budget params
        p_tx_dbm = self.link_budget_input_params['p_tx_dbm']  # W = 10 ** ((P(dBm)-30)/10)
        p_tx = 10 ** (p_tx_dbm / 10)
        fc = self.link_budget_input_params['center_frequency']
        alpha_list = self.link_budget_input_params['alpha_list']

        # Object Simulation creation, with the corresponding results folder
        t_start = datetime(2022, 1, 1)
        sim = Simulation(t_start=t_start)
        np.random.seed(0)

        # sim params on a .txt at the simulation results folder
        with open(os.path.join(sim.results_folder, 'sim_params.txt'), 'w') as file:
            file.write('COMPLETE CONSTELLATION MATH ANALYSIS\n')
            file.write('\nCommon Input parameters\n')
            for k, v in self.link_budget_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
            file.write('\nShifted orbit Input parameters\n')
            for k, v in self.orbital_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
        file.close()

        def generate_results():
            # Results dataframe creation
            results_df = pd.DataFrame(list(product(alpha_list, n_sats_list, inclination_list)),
                                      columns=['alpha', 'n_sats', 'inclination'])

            # Compute time vector for each number of satellites and alphas, respectively
            ts = []
            for _ in alpha_list:
                for _ in inclination_list:
                    for n_sats in n_sats_list:
                        t1 = ad.period_from_semi_major_axis(Re + h_shell)
                        t2 = ad.period_from_semi_major_axis(Re + h_coplanar)
                        max_angular_offset = 2 * np.pi / n_sats
                        t_beta_max = max_angular_offset / (2 * np.pi * (1 / t1 - 1 / t2))

                        time_steps = [sim.t_start + timedelta(seconds=t_beta_max) / n_timesteps * i for i in
                                      range(n_timesteps)]
                        ts.append(time_steps)
            results_df['time'] = ts
            results_df = results_df.explode('time', ignore_index=True)

            def compute_row_results(row):
                n_sats = row['n_sats']
                alpha = row['alpha']
                t = row['time']
                inclination = row['inclination']
                results = []
                p_i_mw = 0
                p_i_mw_shifted_1_coplanar = 0
                p_i_mw_shifted = 0
                p_i_mw_single = 0

                alpha_rad = np.deg2rad(alpha)
                g1 = 2 / (1 - np.cos(np.deg2rad(alpha) / 2))  # Gain in the direction of the main beam
                g1_db = 10 * np.log10(g1)
                half_cone_angle = alpha / 2

                d_i_min = None

                # Interference from the same orbit
                cond1 = n_sats / np.pi * np.arccos(Re / (Re + h_shell))
                cond2 = 1 + n_sats / (2 * np.pi) * alpha_rad
                i_max = int(min(cond1, cond2))
                n_i = i_max - 1

                lambda_fc = c.SPEED_OF_LIGHT / fc
                num = lambda_fc ** 2 * p_tx

                for i in range(2, i_max + 1):
                    den = 8 * (np.pi ** 2) * (1 - np.cos(alpha_rad / 2)) ** 2 * (Re + h_shell) ** 2 * (
                            1 - np.cos(2 * np.pi * i / n_sats))

                    p_i_mw += num / den
                    d_i = np.sqrt(2 * (Re + h_shell) ** 2 * (1 - np.cos(2 * np.pi * i / n_sats)))

                    # Distance to nearest interferer
                    if not d_i_min:
                        d_i_min = d_i
                    elif d_i < d_i_min:
                        d_i_min = d_i
                p_i_mw_single = p_i_mw

                # Interference from the shifted orbits
                # Here the frame of reference is Earth, i.e. both shifted orbits propagate in time
                # Compute orbit propagation progress based on t
                T = ad.period_from_semi_major_axis(Re + h_shell)
                progress = (t - t_start).total_seconds() / T

                # Compute tx and rx 3D position vectors
                true_anomaly_rx = 0 + 2 * np.pi * progress
                true_anomaly_tx = true_anomaly_rx + 2 * np.pi / n_sats
                r0_rx = np.array(
                    [(Re + h_shell) * np.cos(true_anomaly_rx), (Re + h_shell) * np.sin(true_anomaly_rx), 0])
                r0_tx = np.array(
                    [(Re + h_shell) * np.cos(true_anomaly_tx), (Re + h_shell) * np.sin(true_anomaly_tx), 0])
                # Self orbit transformation matrix (omega=0, no shift)
                M1 = ad.orbital_to_GEC_transformation_matrix(0, w, inclination)
                r_rx = np.matmul(M1, np.transpose(r0_rx))
                r_tx = np.matmul(M1, np.transpose(r0_tx))
                r_rx_tx = r_tx - r_rx
                r_rx_tx_norm = np.linalg.norm(r_rx_tx)

                for i in range(1, n_orbits):
                    # Shifted orbits orbital transformation matrix
                    omega = i * 360 / n_orbits  # RAAN in degrees
                    M2 = ad.orbital_to_GEC_transformation_matrix(omega, w, inclination)
                    for j in range(n_sats):
                        true_anomaly_j_rx = (2 * np.pi * progress + 2 * np.pi * j / n_sats) % (
                                2 * np.pi)  # Where the interferer is pointing
                        true_anomaly_j1_tx = (2 * np.pi * progress + 2 * np.pi * ((j + 1) % n_sats) / n_sats) % (
                                2 * np.pi)  # interferer

                        r0_jrx = np.array(
                            [(Re + h_shell) * np.cos(true_anomaly_j_rx), (Re + h_shell) * np.sin(true_anomaly_j_rx), 0])
                        r0_j1tx = np.array(
                            [(Re + h_shell) * np.cos(true_anomaly_j1_tx), (Re + h_shell) * np.sin(true_anomaly_j1_tx),
                             0])

                        r_jrx = np.matmul(M2, np.transpose(r0_jrx))
                        r_j1tx = np.matmul(M2, np.transpose(r0_j1tx))

                        r_rx_j1tx = r_j1tx - r_rx

                        r_j1tx_jrx = r_jrx - r_j1tx
                        r_j1tx_rx = -r_rx_j1tx

                        # From receiver perspective
                        dot_prod = np.dot(r_rx_tx, r_rx_j1tx)
                        r_rx_j1tx_norm = np.linalg.norm(r_rx_j1tx)
                        theta_j = np.rad2deg(
                            np.arccos(dot_prod / (r_rx_tx_norm * r_rx_j1tx_norm)))  # Angle Interferer-Rx-Tx

                        # From interferer perspective
                        dot_prod = np.dot(r_j1tx_jrx, r_j1tx_rx)
                        d_i = r_rx_j1tx_norm
                        theta_j1 = np.rad2deg(
                            np.arccos(
                                dot_prod / (np.linalg.norm(r_j1tx_jrx) * d_i)))  # Angle Interferer+1-Interferer-Rx

                        # Check no blockage from the Earth
                        dot_prod = np.dot(r_rx, r_jrx)
                        V = (Re ** 2 * (((Re + h_shell) ** 2 + (Re + h_shell) ** 2) - 2 * (dot_prod))) - (
                                ((Re + h_shell) ** 2) * ((Re + h_shell) ** 2)) + (dot_prod) ** 2

                        if V <= 0 and abs(theta_j) <= half_cone_angle and abs(theta_j1) <= half_cone_angle:
                            # Distance to nearest interferer
                            if not d_i_min:
                                d_i_min = d_i
                            elif d_i < d_i_min:
                                d_i_min = d_i
                            # Interference linear
                            p_i_dbm = rx_power(p_tx_dbm, d_i, fc, l_abs=0, g_tx=g1_db)
                            p_i_mw += (10 ** (p_i_dbm / 10))
                            # Number of interferers
                            n_i += 1
                p_i_mw_only_shifted = p_i_mw

                # Interference from the co-planar orbit
                # Here the frame of reference is the orbit of inerest, i.e. the orbit of interest is static to te viewer
                # Compute orbit propagation progress based on t
                T1 = ad.period_from_semi_major_axis(Re + h_shell)
                T2 = ad.period_from_semi_major_axis(Re + h_coplanar)

                T_beta_max = max_angular_offset / (2 * np.pi * (1 / T1 - 1 / T2))
                progress = (t - t_start).total_seconds() / T_beta_max
                beta = progress * max_angular_offset

                A_angle = np.pi / 2 - np.pi / n_sats
                AC_unit = np.array([np.sin(A_angle), -np.cos(A_angle)])
                for j in range(n_sats):
                    ############ Psi j ############
                    A_vec = np.array([0, Re + h_shell])

                    O_angle = -beta + 2 * np.pi * j / n_sats

                    B_vec = (Re + h_coplanar) * np.array([np.sin(O_angle), np.cos(O_angle)])
                    AB_vec = B_vec - A_vec

                    # Dot product
                    dot = AC_unit @ AB_vec
                    # Norm
                    AB_norm = np.linalg.norm(AB_vec)
                    # Arc cosine
                    # psi_j = np.arccos(dot / AB_norm)  # WRONG psi_j has to be signed!
                    cross = np.cross(AC_unit, AB_vec)
                    psi_j = np.arctan2(cross, dot)
                    psi_j_deg = psi_j * 180 / np.pi

                    ############ Psi j interferer ############
                    O_angle_t = O_angle - 2 * np.pi / n_sats
                    D_vec = (Re + h_coplanar) * np.array([np.sin(O_angle_t), np.cos(O_angle_t)])
                    BD_vec = D_vec - B_vec
                    BA_vec = A_vec - B_vec

                    # Dot product
                    dot = BD_vec @ BA_vec
                    # Norm
                    BD_norm = np.linalg.norm(BD_vec)
                    BA_norm = np.linalg.norm(BA_vec)
                    # Arc cosine
                    # psi_j_i = np.arccos(dot / (BD_norm * BA_norm))  # WRONG psi_j_i has to be signed!
                    cross = np.cross(BD_vec, BA_vec)
                    psi_j_i = np.arctan2(cross, dot)
                    psi_j_i_deg = psi_j_i * 180 / np.pi

                    # Visibility with interferer (not blocked by Earth)
                    j_visible = psi_j > np.pi / n_sats - np.arccos(Re / (Re + h_shell))

                    d_i = np.linalg.norm(AB_vec)

                    if j_visible and abs(psi_j_deg) <= half_cone_angle and abs(psi_j_i_deg) <= half_cone_angle:
                        # Distance to nearest interferer
                        if not d_i_min:
                            d_i_min = d_i
                        elif d_i < d_i_min:
                            d_i_min = d_i
                        # Number of interferers
                        n_i += 1
                        # Interference linear
                        p_i_dbm = rx_power(p_tx_dbm, d_i, fc, l_abs=0, g_tx=g1_db)
                        p_i_mw += (10 ** (p_i_dbm / 10))
                p_i_mw_shifted_1_coplanar = p_i_mw

                # Interference from multiple co-planar orbits
                # Here the frame of reference is Earth, i.e. both shifted orbits propagate in time
                # Compute orbit propagation progress based on t
                T = ad.period_from_semi_major_axis(Re + h_coplanar)
                progress = (t - t_start).total_seconds() / T
                for i in range(1, n_orbits):
                    # Shifted orbits orbital transformation matrix
                    omega = i * 360 / n_orbits  # RAAN in degrees
                    M2 = ad.orbital_to_GEC_transformation_matrix(omega, w, inclination)
                    for j in range(n_sats):
                        true_anomaly_j_rx = (2 * np.pi * progress + 2 * np.pi * j / n_sats) % (
                                2 * np.pi)  # Where the interferer is pointing
                        true_anomaly_j1_tx = (2 * np.pi * progress + 2 * np.pi * ((j + 1) % n_sats) / n_sats) % (
                                2 * np.pi)  # interferer

                        r0_jrx = np.array([(Re + h_coplanar) * np.cos(true_anomaly_j_rx),
                                           (Re + h_coplanar) * np.sin(true_anomaly_j_rx), 0])
                        r0_j1tx = np.array(
                            [(Re + h_coplanar) * np.cos(true_anomaly_j1_tx),
                             (Re + h_coplanar) * np.sin(true_anomaly_j1_tx), 0])

                        r_jrx = np.matmul(M2, np.transpose(r0_jrx))
                        r_j1tx = np.matmul(M2, np.transpose(r0_j1tx))

                        r_rx_j1tx = r_j1tx - r_rx

                        r_j1tx_jrx = r_jrx - r_j1tx
                        r_j1tx_rx = -r_rx_j1tx

                        # From receiver perspective
                        dot_prod = np.dot(r_rx_tx, r_rx_j1tx)
                        r_rx_j1tx_norm = np.linalg.norm(r_rx_j1tx)
                        theta_j = np.rad2deg(
                            np.arccos(dot_prod / (r_rx_tx_norm * r_rx_j1tx_norm)))  # Angle Interferer-Rx-Tx

                        # From interferer perspective
                        dot_prod = np.dot(r_j1tx_jrx, r_j1tx_rx)
                        d_i = r_rx_j1tx_norm
                        theta_j1 = np.rad2deg(
                            np.arccos(
                                dot_prod / (np.linalg.norm(r_j1tx_jrx) * d_i)))  # Angle Interferer+1-Interferer-Rx

                        # Check no blockage from the Earth
                        dot_prod = np.dot(r_rx, r_jrx)
                        V = (Re ** 2 * (((Re + h_shell) ** 2 + (Re + h_coplanar) ** 2) - 2 * (dot_prod))) - (
                                ((Re + h_shell) ** 2) * ((Re + h_coplanar) ** 2)) + (dot_prod) ** 2

                        if V <= 0 and abs(theta_j) <= half_cone_angle and abs(theta_j1) <= half_cone_angle:
                            # Distance to nearest interferer
                            if not d_i_min:
                                d_i_min = d_i
                            elif d_i < d_i_min:
                                d_i_min = d_i
                            # Interference linear
                            p_i_dbm = rx_power(p_tx_dbm, d_i, fc, l_abs=0, g_tx=g1_db)
                            p_i_mw += (10 ** (p_i_dbm / 10))
                            # Number of interferers
                            n_i += 1

                results.append(r_rx_tx_norm)  # Tx-Rx distance
                p_rx_dbm = rx_power(p_tx_dbm, r_rx_tx_norm, fc, l_abs=0, g_tx=g1_db)  # Rx power
                results.append(p_rx_dbm)
                p_rx_linear = 10 ** (p_rx_dbm / 10)
                results.append(p_rx_linear)
                results.append(d_i_min)
                if p_i_mw == 0:
                    results.append(0)
                    results.append(None)
                else:
                    results.append(p_i_mw)
                    results.append(10 * np.log10(p_i_mw))
                results.append(n_i)

                # SIR
                results.append(p_rx_linear / p_i_mw if p_i_mw else None)

                # SNR
                system_t = self.link_budget_input_params['T_system']
                bw = self.link_budget_input_params['bandwidth']
                p_noise_mw = c.BOLTZMAN * system_t * bw * 1e3
                results.append(p_rx_linear / p_noise_mw)

                # SINR
                sinr_linear = p_rx_linear / (p_i_mw + p_noise_mw)
                results.append(sinr_linear)

                # Capacity (Gbps)
                results.append(bw * np.log2(1 + sinr_linear) * 1e-9 if sinr_linear != 0 else None)

                # Intermediate capacities
                sinr_linear = p_rx_linear / (p_i_mw_shifted_1_coplanar + p_noise_mw)
                results.append(bw * np.log2(1 + sinr_linear) * 1e-9 if sinr_linear != 0 else None)

                sinr_linear = p_rx_linear / (p_i_mw_only_shifted + p_noise_mw)
                results.append(bw * np.log2(1 + sinr_linear) * 1e-9 if sinr_linear != 0 else None)

                sinr_linear = p_rx_linear / (p_i_mw_single + p_noise_mw)
                results.append(bw * np.log2(1 + sinr_linear) * 1e-9 if sinr_linear != 0 else None)

                return results

            results_df[self.results_columns] = results_df.progress_apply(compute_row_results, axis=1,
                                                                         result_type='expand')
            return results_df

        # Dataframe cache file format
        df_extension = self.orbital_input_params['math_results_extension'] + self.link_budget_input_params['extension']
        df_path = os.path.join(cache_path, df_extension)
        rep = True if not os.path.isfile(df_path) or replace else False

        load_or_recompute_and_cache(df_path, generate_results, replace=rep, verbose=True)

    ####### Misc. #########
    def coplanar_higher_orbits_math(self):
        """Mathematical analysis of the Co-planar orbits scenario from the higher orbit perspective

        Returns:
            Dataframe: Results in a Dataframe format. Input columns: ['alpha', 'n_sats', 'h_high', 'time']

        """
        # ------Inputs------
        alpha_list = self.link_budget_input_params['alpha_list']
        n_sats_list = self.orbital_input_params['n_sats_list']
        # Altitude vector for the upper orbit
        h_high_list = np.arange(self.orbital_input_params['h_lims'][0],
                                self.orbital_input_params['h_lims'][1],
                                self.orbital_input_params['h_resolution_math'])
        h_low = self.orbital_input_params['h_low']

        beta_max = 2 * np.pi * 1 / min(n_sats_list)
        beta_steps = self.orbital_input_params['beta_steps_math']
        betas = np.linspace(0, beta_max, beta_steps)

        # Link budget inputs
        p_tx_dbm = self.link_budget_input_params['p_tx_dbm']  # W = 10 ** ((P(dBm)-30)/10)
        p_tx = 10 ** (p_tx_dbm / 10)
        fc = self.link_budget_input_params['center_frequency']

        replace = self.orbital_input_params['math_replace']
        higher_orbit_only = self.orbital_input_params['higher_orbit_only']
        # ------------------

        # Object Simulation creation, with the corresponding results folder
        t_start = datetime(2022, 1, 1)
        sim = Simulation(t_start=t_start)
        np.random.seed(0)

        # sim params on a .txt at the simulation results folder
        with open(os.path.join(sim.results_folder, 'sim_params.txt'), 'w') as file:
            file.write('COPLANAR ORBITS HIGHER ORBIT MATH ANALYSIS\n')
            file.write('\nCommon Input parameters\n')
            for k, v in self.link_budget_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
            file.write('\nCoplanar orbit Input parameters\n')
            for k, v in self.orbital_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
        file.close()

        def generate_results():
            # Results dataframe creation
            results_df = pd.DataFrame(list(product(alpha_list, n_sats_list, h_high_list)),
                                      columns=['alpha', 'n_sats', 'h_high'])

            # Compute angular offset and time vector for each number of satellites and altitudes, respectively
            ts = []
            for _ in n_sats_list:
                for _ in alpha_list:
                    for h_high in h_high_list:
                        t1 = ad.period_from_semi_major_axis(Re + h_low)
                        # T2 = ad.period_from_semi_major_axis(Re + h_high)
                        # We simulate the same interval of time for all h of the upper orbit
                        # The simulated time corresponds to the largest period of the interference pattern, i.e.,
                        # when the two orbits are the closest together (smaller relative angular speed)
                        t2 = ad.period_from_semi_major_axis(Re + min(h_high_list))

                        t_beta_max = beta_max / (2 * np.pi * (1 / t1 - 1 / t2))

                        time_steps = [sim.t_start + timedelta(seconds=t_beta_max) / beta_steps * i for i in
                                      range(beta_steps)]
                        ts.append(time_steps)
            results_df['time'] = ts
            results_df = results_df.explode('time', ignore_index=True)

            def compute_row_results(row):
                n_sats = row['n_sats']
                alpha = row['alpha']
                h_high = row['h_high']
                t = row['time']
                results = []
                p_i_mw = 0
                n_i = 0

                alpha_rad = np.deg2rad(alpha)

                T1 = ad.period_from_semi_major_axis(Re + h_low)
                T2 = ad.period_from_semi_major_axis(Re + h_high)

                T_beta_max = beta_max / (2 * np.pi * (1 / T1 - 1 / T2))
                beta = ((t - t_start).total_seconds() / T_beta_max) * beta_max

                d_i_min = None

                g1 = 2 / (1 - np.cos(np.deg2rad(alpha) / 2))  # Gain in the direction of the main beam
                g1_db = 10 * np.log10(g1)
                half_cone_angle = alpha / 2

                # If we consider same orbit as interference too
                if not higher_orbit_only:
                    cond1 = n_sats / np.pi * np.arccos(Re / (Re + h_high))
                    cond2 = 1 + n_sats / (2 * np.pi) * alpha_rad
                    i_max = int(min(cond1, cond2))
                    n_i = i_max - 1

                    lambda_fc = c.SPEED_OF_LIGHT / fc
                    num = lambda_fc ** 2 * p_tx

                    for i in range(2, i_max + 1):
                        den = 8 * (np.pi ** 2) * (1 - np.cos(alpha_rad / 2)) ** 2 * (Re + h_low) ** 2 * (
                                1 - np.cos(2 * np.pi * i / n_sats))

                        p_i_mw += num / den
                        d_i = np.sqrt(2 * (Re + h_low) ** 2 * (1 - np.cos(2 * np.pi * i / n_sats)))

                        # Distance to nearest interferer
                        if not d_i_min:
                            d_i_min = d_i
                        elif d_i < d_i_min:
                            d_i_min = d_i

                # Lower orbit interference
                for i in range(n_sats):
                    ############ Xi i ############
                    D_vec = np.array([0, Re + h_high])

                    O_angle = beta + 2 * np.pi * i / n_sats

                    C_vec = (Re + h_low) * np.array([np.sin(O_angle), np.cos(O_angle)])
                    DC_vec = C_vec - D_vec

                    D_angle = np.pi / 2 - np.pi / n_sats
                    DB_unit = np.array([np.sin(D_angle), -np.cos(D_angle)])

                    # Dot product
                    dot = DB_unit @ DC_vec
                    # Norm
                    DC_norm = np.linalg.norm(DC_vec)
                    # Arc cosine
                    # psi_i = np.arccos(dot / DC_norm)  # WRONG xi_i has to be signed!
                    num_arctan = np.cross(DB_unit, DC_vec)
                    xi_i = np.arctan2(num_arctan, dot)
                    xi_i_deg = xi_i * 180 / np.pi

                    ############ Xi i interferer ############
                    O_angle_t = O_angle - 2 * np.pi / n_sats
                    A_vec = (Re + h_low) * np.array([np.sin(O_angle_t), np.cos(O_angle_t)])
                    CA_vec = A_vec - C_vec
                    CD_vec = D_vec - C_vec

                    # Dot product
                    dot = CA_vec @ CD_vec
                    # Norm
                    CA_norm = np.linalg.norm(CA_vec)
                    CD_norm = np.linalg.norm(CD_vec)
                    # Arc cosine
                    # xi_i_t = np.arccos(dot / (CA_norm * CD_norm))  # WRONG xi_i_t has to be signed!
                    num_arctan = np.cross(CA_vec, CD_vec)
                    xi_i_j = np.arctan2(num_arctan, dot)
                    xi_i_j_deg = xi_i_j * 180 / np.pi

                    # Visibility with interferer (not blocked by Earth
                    i_visible = xi_i > np.pi / n_sats - np.arccos(Re / (Re + h_high))

                    d_i = np.linalg.norm(CD_vec)

                    if i_visible and abs(xi_i_deg) <= half_cone_angle and abs(xi_i_j_deg) <= half_cone_angle:
                        # Distance to nearest interferer
                        if not d_i_min:
                            d_i_min = d_i
                        elif d_i < d_i_min:
                            d_i_min = d_i
                        # Number of interferers
                        n_i += 1
                        # Interference linear
                        p_i_dbm = rx_power(p_tx_dbm, d_i, fc, l_abs=0, g_tx=g1_db)
                        p_i_mw += (10 ** (p_i_dbm / 10))

                d_tx_rx = np.sqrt(2 * (Re + h_high) ** 2 * (1 - np.cos(2 * np.pi / n_sats)))
                results.append(d_tx_rx)  # Tx-Rx distance
                p_rx_dbm = rx_power(p_tx_dbm, d_tx_rx, fc, l_abs=0, g_tx=g1_db)  # Rx power
                results.append(p_rx_dbm)
                p_rx_linear = 10 ** (p_rx_dbm / 10)
                results.append(p_rx_linear)
                results.append(d_i_min)
                results.append(p_i_mw)
                if p_i_mw == 0:
                    results.append(None)
                else:
                    results.append(10 * np.log10(p_i_mw))
                results.append(n_i)

                # SIR
                results.append(p_rx_linear / p_i_mw if p_i_mw else None)

                # SNR
                system_t = self.link_budget_input_params['T_system']
                bw = self.link_budget_input_params['bandwidth']
                p_noise_mw = c.BOLTZMAN * system_t * bw * 1e3
                results.append(p_rx_linear / p_noise_mw)

                # SINR
                sinr_linear = p_rx_linear / (p_i_mw + p_noise_mw)
                results.append(sinr_linear)

                # Capacity (Gbps)
                results.append(bw * np.log2(1 + sinr_linear) * 1e-9 if sinr_linear != 0 else None)

                return results

            results_df[self.results_columns] = results_df.progress_apply(compute_row_results, axis=1,
                                                                         result_type='expand')

            return results_df

        # Dataframe cache file format
        df_extension = self.orbital_input_params['math_results_higher_extension'] + '_higher_orbit_only_' + str(
            higher_orbit_only) \
                       + self.link_budget_input_params['extension']
        df_path = os.path.join(cache_path, df_extension)
        rep = True if not os.path.isfile(df_path) or replace else False

        load_or_recompute_and_cache(df_path, generate_results, replace=rep, verbose=True)

    def coplanar_higher_orbits_simulation(self):
        """Simulation analysis of the Co-planar orbits scenario from the higher orbit

        Returns:
            Dataframe: Results in a Dataframe format. Input columns: ['alpha', 'n_sats', 'h_high', 'time']

        """
        # ------Inputs------
        alpha_list = self.link_budget_input_params['alpha_list']
        n_sats_list = self.orbital_input_params['n_sats_list']
        # Altitude vector for the upper orbit
        h_high_list = np.arange(self.orbital_input_params['h_lims'][0],
                                self.orbital_input_params['h_lims'][1],
                                self.orbital_input_params['h_resolution_sim'])
        h_low = self.orbital_input_params['h_low']

        beta_max = 2 * np.pi * 1 / min(n_sats_list)
        beta_steps = self.orbital_input_params['beta_steps_sim']

        # Link budget inputs
        p_tx_dbm = self.link_budget_input_params['p_tx_dbm']  # W = 10 ** ((P(dBm)-30)/10)
        fc = self.link_budget_input_params['center_frequency']

        replace = self.orbital_input_params['sim_replace']
        higher_orbit_only = self.orbital_input_params['higher_orbit_only']
        # ------------------

        # Object Simulation creation, with the corresponding results folder
        t_start = datetime(2022, 1, 1)
        sim = Simulation(t_start=t_start)
        np.random.seed(0)

        # sim params on a .txt at the simulation results folder
        with open(os.path.join(sim.results_folder, 'sim_params.txt'), 'w') as file:
            file.write('COPLANAR ORBITS HIGHER ORBIT MATH ANALYSIS\n')
            file.write('\nCommon Input parameters\n')
            for k, v in self.link_budget_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
            file.write('\nCoplanar orbit Input parameters\n')
            for k, v in self.orbital_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
        file.close()

        def generate_results():
            # Results dataframe creation
            results_df = pd.DataFrame(list(product(alpha_list, n_sats_list, h_high_list)),
                                      columns=['alpha', 'n_sats', 'h_high'])

            # Compute angular offset and time vector for each number of satellites and altitudes, respectively
            ts = []
            for _ in n_sats_list:
                for _ in alpha_list:
                    for h_high in h_high_list:
                        T1 = ad.period_from_semi_major_axis(Re + h_low)
                        # T2 = ad.period_from_semi_major_axis(Re + h_high)
                        T2 = ad.period_from_semi_major_axis(Re + min(h_high_list))

                        T_beta_max = beta_max / (2 * np.pi * (1 / T1 - 1 / T2))

                        time_steps = [sim.t_start + timedelta(seconds=T_beta_max) / beta_steps * i for i in
                                      range(beta_steps)]
                        ts.append(time_steps)
            results_df['time'] = ts
            results_df = results_df.explode('time', ignore_index=True)

            # Fixed orbital parameters
            e = 0.0  # eccentricity
            omega = 0  # Right ascension of ascending node in degrees
            inc = 0  # Inclination in degrees
            w = 0  # Argument of perigee in degrees

            rx_id = 0
            tx_id = 1

            def compute_row_results(row):
                n_sats = row['n_sats']
                h_high = row['h_high']
                alpha = row['alpha']
                t = row['time']
                results = []

                g1 = 2 / (1 - np.cos(np.deg2rad(alpha) / 2))  # Gain in the direction of the main beam
                g1_db = 10 * np.log10(g1)
                half_cone_angle = alpha / 2

                d_i_min = None

                # Orbit objects creation
                # Lower orbit
                a = Re + h_low  # Semimajor axis in meters
                orbit_low = Orbit(sim, e, a, omega, inc, w, initial_anomaly=0)
                orbit_low.add_satellites(n_sats=n_sats)

                # Higher orbit
                a = Re + h_high  # Semimajor axis in meters
                orbit_high = Orbit(sim, e, a, omega, inc, w, initial_anomaly=0)
                orbit_high.add_satellites(n_sats=n_sats)

                # Constellation creation and propagation
                const = Constellation(sim, orbits=[orbit_low, orbit_high])
                const.update_SSPs(time=t)  # Propagate orbit to corresponding timestep

                # Link of interest in the higher orbit
                rx_sat = const.orbits[1].satellites[rx_id]
                tx_sat = const.orbits[1].satellites[tx_id]

                # Plot antenna diagrams

                # Distance
                distance = tk.sat_to_sat_disance(tx_sat.xyz_r, rx_sat.xyz_r)
                results.append(distance)

                # Rx power
                p_rx_dbm = rx_power(p_tx_dbm, distance, fc, l_abs=0, g_tx=g1_db)
                results.append(p_rx_dbm)
                p_rx_linear = 10 ** (p_rx_dbm / 10)
                results.append(p_rx_linear)

                # Interference
                p_i_mw = 0
                n_i = 0

                if higher_orbit_only:
                    orbits = const.orbits[1:]
                else:
                    orbits = const.orbits

                for o, orbit in enumerate(orbits):
                    for i, sat in enumerate(orbit.satellites):
                        if orbit == const.orbits[1] and (i == rx_id or (i + 1) % len(
                                orbit.satellites) == rx_id):  # Link of interest is not considered interference
                            continue
                        i_rx_sat = orbit.satellites[i]
                        i_tx_sat = orbit.satellites[(i + 1) % len(orbit.satellites)]
                        if tk.check_interference(rx_sat, tx_sat, i_rx_sat, i_tx_sat,
                                                 half_cone_angle=half_cone_angle):
                            n_i += 1
                            distance_interferer = tk.sat_to_sat_disance(i_tx_sat.xyz_r, rx_sat.xyz_r)
                            g_i_tx = g1_db
                            p_i_dbm = rx_power(p_tx_dbm, distance_interferer, fc, l_abs=0, g_tx=g_i_tx)
                            p_i_mw += (10 ** (p_i_dbm / 10))
                            # Distance to nearest interferer
                            if not d_i_min:
                                d_i_min = distance_interferer
                            elif distance_interferer < d_i_min:
                                d_i_min = distance_interferer
                results.append(d_i_min)
                results.append(p_i_mw)
                if p_i_mw == 0:
                    results.append(None)
                else:
                    results.append(10 * np.log10(p_i_mw))
                results.append(n_i)

                # SIR
                results.append(p_rx_linear / p_i_mw if p_i_mw else None)

                # SNR
                system_t = self.link_budget_input_params['T_system']
                bw = self.link_budget_input_params['bandwidth']
                p_noise_mw = c.BOLTZMAN * system_t * bw * 1e3
                results.append(p_rx_linear / p_noise_mw)

                # SINR
                sinr_linear = p_rx_linear / (p_i_mw + p_noise_mw)
                results.append(sinr_linear)

                # Capacity (Gbps)
                results.append(bw * np.log2(1 + sinr_linear) * 1e-9 if sinr_linear != 0 else None)

                return results

            results_df[self.results_columns] = results_df.progress_apply(compute_row_results, axis=1,
                                                                         result_type='expand')

            return results_df

        # Dataframe cache file format
        df_extension = self.orbital_input_params['sim_results_higher_extension'] + '_higher_orbit_only_' + str(
            higher_orbit_only) \
                       + self.link_budget_input_params['extension']
        df_path = os.path.join(cache_path, df_extension)
        rep = True if not os.path.isfile(df_path) or replace else False

        load_or_recompute_and_cache(df_path, generate_results, replace=rep, verbose=True)

    def beamwidth_single_orbit_math(self):
        """Mathematical analysis of the Single orbit scenario

        Returns:
            Dataframe: Results in a Dataframe format. Input columns: ['alpha', 'n_sats', 'h']

        """
        # ------Inputs------
        alpha_list = np.arange(self.orbital_input_params['alpha_lims'][0],
                               self.orbital_input_params['alpha_lims'][1],
                               self.orbital_input_params['alpha_resolution_math'])
        n_sats_list = self.orbital_input_params['n_sats_list']
        h_list = np.arange(self.orbital_input_params['h_lims'][0],
                           self.orbital_input_params['h_lims'][1],
                           self.orbital_input_params[
                               'h_resolution_math'])  # altitude difference between low and upper orbit
        input_columns = ['alpha', 'n_sats', 'h']
        replace = self.orbital_input_params['math_replace']

        # Link budget inputs
        p_tx_dbm = self.link_budget_input_params['p_tx_dbm']  # W = 10 ** ((P(dBm)-30)/10)
        p_tx = 10 ** (p_tx_dbm / 10)
        fc = self.link_budget_input_params['center_frequency']
        bw = self.link_budget_input_params['bandwidth']
        t_system = self.link_budget_input_params['T_system']
        # ------------------

        # Object Simulation creation, with the corresponding results folder
        t_start = datetime(2022, 1, 1)
        sim = Simulation(t_start=t_start)
        np.random.seed(0)

        # sim params on a .txt at the simulation results folder
        with open(os.path.join(sim.results_folder, 'sim_params.txt'), 'w') as file:
            file.write('SINGLE ORBIT MATH ANALYSIS\n')
            file.write('\nCommon Input parameters\n')
            for k, v in self.link_budget_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
            file.write('\nSingle orbit Input parameters\n')
            for k, v in self.orbital_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
        file.close()

        def generate_results():
            # Results dataframe creation
            results_df = pd.DataFrame(list(product(alpha_list, n_sats_list, h_list)), columns=input_columns)

            def compute_row_results(row):
                n_sats = row['n_sats']
                alpha = row['alpha']
                h = row['h']
                results = []
                p_i_mw = 0

                alpha_rad = np.deg2rad(alpha)

                d_i_min = None

                cond1 = n_sats / np.pi * np.arccos(Re / (Re + h))
                cond2 = 1 + n_sats / (2 * np.pi) * alpha_rad
                i_max = int(min(cond1, cond2))
                n_i = i_max - 1

                lambda_fc = c.SPEED_OF_LIGHT / fc
                num = lambda_fc ** 2 * p_tx
                den_tx = 8 * (np.pi ** 2) * (1 - np.cos(alpha_rad / 2)) ** 2 * (Re + h) ** 2 * (
                        1 - np.cos(2 * np.pi / n_sats))

                p_rx_dbm = (10 * np.log10(num / den_tx))
                d_tx_rx = np.sqrt(2 * (Re + h) ** 2 * (1 - np.cos(2 * np.pi / n_sats)))

                for i in range(2, i_max + 1):
                    den = 8 * (np.pi ** 2) * (1 - np.cos(alpha_rad / 2)) ** 2 * (Re + h) ** 2 * (
                            1 - np.cos(2 * np.pi * i / n_sats))

                    p_i_mw += num / den
                    d_i = np.sqrt(2 * (Re + h) ** 2 * (1 - np.cos(2 * np.pi * i / n_sats)))

                    # Distance to nearest interferer
                    if not d_i_min:
                        d_i_min = d_i
                    elif d_i < d_i_min:
                        d_i_min = d_i

                results.append(d_tx_rx)  # Tx-Rx distance
                results.append(p_rx_dbm)
                p_rx_linear = 10 ** (p_rx_dbm / 10)
                results.append(p_rx_linear)
                results.append(d_i_min)
                results.append(p_i_mw)
                if p_i_mw == 0:
                    results.append(None)
                else:
                    results.append(10 * np.log10(p_i_mw))
                results.append(n_i)

                # SIR
                results.append(p_rx_linear / p_i_mw if p_i_mw else None)

                # SNR
                system_t = self.link_budget_input_params['T_system']
                bw = self.link_budget_input_params['bandwidth']
                p_noise_mw = c.BOLTZMAN * system_t * bw * 1e3
                results.append(p_rx_linear / p_noise_mw)

                # SINR
                sinr_linear = p_rx_linear / (p_i_mw + p_noise_mw)
                results.append(sinr_linear)

                # Capacity (Gbps)
                results.append(bw * np.log2(1 + sinr_linear) * 1e-9 if sinr_linear != 0 else None)

                '''
                # Link budget
                g1 = 2 / (1 - np.cos(alpha_rad / 2))  # Gain in the direction of the main beam
                g1_db = 10 * np.log10(g1)
                rx_snr = snr(p_tx_dbm, d_tx_rx, fc, g1_db, bw, antenna_T, l_abs=0, g_rx=None, nf=NF_db)
                results.append(rx_snr)
                '''

                return results

            results_df[self.results_columns] = results_df.progress_apply(compute_row_results, axis=1,
                                                                         result_type='expand')

            return results_df

        # Dataframe cache file format
        df_extension = self.orbital_input_params['math_results_extension'] + self.link_budget_input_params['extension']
        df_path = os.path.join(cache_path, df_extension)
        rep = True if not os.path.isfile(df_path) or replace else False

        load_or_recompute_and_cache(df_path, generate_results, replace=rep, verbose=True)

    def beamwidth_single_orbit_simulation(self):
        """Simulation analysis of the Single orbit scenario

        Returns:
            Dataframe: Results in a Dataframe format. Input columns: ['alpha', 'n_sats', 'h']

        """
        # ------Inputs------
        alpha_list = np.arange(self.orbital_input_params['alpha_lims'][0],
                               self.orbital_input_params['alpha_lims'][1],
                               self.orbital_input_params['alpha_resolution_sim'])
        n_sats_list = self.orbital_input_params['n_sats_list']
        h_list = np.arange(self.orbital_input_params['h_lims'][0],
                           self.orbital_input_params['h_lims'][1],
                           self.orbital_input_params[
                               'h_resolution_sim'])  # altitude difference between low and upper orbit
        input_columns = ['alpha', 'n_sats', 'h']
        replace = self.orbital_input_params['sim_replace']

        # Link budget inputs
        p_tx_dbm = self.link_budget_input_params['p_tx_dbm']  # W = 10 ** ((P(dBm)-30)/10)
        p_tx = 10 ** (p_tx_dbm / 10)
        fc = self.link_budget_input_params['center_frequency']
        bw = self.link_budget_input_params['bandwidth']
        t_system = self.link_budget_input_params['T_system']
        # ------------------

        # Object Simulation creation, with the corresponding results folder
        t_start = datetime(2022, 1, 1)
        sim = Simulation(t_start=t_start)
        np.random.seed(0)

        # sim params on a .txt at the simulation results folder
        with open(os.path.join(sim.results_folder, 'sim_params.txt'), 'w') as file:
            file.write('SINGLE ORBIT SIMULATION ANALYSIS\n')
            file.write('\nCommon Input parameters\n')
            for k, v in self.link_budget_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
            file.write('\nSingle orbit Input parameters\n')
            for k, v in self.orbital_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
        file.close()

        def generate_results():
            # Results dataframe creation
            results_df = pd.DataFrame(list(product(alpha_list, n_sats_list, h_list)), columns=['alpha', 'n_sats', 'h'])

            # Fixed orbital parameters
            e = 0.0  # eccentricity
            omega = 0  # Right ascension of ascending node in degrees
            inc = 0  # Inclination in degrees
            w = 0  # Argument of perigee in degrees

            rx_id = 0
            tx_id = 1

            def compute_row_results(row):
                n_sats = int(row['n_sats'])
                alpha = row['alpha']
                h = row['h']
                results = []

                alpha_rad = np.deg2rad(alpha)
                half_cone_angle = alpha / 2
                g1 = 2 / (1 - np.cos(alpha_rad / 2))  # Gain in the direction of the main beam
                g1_db = 10 * np.log10(g1)

                d_i_min = None

                a = Re + h
                orbit = Orbit(sim, e, a, omega, inc, w)
                orbit.add_satellites(n_sats=n_sats)
                const = Constellation(sim, orbits=[orbit])
                const.update_SSPs()  # Required before computing distances between satellites

                tx_sat = const.satellites[tx_id]
                rx_sat = const.satellites[rx_id]

                # Distance
                distance = tk.sat_to_sat_disance(tx_sat.xyz_r, rx_sat.xyz_r)
                results.append(distance)

                # Rx power
                p_rx_dbm = rx_power(p_tx_dbm, distance, fc, l_abs=0, g_tx=g1_db)
                results.append(p_rx_dbm)
                p_rx_linear = 10 ** (p_rx_dbm / 10)
                results.append(p_rx_linear)

                # Debug
                if n_sats == 20 and h == 500e3 and alpha == 60:
                    a = 0

                # Interference
                p_i_mw = 0
                n_i = 0
                for i, sat in enumerate(const.satellites[:-1]):
                    if i == 0:
                        pass  # If not, we would be considering the link of interest as interference
                    else:
                        i_rx_sat = const.satellites[i]
                        i_tx_sat = const.satellites[(i + 1) % n_sats]
                        if tk.check_interference(rx_sat, tx_sat, i_rx_sat, i_tx_sat, half_cone_angle=half_cone_angle):
                            n_i += 1
                            distance_interferer = tk.sat_to_sat_disance(i_tx_sat.xyz_r, rx_sat.xyz_r)
                            g_i_tx = g1_db
                            p_i_dbm = rx_power(p_tx_dbm, distance_interferer, fc, l_abs=0, g_tx=g_i_tx)
                            p_i_mw += (10 ** (p_i_dbm / 10))
                            if not d_i_min:
                                d_i_min = distance_interferer
                            elif distance_interferer < d_i_min:
                                d_i_min = distance_interferer
                results.append(d_i_min)
                results.append(p_i_mw)
                if p_i_mw == 0:
                    results.append(None)
                else:
                    results.append(10 * np.log10(p_i_mw))
                results.append(n_i)

                # SIR
                results.append(p_rx_linear / p_i_mw if p_i_mw else None)

                # SNR
                system_t = self.link_budget_input_params['T_system']
                bw = self.link_budget_input_params['bandwidth']
                p_noise_mw = c.BOLTZMAN * system_t * bw * 1e3
                results.append(p_rx_linear / p_noise_mw)

                # SINR
                sinr_linear = p_rx_linear / (p_i_mw + p_noise_mw)
                results.append(sinr_linear)

                # Capacity (Gbps)
                results.append(bw * np.log2(1 + sinr_linear) * 1e-9 if sinr_linear != 0 else None)

                return results

            results_df[self.results_columns] = results_df.progress_apply(compute_row_results, axis=1,
                                                                         result_type='expand')

            return results_df

        # Dataframe cache file format
        df_extension = self.orbital_input_params['sim_results_extension'] + self.link_budget_input_params['extension']
        df_path = os.path.join(cache_path, df_extension)
        rep = True if not os.path.isfile(df_path) or replace else False

        load_or_recompute_and_cache(df_path, generate_results, replace=rep, verbose=True)

    def shifted_orbits_pdf(self):
        """Mathematical analysis of the Shifted orbits scenario

        Returns:
            Dataframe: Results in a Dataframe format. Input columns:
            ['alpha', 'n_sats', 'h', 'inclination', 'beta', 'time']

        """
        # ------Inputs------
        try:
            alpha_list = np.arange(self.orbital_input_params['alpha_lims'][0],
                                   self.orbital_input_params['alpha_lims'][1],
                                   self.orbital_input_params['alpha_resolution_math'])
        except:
            alpha_list = self.link_budget_input_params['alpha_list']

        try:
            n_sats_list = self.orbital_input_params['n_sats_list']
        except:
            try:
                n_sats_list = self.link_budget_input_params['n_sats_list']
            except:
                n_sats_list = np.arange(self.orbital_input_params['n_sats_lims'][0],
                                        self.orbital_input_params['n_sats_lims'][1],
                                        self.orbital_input_params['n_sats_resolution_math'])

        try:
            inclination_list = np.arange(self.orbital_input_params['inclination_lims'][0],
                                         self.orbital_input_params['inclination_lims'][1],
                                         self.orbital_input_params['inclination_resolution_math'])
        except:
            inclination_list = self.link_budget_input_params['inclination_list']

        h_list = np.arange(self.orbital_input_params['h_lims'][0],
                           self.orbital_input_params['h_lims'][1],
                           self.orbital_input_params['h_resolution_math'])

        n_timesteps = self.orbital_input_params['n_timesteps_math']

        # Fixed orbital parameters
        omega = self.orbital_input_params['RAA difference']  # Right ascension of ascending node in degrees
        w = 0  # Argument of perigee in degrees

        # Link budget inputs
        p_tx_dbm = self.link_budget_input_params['p_tx_dbm']  # W = 10 ** ((P(dBm)-30)/10)
        p_tx = 10 ** (p_tx_dbm / 10)
        fc = self.link_budget_input_params['center_frequency']

        replace = self.orbital_input_params['replace_math']
        other_orbit_only = self.orbital_input_params['other_orbit_only']
        # ------------------

        # Object Simulation creation, with the corresponding results folder
        t_start = datetime(2022, 1, 1)
        sim = Simulation(t_start=t_start)
        np.random.seed(0)

        # sim params on a .txt at the simulation results folder
        with open(os.path.join(sim.results_folder, 'sim_params.txt'), 'w') as file:
            file.write('SHIFTED ORBITS MATH ANALYSIS\n')
            file.write('\nCommon Input parameters\n')
            for k, v in self.link_budget_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
            file.write('\nShifted orbit Input parameters\n')
            for k, v in self.orbital_input_params.items():
                file.write(k + ': ' + str(v) + '\n')
        file.close()

        def generate_results():
            # Results dataframe creation
            results_df = pd.DataFrame(list(product(alpha_list, n_sats_list, h_list, inclination_list)),
                                      columns=['alpha', 'n_sats', 'h', 'inclination'])

            # Compute angular offset and time vector for each number of satellites and altitudes, respectively
            betas = []
            ts = []
            for _ in alpha_list:
                for n_sats in n_sats_list:
                    max_angular_offset = 2 * np.pi / n_sats
                    for h in h_list:
                        for _ in inclination_list:
                            beta_vec = np.linspace(0, max_angular_offset,
                                                   self.orbital_input_params['beta_steps_math'])
                            betas.append(beta_vec)

                            T = ad.period_from_semi_major_axis(Re + h)
                            time_steps = pd.date_range(start=sim.t_start, end=sim.t_start + 2 * timedelta(seconds=T),
                                                       periods=n_timesteps)

                            # Adjust time vectors to expand on the SIR drops windows. Windows:
                            minutes = [7,
                                       30, 31 + 25,
                                       60 + 17, 60 + 18 + 25,
                                       120 + 5, 120 + 6 + 25,
                                       120 + 51]
                            windows = [sim.t_start + timedelta(minutes=minute) for minute in minutes]
                            # Delete windows
                            time_steps = time_steps[(time_steps > windows[0]) &
                                                    (~((time_steps > windows[1]) & (time_steps < windows[2]))) &
                                                    (~((time_steps > windows[3]) & (time_steps < windows[4]))) &
                                                    (~((time_steps > windows[5]) & (time_steps < windows[6]))) &
                                                    (time_steps < windows[7])]

                            n_timesteps_expanded = 1000

                            # Create expanded timesteps
                            window_1 = pd.date_range(start=sim.t_start, end=windows[0], periods=n_timesteps_expanded)
                            window_2 = pd.date_range(start=windows[1], end=windows[2], periods=n_timesteps_expanded)
                            window_3 = pd.date_range(start=windows[3], end=windows[4], periods=n_timesteps_expanded)
                            window_4 = pd.date_range(start=windows[5], end=windows[6], periods=n_timesteps_expanded)
                            window_5 = pd.date_range(start=windows[7], end=sim.t_start + 2 * timedelta(seconds=T),
                                                     periods=n_timesteps_expanded)
                            window_vecs = [window_1, window_2, window_3, window_4, window_5]

                            # Insert expanded timesteps into windows
                            for window in window_vecs:
                                time_steps = time_steps.append(window)
                            time_steps = time_steps.sort_values()

                            for _ in beta_vec:
                                ts.append(time_steps)
            results_df['beta'] = betas
            results_df = results_df.explode('beta', ignore_index=True)
            results_df['time'] = ts
            results_df = results_df.explode('time', ignore_index=True)

            def compute_row_results(row):
                alpha = row['alpha']
                n_sats = row['n_sats']
                h = row['h']
                beta = row['beta']
                t = row['time']
                inc = row['inclination']
                results = []
                p_i_mw = 0
                n_i = 0

                alpha_rad = np.deg2rad(alpha)
                g1 = 2 / (1 - np.cos(np.deg2rad(alpha) / 2))  # Gain in the direction of the main beam
                g1_db = 10 * np.log10(g1)
                half_cone_angle = alpha / 2

                T = ad.period_from_semi_major_axis(Re + h)
                progress = (t - t_start).total_seconds() / T

                true_anomaly_rx = 0 + 2 * np.pi * progress
                true_anomaly_tx = true_anomaly_rx + 2 * np.pi / n_sats

                r0_rx = np.array([(Re + h) * np.cos(true_anomaly_rx), (Re + h) * np.sin(true_anomaly_rx), 0])
                r0_tx = np.array([(Re + h) * np.cos(true_anomaly_tx), (Re + h) * np.sin(true_anomaly_tx), 0])

                M1 = ad.orbital_to_GEC_transformation_matrix(0, w, inc)
                M2 = ad.orbital_to_GEC_transformation_matrix(omega, w, inc)

                r_rx = np.matmul(M1, np.transpose(r0_rx))
                r_tx = np.matmul(M1, np.transpose(r0_tx))
                r_rx_tx = r_tx - r_rx
                r_rx_tx_norm = np.linalg.norm(r_rx_tx)

                d_i_min = None

                # If we consider same orbit as interference too
                if not other_orbit_only:
                    cond1 = n_sats / np.pi * np.arccos(Re / (Re + h))
                    cond2 = 1 + n_sats / (2 * np.pi) * alpha_rad
                    i_max = int(min(cond1, cond2))
                    n_i = i_max - 1

                    lambda_fc = c.SPEED_OF_LIGHT / fc
                    num = lambda_fc ** 2 * p_tx

                    for i in range(2, i_max + 1):
                        den = 8 * (np.pi ** 2) * (1 - np.cos(alpha_rad / 2)) ** 2 * (Re + h) ** 2 * (
                                1 - np.cos(2 * np.pi * i / n_sats))

                        p_i_mw += num / den
                        d_i = np.sqrt(2 * (Re + h) ** 2 * (1 - np.cos(2 * np.pi * i / n_sats)))

                        # Distance to nearest interferer
                        if not d_i_min:
                            d_i_min = d_i
                        elif d_i < d_i_min:
                            d_i_min = d_i

                # Shifted orbit interference
                for j in range(n_sats):
                    true_anomaly_j_rx = (beta + 2 * np.pi * progress + 2 * np.pi * j / n_sats) % (
                            2 * np.pi)  # Where the interferer is pointing
                    true_anomaly_j1_tx = (beta + 2 * np.pi * progress + 2 * np.pi * ((j + 1) % n_sats) / n_sats) % (
                            2 * np.pi)  # interferer

                    r0_jrx = np.array(
                        [(Re + h) * np.cos(true_anomaly_j_rx), (Re + h) * np.sin(true_anomaly_j_rx), 0])
                    r0_j1tx = np.array(
                        [(Re + h) * np.cos(true_anomaly_j1_tx), (Re + h) * np.sin(true_anomaly_j1_tx), 0])

                    r_jrx = np.matmul(M2, np.transpose(r0_jrx))
                    r_j1tx = np.matmul(M2, np.transpose(r0_j1tx))

                    r_rx_j1tx = r_j1tx - r_rx

                    r_j1tx_jrx = r_jrx - r_j1tx
                    r_j1tx_rx = -r_rx_j1tx

                    # From receiver perspective
                    dot_prod = np.dot(r_rx_tx, r_rx_j1tx)
                    r_rx_j1tx_norm = np.linalg.norm(r_rx_j1tx)
                    theta_j = np.rad2deg(
                        np.arccos(dot_prod / (r_rx_tx_norm * r_rx_j1tx_norm)))  # Angle Interferer-Rx-Tx

                    # From interferer perspective
                    dot_prod = np.dot(r_j1tx_jrx, r_j1tx_rx)
                    d_i = r_rx_j1tx_norm
                    theta_j1 = np.rad2deg(
                        np.arccos(
                            dot_prod / (np.linalg.norm(r_j1tx_jrx) * d_i)))  # Angle Interferer+1-Interferer-Rx

                    # Check no blockage from the Earth
                    dot_prod = np.dot(r_rx, r_jrx)
                    V = (Re ** 2 * (((Re + h) ** 2 + (Re + h) ** 2) - 2 * (dot_prod))) - (
                            ((Re + h) ** 2) * ((Re + h) ** 2)) + (dot_prod) ** 2

                    if V <= 0 and abs(theta_j) <= half_cone_angle and abs(theta_j1) <= half_cone_angle:
                        # Distance to nearest interferer
                        if not d_i_min:
                            d_i_min = d_i
                        elif d_i < d_i_min:
                            d_i_min = d_i
                        # Interference linear
                        p_i_dbm = rx_power(p_tx_dbm, d_i, fc, l_abs=0, g_tx=g1_db)
                        p_i_mw += (10 ** (p_i_dbm / 10))
                        # Number of interferers
                        n_i += 1

                results.append(r_rx_tx_norm)  # Tx-Rx distance
                p_rx_dbm = rx_power(p_tx_dbm, r_rx_tx_norm, fc, l_abs=0, g_tx=g1_db)  # Rx power
                results.append(p_rx_dbm)
                p_rx_linear = 10 ** (p_rx_dbm / 10)
                results.append(p_rx_linear)
                results.append(d_i_min)
                if p_i_mw == 0:
                    results.append(0)
                    results.append(None)
                else:
                    results.append(p_i_mw)
                    results.append(10 * np.log10(p_i_mw))
                results.append(n_i)

                # SIR
                results.append(p_rx_linear / p_i_mw if p_i_mw else None)

                # SNR
                system_t = self.link_budget_input_params['T_system']
                bw = self.link_budget_input_params['bandwidth']
                p_noise_mw = c.BOLTZMAN * system_t * bw * 1e3
                results.append(p_rx_linear / p_noise_mw)

                # SINR
                sinr_linear = p_rx_linear / (p_i_mw + p_noise_mw)
                results.append(sinr_linear)

                # Capacity (Gbps)
                results.append(bw * np.log2(1 + sinr_linear) * 1e-9 if sinr_linear != 0 else None)

                return results

            results_df[self.results_columns] = results_df.progress_apply(compute_row_results, axis=1,
                                                                         result_type='expand')

            return results_df

        # Dataframe cache file format
        df_extension = self.orbital_input_params['math_results_extension'] + '_other_orbit_only_' + str(
            other_orbit_only) \
                       + self.link_budget_input_params['extension']
        df_path = os.path.join(cache_path, df_extension)
        rep = True if not os.path.isfile(df_path) or replace else False

        load_or_recompute_and_cache(df_path, generate_results, replace=rep, verbose=True)

    def coplanar_testing_plot_from_above(self):
        n_sats = 40
        h_low = 500e3
        h_high = 510e3
        # Computing period of simulation based on input parameters
        T1 = ad.period_from_semi_major_axis(c.EARTH_RADIUS_M + h_low)
        T2 = ad.period_from_semi_major_axis(c.EARTH_RADIUS_M + h_high)
        inc_beta_max = 2 * np.pi / n_sats
        T_beta_max = inc_beta_max / (2 * np.pi * (1 / T1 - 1 / T2))

        # Simulation initialization
        start_time = datetime(2022, 1, 1)
        end_time = start_time + timedelta(seconds=T_beta_max)
        time_step = (end_time - start_time) / self.orbital_input_params['beta_steps_math']
        sim = Simulation(t_start=start_time, t_end=end_time, t_step=time_step)
        np.random.seed(0)

        # Orbital parameters and orbit object creation
        e = 0.0  # eccentricity
        inc = 50  # Inclination in degrees
        w = 0  # Argument of perigee in degrees
        omega = 0  # Right ascension of ascending node in degrees

        # Lower orbit
        a = c.EARTH_RADIUS_M + h_low  # Semimajor axis in meters
        orbit_low = Orbit(sim, e, a, omega, inc, w, initial_anomaly=0)
        orbit_low.add_satellites(n_sats=n_sats)

        # Higher orbit
        a = c.EARTH_RADIUS_M + h_high  # Semimajor axis in meters
        orbit_high = Orbit(sim, e, a, omega, inc, w, initial_anomaly=0)
        orbit_high.add_satellites(n_sats=n_sats)
        const = Constellation(sim, orbits=[orbit_low, orbit_high])
        const.update_SSPs()  # Required before computing distances between satellites

        for t in tqdm(const.simulation.ts):
            # Plot from above for t
            beta = ((t - sim.t_start).total_seconds() / T_beta_max) * inc_beta_max
            plot_from_above(n_sats, h_low, n_sats, h_high, anomaly_diff=beta, sim=sim,
                            ext='{}'.format(t.strftime("%Y%m%d%H%M%S")), format='.jpg')

        # Video
        images = []
        for t in tqdm(const.simulation.ts):
            images.append(imageio.imread(
                os.path.join(sim.figures_folder, "2d_plot" + '{}'.format(t.strftime("%Y%m%d%H%M%S")) + ".jpg")))
        imageio.mimsave(os.path.join(const.simulation.figures_folder, 'Video_{}_fps.mp4'.format(6)), images, fps=5)


if __name__ == '__main__':
    pass
