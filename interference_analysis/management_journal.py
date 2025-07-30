import os.path
from simulator.file_utils import load_files
from analysis import Analysis, cache_path
from plotter import Plotter
import numpy as np

"""Management script

Main code creating instances of the Analysis and Plotter classes.
The simulations input parameters are defined here.
"""

####### General #######
# THz
thz_input_params = {'p_tx_dbm': 27,  # W = 10 ** ((P(dBm)-30)/10)
                    'bandwidth': 10e9,
                    'T_system': 100,
                    'center_frequency': 130e9,
                    'alpha_list': [1, 3, 5],
                    'extension': '_thz.dat'
                    }

# mmWave
mmwave_input_params = {'p_tx_dbm': 60,  # W = 10 ** ((P(dBm)-30)/10)
                       'bandwidth': 0.4e9,
                       'T_system': 100,
                       'center_frequency': 38e9,
                       'alpha_list': [3, 5, 10, 20, 30, 40],
                       'extension': '_mmWave.dat'
                       }

####### Single orbits #######
# Link budget parameters
single_sir_vs_nsats_figure_input_params = {'p_tx_dbm': 60,  # W = 10 ** ((P(dBm)-30)/10)
                                           'bandwidth': 0.4e9,
                                           'T_system': 100,
                                           'center_frequency': 38e9,
                                           'alpha_list': [3, 5, 10, 20, 30, 40],
                                           'extension': '_sir_single_journal.dat'
                                           }

# orbital parameters
single_orbit_params = {'h_lims': [500 * 1e3, 2000.1 * 1e3],
                       'n_sats_lims': [10, 201],
                       # Math
                       'n_sats_resolution_math': 1,
                       'h_resolution_math': 10 * 1e3,  # height resolution
                       'math_results_extension': 'single_orbit_math',
                       'math_replace': False,  # Change to False to use cache files, if available
                       # Simulation
                       'n_sats_resolution_sim': 10,
                       'h_resolution_sim': 100 * 1e3,
                       'sim_results_extension': 'single_orbit_simulation',
                       'sim_replace': False  # Change to False to use cache files, if available
                       }

####### Coplanar orbits #######
# Link budget parameters
coplanar_sir_interfernce_pdf_input_params = {'p_tx_dbm': 60,  # W = 10 ** ((P(dBm)-30)/10)
                                             'bandwidth': 0.4e9,
                                             'T_system': 100,
                                             'center_frequency': 38e9,
                                             'alpha_list': [10, 20, 30],
                                             'extension': '_sir_coplanar_journal.dat'
                                             }

# orbital parameters
coplanar_orbits_params = {'h_low': 500e3,
                          'h_lims': [510 * 1e3, 1000.1 * 1e3],
                          'n_sats_list': [50, 100],  # Iterable for number of satellites in orbit
                          'higher_orbit_only': False,
                          # Math
                          'beta_steps_math': 100,
                          'h_resolution_math': 5e3,  # height resolution
                          'math_results_extension': 'coplanar_orbits_math',
                          'math_replace': False,  # Change to False to use cache files, if available
                          # Simulation
                          'beta_steps_sim': 20,
                          'h_resolution_sim': 25e3,
                          'sim_results_extension': 'coplanar_orbits_simulation',
                          'sim_replace': False,  # Change to False to use cache files, if available
                          # Higher orbit analysis
                          'math_results_higher_extension': 'coplanar_higher_orbits_math',
                          'sim_results_higher_extension': 'coplanar_higher_orbits_simulation'
                          }

coplanar_sir_interfernce_pdf_orbital_params = {'h_low': 500e3,
                                               'h_lims': [510e3, 510.1e3],
                                               'n_sats_list': [50, 100],  # Iterable for number of satellites in orbit
                                               'higher_orbit_only': False,
                                               # Math
                                               'beta_steps_math': 100000,
                                               'h_resolution_math': 5e3,  # height resolution
                                               'math_results_extension': 'coplanar_orbits_math',
                                               'math_replace': False,  # Change to False to use cache files, if available
                                               # Simulation
                                               'beta_steps_sim': 20,
                                               'h_resolution_sim': 25e3,
                                               'sim_results_extension': 'coplanar_orbits_simulation',
                                               'sim_replace': False,  # Change to False to use cache files, if available
                                               # Higher orbit analysis
                                               'math_results_higher_extension': 'coplanar_higher_orbits_math',
                                               'sim_results_higher_extension': 'coplanar_higher_orbits_simulation'
                                               }

# coplanar_sir_interfernce_pdf_orbital_params = {'h_low': 500e3,
#                                                'h_lims': [510e3, 510.1e3],
#                                                'n_sats_list':  np.arange(10, 500, 20),  # Iterable for number of satellites in orbit
#                                                'higher_orbit_only': False,
#                                                # Math
#                                                'beta_steps_math': 100,
#                                                'h_resolution_math': 5e3,  # height resolution
#                                                'math_results_extension': 'coplanar_orbits_math',
#                                                'math_replace': True,  # Change to False to use cache files, if available
#                                                # Simulation
#                                                'beta_steps_sim': 20,
#                                                'h_resolution_sim': 25e3,
#                                                'sim_results_extension': 'coplanar_orbits_simulation',
#                                                'sim_replace': False,  # Change to False to use cache files, if available
#                                                # Higher orbit analysis
#                                                'math_results_higher_extension': 'coplanar_higher_orbits_math',
#                                                'sim_results_higher_extension': 'coplanar_higher_orbits_simulation'
#                                                }

####### Shifted orbits #######
# Link budget parameters
shifted_sir_figures_input_params = {'p_tx_dbm': 60,  # W = 10 ** ((P(dBm)-30)/10)
                                    'bandwidth': 0.4e9,
                                    'T_system': 100,
                                    'center_frequency': 38e9,
                                    'alpha_list': [3, 5, 10, 20, 30, 40],
                                    'extension': '_sir_figure.dat'
                                    }

# THz
shifted_sinr_thz_input_params = {'p_tx_dbm': 27,  # W = 10 ** ((P(dBm)-30)/10)
                                 'bandwidth': 10e9,
                                 'T_system': 100,
                                 'center_frequency': 130e9,
                                 'alpha_list': [1, 3, 5],
                                 'n_sats_list': [100, 150, 200],
                                 'extension': '_thz.dat'
                                 }

shifted_capacity_thz_input_params = {'p_tx_dbm': 27,  # W = 10 ** ((P(dBm)-30)/10)
                                     'bandwidth': 10e9,
                                     'T_system': 100,
                                     'center_frequency': 130e9,
                                     'alpha_list': [1, 5],
                                     'inclination_list': [1, 3],
                                     'extension': '_thz.dat'
                                     }

# mmWave
shifted_sinr_mmwave_input_params = {'p_tx_dbm': 60,  # W = 10 ** ((P(dBm)-30)/10)
                                    'bandwidth': 0.4e9,
                                    'T_system': 100,
                                    'center_frequency': 38e9,
                                    'alpha_list': [3, 5, 10, 20, 30, 40],
                                    'n_sats_list': [10, 25, 50],
                                    'extension': '_mmWave.dat'
                                    }

shifted_capacity_mmwave_input_params = {'p_tx_dbm': 60,  # W = 10 ** ((P(dBm)-30)/10)
                                        'bandwidth': 0.4e9,
                                        'T_system': 100,
                                        'center_frequency': 38e9,
                                        'alpha_list': [10, 40],
                                        'inclination_list': [3, 10],
                                        'extension': '_mmWave.dat'
                                        }

# orbital parameters
shifted_orbits_parameters = {'h_lims': [500 * 1e3, 500.1 * 1e3],
                             'n_sats_list': [20, 40, 60, 80, 100],  # Iterable for number of satellites in orbit
                             'alpha_lims': [1, 30.1],
                             'inclination_lims': [3, 30.1],
                             'RAA difference': 90,
                             'other_orbit_only': False,
                             # Math
                             'h_resolution_math': 10e3,
                             'alpha_resolution_math': 0.5,
                             'inclination_resolution_math': 1,
                             'beta_steps_math': 1,
                             'n_timesteps_math': 100,
                             'math_results_extension': 'shifted_orbits_math',
                             'replace_math': False,
                             # Simulation
                             'h_resolution_sim': 10e3,
                             'alpha_resolution_sim': 2,
                             'inclination_resolution_sim': 4,
                             'beta_steps_sim': 1,
                             'n_timesteps_sim': 20,
                             'sim_results_extension': 'shifted_orbits_simulation',
                             'replace_sim': False
                             }

shifted_sir_figures_orbital_parameters = {'h_lims': [500 * 1e3, 500.1 * 1e3],
                                          'n_sats_list': [20, 40, 60, 80, 100],
                                          # Iterable for number of satellites in orbit
                                          'alpha_lims': [20, 20.1],
                                          'inclination_lims': [3, 9.1],
                                          'RAA difference': 90,
                                          'other_orbit_only': False,
                                          # Math
                                          'h_resolution_math': 10e3,
                                          'alpha_resolution_math': 0.5,
                                          'inclination_resolution_math': 1,
                                          'beta_steps_math': 1,
                                          'n_timesteps_math': 10000,
                                          'math_results_extension': 'shifted_orbits_math',
                                          'replace_math': False,
                                          # Simulation
                                          'h_resolution_sim': 10e3,
                                          'alpha_resolution_sim': 4,
                                          'inclination_resolution_sim': 3,
                                          'beta_steps_sim': 1,
                                          'n_timesteps_sim': 20,
                                          'sim_results_extension': 'shifted_orbits_simulation',
                                          'replace_sim': False
                                          }

shifted_interference_pdf_orbital_parameters = {'h_lims': [500 * 1e3, 500.1 * 1e3],
                                               'n_sats_list': [20, 40, 60, 80, 100],
                                               # Iterable for number of satellites in orbit
                                               'alpha_lims': [20, 20.1],
                                               'inclination_lims': [3, 9.1],
                                               'RAA difference': 90,
                                               'other_orbit_only': False,
                                               # Math
                                               'h_resolution_math': 10e3,
                                               'alpha_resolution_math': 0.5,
                                               'inclination_resolution_math': 1,
                                               'beta_steps_math': 1,
                                               'n_timesteps_math': 100,
                                               'math_results_extension': 'shifted_orbits_math_pdf',
                                               'replace_math': False,
                                               # Simulation
                                               'h_resolution_sim': 10e3,
                                               'alpha_resolution_sim': 4,
                                               'inclination_resolution_sim': 3,
                                               'beta_steps_sim': 1,
                                               'n_timesteps_sim': 20,
                                               'sim_results_extension': 'shifted_orbits_simulation_pdf',
                                               'replace_sim': False
                                               }

shifted_sinr_figure_orbital_parameters = {'h_lims': [500 * 1e3, 500.1 * 1e3],
                                          # Iterable for number of satellites in orbit
                                          'alpha_lims': [1, 30.1],
                                          'inclination_lims': [3, 3.1],
                                          'RAA difference': 90,
                                          'other_orbit_only': False,
                                          # Math
                                          'h_resolution_math': 10e3,
                                          'alpha_resolution_math': 0.5,
                                          'inclination_resolution_math': 3,
                                          'beta_steps_math': 1,
                                          'n_timesteps_math': 100,
                                          'math_results_extension': 'shifted_orbits_math_sinr_fig',
                                          'replace_math': False,
                                          # Simulation
                                          'h_resolution_sim': 10e3,
                                          'alpha_resolution_sim': 4,
                                          'inclination_resolution_sim': 3,
                                          'beta_steps_sim': 1,
                                          'n_timesteps_sim': 20,
                                          'sim_results_extension': 'shifted_orbits_simulation_sinr_fig',
                                          'replace_sim': False
                                          }

shifted_capacity_figure_orbital_parameters = {'h_lims': [500 * 1e3, 500.1 * 1e3],
                                              'n_sats_list': [100, 50],
                                              # Iterable for number of satellites in orbit
                                              'inclination_lims': [3, 18],
                                              'RAA difference': 90,
                                              'other_orbit_only': False,
                                              # Math
                                              'h_resolution_math': 10e3,
                                              'alpha_resolution_math': 0.5,
                                              'inclination_resolution_math': 0.5,
                                              'beta_steps_math': 1,
                                              'n_timesteps_math': 100,
                                              'math_results_extension': 'shifted_orbits_capacity_math',
                                              'replace_math': False,
                                              # Simulation
                                              'h_resolution_sim': 10e3,
                                              'alpha_resolution_sim': 4,
                                              'inclination_resolution_sim': 3,
                                              'beta_steps_sim': 1,
                                              'n_timesteps_sim': 20,
                                              'sim_results_extension': 'shifted_orbits_capacity_simulation',
                                              'replace_sim': False
                                              }

shifted_capacity_n_sats_figure_orbital_parameters = {'h_lims': [500 * 1e3, 500.1 * 1e3],
                                                     'n_sats_lims': [10, 201],
                                                     # Iterable for number of satellites in orbit
                                                     'RAA difference': 90,
                                                     'other_orbit_only': False,
                                                     # Math
                                                     'h_resolution_math': 10e3,
                                                     'n_sats_resolution_math': 1,
                                                     'beta_steps_math': 1,
                                                     'n_timesteps_math': 1000,
                                                     'math_results_extension': 'shifted_orbits_capacity_vs_n_sats_math',
                                                     'replace_math': False}

####### Complete constellation #######
# Link budget params
thz_complete_input_params = {'p_tx_dbm': 27,  # W = 10 ** ((P(dBm)-30)/10)
                             'bandwidth': 10e9,
                             'T_system': 355,
                             'center_frequency': 130e9,
                             'alpha_list': [1],
                             'extension': '_thz.dat'
                             }

# mmWave
mmwave_complete_input_params = {'p_tx_dbm': 60,  # W = 10 ** ((P(dBm)-30)/10)
                                'bandwidth': 0.4e9,
                                'T_system': 355,
                                'center_frequency': 38e9,
                                'alpha_list': [5],
                                'extension': '_mmWave.dat'
                                }

# Orbital params
complete_constellation_orbital_parameters = {'h': 500e3,
                                             'h_diff': 10e3,
                                             'n_sats_lims': [10, 511],
                                             'n_sats_resolution': 20,
                                             'n_orbits': 10,
                                             'inclination': [3, 50],
                                             'n_timesteps_math': 100,
                                             'math_results_extension': 'complete_constellation',
                                             'math_replace': False
                                             }

# results_columns = ['tx_rx_distance', 'rx_power', 'rx_power_linear', 'd_interferer', 'interference_linear',
#                    'interference', 'n_interferers', 'sir_linear', 'snr_linear', 'sinr_linear', 'capacity_gbps']

results_columns = ['tx_rx_distance', 'rx_power', 'rx_power_linear', 'd_interferer',
                   'interference_linear', 'interference', 'n_interferers', 'sir_linear', 'snr_linear', 'sinr_linear',
                   'capacity_gbps', 'capacity_shifted_1_coplanar', 'capacity_shifted', 'capacity_single']


def single_orbit_sir_vs_nsats_figure_5a():
    # SIR figure
    # Create analysis object
    single_orbit_analysis = Analysis(single_sir_vs_nsats_figure_input_params, single_orbit_params, results_columns)

    # Compute or cache Analysis results
    single_orbit_analysis.single_orbit_math()
    single_orbit_analysis.single_orbit_simulation()

    # Load results
    sir_figure_math_results_path = os.path.join(cache_path, single_orbit_params['math_results_extension'] +
                                                single_sir_vs_nsats_figure_input_params['extension'])
    sir_figure_sim_results_path = os.path.join(cache_path, single_orbit_params['sim_results_extension'] +
                                               single_sir_vs_nsats_figure_input_params['extension'])

    sir_figure_math_results_df = load_files(sir_figure_math_results_path)
    sir_figure_orbit_sim_results_df = load_files(sir_figure_sim_results_path)

    # Create plotter object
    single_orbit_plotter = Plotter(results_columns,
                                   link_budget_input_params=single_sir_vs_nsats_figure_input_params,
                                   orbital_input_params=single_orbit_params,
                                   math_results_df=sir_figure_math_results_df,
                                   sim_results_df=sir_figure_orbit_sim_results_df)

    single_orbit_plotter.journal_single_sir_vs_number_of_satellites(file_ext='.pdf')


def single_orbit_sinr_capacity_figures_5b_5c():
    # Create Analysis object
    thz_single_orbit_analysis = Analysis(thz_input_params, single_orbit_params, results_columns)
    mmwave_single_orbit_analysis = Analysis(mmwave_input_params, single_orbit_params, results_columns)

    # Compute or cache Analysis results
    thz_single_orbit_analysis.single_orbit_math()
    thz_single_orbit_analysis.single_orbit_simulation()
    mmwave_single_orbit_analysis.single_orbit_math()
    mmwave_single_orbit_analysis.single_orbit_simulation()

    # Load results
    thz_math_results_path = os.path.join(cache_path,
                                         single_orbit_params['math_results_extension'] + thz_input_params['extension'])
    thz_sim_results_path = os.path.join(cache_path,
                                        single_orbit_params['sim_results_extension'] + thz_input_params['extension'])
    mmwave_math_results_path = os.path.join(cache_path,
                                            single_orbit_params['math_results_extension'] + mmwave_input_params[
                                                'extension'])
    mmwave_sim_results_path = os.path.join(cache_path,
                                           single_orbit_params['sim_results_extension'] + mmwave_input_params[
                                               'extension'])
    thz_single_orbit_math_results_df = load_files(thz_math_results_path)
    thz_single_orbit_sim_results_df = load_files(thz_sim_results_path)
    mmwave_single_orbit_math_results_df = load_files(mmwave_math_results_path)
    mmwave_single_orbit_sim_results_df = load_files(mmwave_sim_results_path)

    # Create plotter object
    single_orbit_plotter = Plotter(results_columns,
                                   thz_input_params=thz_input_params,
                                   mmwave_input_params=mmwave_input_params,
                                   orbital_input_params=single_orbit_params,
                                   thz_math_results_df=thz_single_orbit_math_results_df,
                                   thz_sim_results_df=thz_single_orbit_sim_results_df,
                                   mmwave_math_results_df=mmwave_single_orbit_math_results_df,
                                   mmwave_sim_results_df=mmwave_single_orbit_sim_results_df)

    single_orbit_plotter.journal_single_capacity_vs_number_of_satellites_v2(file_ext='.pdf', show=True,
                                                                            thz_alpha_list=[1, 3, 5],
                                                                            mmwave_alpha_list=[5, 10, 30])

    single_orbit_plotter.journal_single_sinr_vs_number_of_satellites_v2(file_ext='.pdf', show=True,
                                                                     thz_alpha_list=[1, 3, 5],
                                                                     mmwave_alpha_list=[5, 10, 30])


def coplanar_orbits_sir_figures_6a_6b_6c():
    # Create Analysis object
    coplanar_orbits_analysis = Analysis(coplanar_sir_interfernce_pdf_input_params,
                                        coplanar_sir_interfernce_pdf_orbital_params,
                                        results_columns)

    # Compute or cache Analysis results
    coplanar_orbits_analysis.coplanar_higher_orbits_math()
    coplanar_orbits_analysis.coplanar_higher_orbits_simulation()
    coplanar_orbits_analysis.coplanar_orbits_math()
    coplanar_orbits_analysis.coplanar_orbits_simulation()

    # Load results
    math_higher_results_path = os.path.join(cache_path, coplanar_sir_interfernce_pdf_orbital_params[
        'math_results_higher_extension'] +
                                            '_higher_orbit_only_' + str(
        coplanar_sir_interfernce_pdf_orbital_params['higher_orbit_only'])
                                            + coplanar_sir_interfernce_pdf_input_params['extension'])
    sim_higher_results_path = os.path.join(cache_path,
                                           coplanar_sir_interfernce_pdf_orbital_params['sim_results_higher_extension'] +
                                           '_higher_orbit_only_' + str(
                                               coplanar_sir_interfernce_pdf_orbital_params['higher_orbit_only'])
                                           + coplanar_sir_interfernce_pdf_input_params['extension'])
    math_lower_results_path = os.path.join(cache_path,
                                           coplanar_sir_interfernce_pdf_orbital_params['math_results_extension'] +
                                           '_higher_orbit_only_' + str(
                                               coplanar_sir_interfernce_pdf_orbital_params['higher_orbit_only'])
                                           + coplanar_sir_interfernce_pdf_input_params['extension'])
    sim_lower_results_path = os.path.join(cache_path,
                                          coplanar_sir_interfernce_pdf_orbital_params['sim_results_extension'] +
                                          '_higher_orbit_only_' + str(
                                              coplanar_sir_interfernce_pdf_orbital_params['higher_orbit_only'])
                                          + coplanar_sir_interfernce_pdf_input_params['extension'])

    coplanar_orbits_math_results_df_higher = load_files(math_higher_results_path)
    coplanar_orbits_sim_results_df_higher = load_files(sim_higher_results_path)
    coplanar_orbits_math_results_df_lower = load_files(math_lower_results_path)
    coplanar_orbits_sim_results_df_lower = load_files(sim_lower_results_path)

    # Plotter object
    coplanar_orbits_plotter = Plotter(results_columns,
                                      link_budget_input_params=coplanar_sir_interfernce_pdf_input_params,
                                      orbital_input_params=coplanar_sir_interfernce_pdf_orbital_params,
                                      math_results_df=coplanar_orbits_math_results_df_lower,
                                      sim_results_df=coplanar_orbits_sim_results_df_lower,
                                      high_math_results_df=coplanar_orbits_math_results_df_higher,
                                      high_sim_results_df=coplanar_orbits_sim_results_df_higher)

    coplanar_orbits_plotter.journal_coplanar_pdf_interference_time(file_ext='.pdf', alpha_list=[30, 20, 10],
                                                                   n_sats_list=[50, 100], bins=200)
    coplanar_orbits_plotter.journal_coplanar_sir_vs_time_higher_vs_lower(file_ext='_alpha10.pdf', alpha_list=[10],
                                                                         n_sats_list=[50, 100])
    coplanar_orbits_plotter.journal_coplanar_sir_vs_time_higher_vs_lower(file_ext='_alpha30.pdf', alpha_list=[30],
                                                                         n_sats_list=[50, 100])


def coplanar_orbits_sinr_capacity_figures_7a_7b():
    # Create Analysis object
    thz_coplanar_orbits_analysis = Analysis(thz_input_params, coplanar_orbits_params, results_columns)
    mmwave_coplanar_orbits_analysis = Analysis(mmwave_input_params, coplanar_orbits_params, results_columns)

    # Compute or cache Analysis results
    thz_coplanar_orbits_analysis.coplanar_orbits_math()
    # thz_coplanar_orbits_analysis.coplanar_orbits_simulation()
    mmwave_coplanar_orbits_analysis.coplanar_orbits_math()
    # mmwave_coplanar_orbits_analysis.coplanar_orbits_simulation()

    # Load results
    thz_math_results_path = os.path.join(cache_path, coplanar_orbits_params['math_results_extension'] +
                                         '_higher_orbit_only_' + str(coplanar_orbits_params['higher_orbit_only']) +
                                         thz_input_params['extension'])
    # thz_sim_results_path = os.path.join(cache_path, coplanar_orbits_params['sim_results_extension'] +
    #                                     '_higher_orbit_only_' + str(coplanar_orbits_params['higher_orbit_only']) +
    #                                     thz_input_params['extension'])
    mmwave_math_results_path = os.path.join(cache_path, coplanar_orbits_params['math_results_extension'] +
                                            '_higher_orbit_only_' + str(coplanar_orbits_params['higher_orbit_only']) +
                                            mmwave_input_params['extension'])
    # mmwave_sim_results_path = os.path.join(cache_path, coplanar_orbits_params['sim_results_extension'] +
    #                                        '_higher_orbit_only_' + str(coplanar_orbits_params['higher_orbit_only']) +
    #                                        mmwave_input_params['extension'])

    thz_coplanar_orbits_math_results_df = load_files(thz_math_results_path)
    # thz_coplanar_orbits_sim_results_df = load_files(thz_sim_results_path)
    mmwave_coplanar_orbits_math_results_df = load_files(mmwave_math_results_path)
    # mmwave_coplanar_orbits_sim_results_df = load_files(mmwave_sim_results_path)

    # Create plotter object
    coplanar_orbits_plotter = Plotter(results_columns,
                                      thz_input_params=thz_input_params,
                                      mmwave_input_params=mmwave_input_params,
                                      orbital_input_params=coplanar_orbits_params,
                                      thz_math_results_df=thz_coplanar_orbits_math_results_df,
                                      # thz_sim_results_df=thz_coplanar_orbits_sim_results_df,
                                      # mmwave_sim_results_df=mmwave_coplanar_orbits_sim_results_df,
                                      mmwave_math_results_df=mmwave_coplanar_orbits_math_results_df)

    coplanar_orbits_plotter.journal_coplanar_sinr_vs_orbit_separation(file_ext='.pdf',
                                                                      thz_alpha_list=[1, 3],
                                                                      mmwave_alpha_list=[10, 30],
                                                                      n_sats_list=[100, 50])
    coplanar_orbits_plotter.journal_coplanar_capacity_vs_orbit_separation(file_ext='.pdf',
                                                                          thz_alpha_list=[1, 3],
                                                                          mmwave_alpha_list=[10, 30],
                                                                          n_sats_list=[100, 50])


def coplanar_orbits_vs_n_sats_figures():
    # Create Analysis object
    thz_coplanar_orbits_analysis = Analysis(thz_complete_input_params, coplanar_sir_interfernce_pdf_orbital_params, results_columns)
    mmwave_coplanar_orbits_analysis = Analysis(mmwave_complete_input_params, coplanar_sir_interfernce_pdf_orbital_params, results_columns)

    # Compute or cache Analysis results
    thz_coplanar_orbits_analysis.coplanar_orbits_math()
    # thz_coplanar_orbits_analysis.coplanar_orbits_simulation()
    mmwave_coplanar_orbits_analysis.coplanar_orbits_math()
    # mmwave_coplanar_orbits_analysis.coplanar_orbits_simulation()

    # Load results
    thz_math_results_path = os.path.join(cache_path, coplanar_orbits_params['math_results_extension'] +
                                         '_higher_orbit_only_' + str(coplanar_orbits_params['higher_orbit_only']) +
                                         thz_complete_input_params['extension'])
    # thz_sim_results_path = os.path.join(cache_path, coplanar_orbits_params['sim_results_extension'] +
    #                                     '_higher_orbit_only_' + str(coplanar_orbits_params['higher_orbit_only']) +
    #                                     thz_input_params['extension'])
    mmwave_math_results_path = os.path.join(cache_path, coplanar_orbits_params['math_results_extension'] +
                                            '_higher_orbit_only_' + str(coplanar_orbits_params['higher_orbit_only']) +
                                            mmwave_complete_input_params['extension'])
    # mmwave_sim_results_path = os.path.join(cache_path, coplanar_orbits_params['sim_results_extension'] +
    #                                        '_higher_orbit_only_' + str(coplanar_orbits_params['higher_orbit_only']) +
    #                                        mmwave_input_params['extension'])

    thz_coplanar_orbits_math_results_df = load_files(thz_math_results_path)
    # thz_coplanar_orbits_sim_results_df = load_files(thz_sim_results_path)
    mmwave_coplanar_orbits_math_results_df = load_files(mmwave_math_results_path)
    # mmwave_coplanar_orbits_sim_results_df = load_files(mmwave_sim_results_path)

    # Create plotter object
    coplanar_orbits_plotter = Plotter(results_columns,
                                      thz_input_params=thz_complete_input_params,
                                      mmwave_input_params=mmwave_complete_input_params,
                                      orbital_input_params=coplanar_sir_interfernce_pdf_orbital_params,
                                      thz_math_results_df=thz_coplanar_orbits_math_results_df,
                                      # thz_sim_results_df=thz_coplanar_orbits_sim_results_df,
                                      # mmwave_sim_results_df=mmwave_coplanar_orbits_sim_results_df,
                                      mmwave_math_results_df=mmwave_coplanar_orbits_math_results_df)

    coplanar_orbits_plotter.coplanar_sir_snr_sinr_capacity_vs_n_sats_plot()


def shifted_orbits_sir_vs_time_figures_8a():
    # Create Analysis object
    shifted_orbits_analysis = Analysis(shifted_sir_figures_input_params, shifted_sir_figures_orbital_parameters,
                                       results_columns)

    # Compute or cache Analysis results
    shifted_orbits_analysis.shifted_orbits_math()
    shifted_orbits_analysis.shifted_orbits_simulation()

    # Load results
    math_results_path = os.path.join(cache_path, shifted_sir_figures_orbital_parameters['math_results_extension'] +
                                     '_other_orbit_only_' +
                                     str(shifted_sir_figures_orbital_parameters['other_orbit_only']) +
                                     shifted_sir_figures_input_params['extension'])
    sim_results_path = os.path.join(cache_path, shifted_sir_figures_orbital_parameters['sim_results_extension'] +
                                    '_other_orbit_only_' +
                                    str(shifted_sir_figures_orbital_parameters['other_orbit_only']) +
                                    shifted_sir_figures_input_params['extension'])
    shifted_orbits_math_results_df = load_files(math_results_path)
    shifted_orbits_sim_results_df = load_files(sim_results_path)

    # Create plotter object
    shifted_orbits_plotter = Plotter(results_columns,
                                     link_budget_input_params=shifted_sir_figures_input_params,
                                     orbital_input_params=shifted_sir_figures_orbital_parameters,
                                     math_results_df=shifted_orbits_math_results_df,
                                     sim_results_df=shifted_orbits_sim_results_df)

    # shifted_orbits_plotter.journal_shifted_pdf(alpha_list=shifted_sir_figures_orbital_parameters['alpha_lims'][0],
    #                                            inclination=3)
    shifted_orbits_plotter.journal_shifted_sir_vs_time(file_ext='.pdf',
                                                       alpha_list=shifted_sir_figures_orbital_parameters['alpha_lims'][
                                                           0], inclination=3)


def shifted_orbits_pdf_figure():
    # Create Analysis object
    shifted_orbits_analysis = Analysis(shifted_sir_figures_input_params, shifted_interference_pdf_orbital_parameters,
                                       results_columns)

    # Compute or cache Analysis results
    shifted_orbits_analysis.shifted_orbits_pdf()
    shifted_orbits_analysis.shifted_orbits_simulation()

    # Load results
    math_results_path = os.path.join(cache_path, shifted_interference_pdf_orbital_parameters['math_results_extension'] +
                                     '_other_orbit_only_' +
                                     str(shifted_interference_pdf_orbital_parameters['other_orbit_only']) +
                                     shifted_sir_figures_input_params['extension'])
    sim_results_path = os.path.join(cache_path, shifted_interference_pdf_orbital_parameters['sim_results_extension'] +
                                    '_other_orbit_only_' +
                                    str(shifted_interference_pdf_orbital_parameters['other_orbit_only']) +
                                    shifted_sir_figures_input_params['extension'])
    shifted_orbits_math_results_df = load_files(math_results_path)
    shifted_orbits_sim_results_df = load_files(sim_results_path)

    # Create plotter object
    shifted_orbits_plotter = Plotter(results_columns,
                                     link_budget_input_params=shifted_sir_figures_input_params,
                                     orbital_input_params=shifted_interference_pdf_orbital_parameters,
                                     math_results_df=shifted_orbits_math_results_df,
                                     sim_results_df=shifted_orbits_sim_results_df)

    shifted_orbits_plotter.journal_shifted_sir_vs_time(file_ext='.pdf',
                                                       alpha_list=shifted_sir_figures_orbital_parameters['alpha_lims'][
                                                           0], inclination=3)
    shifted_orbits_plotter.journal_shifted_pdf(alpha_list=shifted_interference_pdf_orbital_parameters['alpha_lims'][0],
                                               inclination=3)


def shifted_orbits_sinr_figures_8b():
    # Create Analysis object
    thz_shifted_orbits_analysis = Analysis(shifted_sinr_thz_input_params, shifted_sinr_figure_orbital_parameters,
                                           results_columns)
    mmwave_shifted_orbits_analysis = Analysis(shifted_sinr_mmwave_input_params, shifted_sinr_figure_orbital_parameters,
                                              results_columns)

    # Compute or cache Analysis results
    thz_shifted_orbits_analysis.shifted_orbits_math()
    mmwave_shifted_orbits_analysis.shifted_orbits_math()

    # Load results
    thz_results_path = os.path.join(cache_path, shifted_sinr_figure_orbital_parameters['math_results_extension'] +
                                    '_other_orbit_only_' +
                                    str(shifted_sinr_figure_orbital_parameters['other_orbit_only']) +
                                    shifted_sinr_thz_input_params['extension'])
    mmwave_results_path = os.path.join(cache_path, shifted_sinr_figure_orbital_parameters['math_results_extension'] +
                                       '_other_orbit_only_' +
                                       str(shifted_sinr_figure_orbital_parameters['other_orbit_only']) +
                                       shifted_sinr_mmwave_input_params['extension'])
    shifted_orbits_thz_results_df = load_files(thz_results_path)
    shifted_orbits_mmwave_results_df = load_files(mmwave_results_path)

    # Create plotter object
    shifted_orbits_plotter = Plotter(results_columns,
                                     thz_input_params=shifted_sinr_thz_input_params,
                                     mmwave_input_params=shifted_sinr_mmwave_input_params,
                                     orbital_input_params=shifted_sinr_figure_orbital_parameters,
                                     thz_math_results_df=shifted_orbits_thz_results_df,
                                     mmwave_math_results_df=shifted_orbits_mmwave_results_df)

    shifted_orbits_plotter.journal_shifted_sinr_vs_beamwidth(file_ext='.pdf', inclination=3)


def shifted_orbits_capacity_figures():
    # Create Analysis object
    thz_shifted_orbits_analysis = Analysis(thz_input_params, shifted_capacity_figure_orbital_parameters,
                                           results_columns)
    mmwave_shifted_orbits_analysis = Analysis(mmwave_input_params, shifted_capacity_figure_orbital_parameters,
                                              results_columns)

    # Compute or cache Analysis results
    thz_shifted_orbits_analysis.shifted_orbits_math()
    mmwave_shifted_orbits_analysis.shifted_orbits_math()

    # Load results
    thz_results_path = os.path.join(cache_path, shifted_capacity_figure_orbital_parameters['math_results_extension'] +
                                    '_other_orbit_only_' +
                                    str(shifted_capacity_figure_orbital_parameters['other_orbit_only']) +
                                    thz_input_params['extension'])
    mmwave_results_path = os.path.join(cache_path,
                                       shifted_capacity_figure_orbital_parameters['math_results_extension'] +
                                       '_other_orbit_only_' +
                                       str(shifted_capacity_figure_orbital_parameters['other_orbit_only']) +
                                       mmwave_input_params['extension'])
    shifted_orbits_thz_results_df = load_files(thz_results_path)
    shifted_orbits_mmwave_results_df = load_files(mmwave_results_path)

    # Create plotter object
    shifted_orbits_plotter = Plotter(results_columns,
                                     thz_input_params=thz_input_params,
                                     mmwave_input_params=mmwave_input_params,
                                     orbital_input_params=shifted_capacity_figure_orbital_parameters,
                                     thz_math_results_df=shifted_orbits_thz_results_df,
                                     mmwave_math_results_df=shifted_orbits_mmwave_results_df)

    shifted_orbits_plotter.journal_shifted_capacity_vs_inclination(file_ext='.pdf', mmwave_alpha_list=[10, 30],
                                                                   thz_alpha_list=[1, 3])


def shifted_orbits_capacity_n_sats_figure_8c():
    # Create Analysis object
    thz_shifted_orbits_analysis = Analysis(shifted_capacity_thz_input_params,
                                           shifted_capacity_n_sats_figure_orbital_parameters,
                                           results_columns)
    mmwave_shifted_orbits_analysis = Analysis(shifted_capacity_mmwave_input_params,
                                              shifted_capacity_n_sats_figure_orbital_parameters,
                                              results_columns)

    # Compute or cache Analysis results
    thz_shifted_orbits_analysis.shifted_orbits_math()
    mmwave_shifted_orbits_analysis.shifted_orbits_math()

    # Load results
    thz_results_path = os.path.join(cache_path,
                                    shifted_capacity_n_sats_figure_orbital_parameters['math_results_extension'] +
                                    '_other_orbit_only_' +
                                    str(shifted_capacity_n_sats_figure_orbital_parameters['other_orbit_only']) +
                                    shifted_capacity_thz_input_params['extension'])
    mmwave_results_path = os.path.join(cache_path,
                                       shifted_capacity_n_sats_figure_orbital_parameters['math_results_extension'] +
                                       '_other_orbit_only_' +
                                       str(shifted_capacity_n_sats_figure_orbital_parameters['other_orbit_only']) +
                                       shifted_capacity_mmwave_input_params['extension'])
    shifted_orbits_thz_results_df = load_files(thz_results_path)
    shifted_orbits_mmwave_results_df = load_files(mmwave_results_path)

    # Create plotter object
    shifted_orbits_plotter = Plotter(results_columns,
                                     thz_input_params=shifted_capacity_thz_input_params,
                                     mmwave_input_params=shifted_capacity_mmwave_input_params,
                                     orbital_input_params=shifted_capacity_n_sats_figure_orbital_parameters,
                                     thz_math_results_df=shifted_orbits_thz_results_df,
                                     mmwave_math_results_df=shifted_orbits_mmwave_results_df)

    shifted_orbits_plotter.journal_shifted_capacity_vs_number_of_satellites(file_ext='.pdf')


def complete_constellation_figure_9():
    # Combination of all three scenarios. One constellation with multiple shifted orbits forming a shell and a coplanar
    # orbit in the orbital plane of the receiver of interest
    thz_complete_constellation_analysis = Analysis(thz_complete_input_params,
                                                   complete_constellation_orbital_parameters,
                                                   results_columns)
    mmwave_complete_constellation_analysis = Analysis(mmwave_complete_input_params,
                                                      complete_constellation_orbital_parameters,
                                                      results_columns)

    # Compute or cache analysis results
    thz_complete_constellation_analysis.complete_constellation_math()
    mmwave_complete_constellation_analysis.complete_constellation_math()

    # Load results
    thz_results_path = os.path.join(cache_path,
                                    complete_constellation_orbital_parameters['math_results_extension'] +
                                    thz_complete_input_params['extension'])
    mmwave_results_path = os.path.join(cache_path,
                                       complete_constellation_orbital_parameters['math_results_extension'] +
                                       mmwave_complete_input_params['extension'])
    thz_results_df = load_files(thz_results_path)
    mmwave_results_df = load_files(mmwave_results_path)

    # Create plotter object
    complete_constellation_plotter = Plotter(results_columns,
                                             thz_input_params=thz_complete_input_params,
                                             mmwave_input_params=mmwave_complete_input_params,
                                             orbital_input_params=complete_constellation_orbital_parameters,
                                             thz_math_results_df=thz_results_df,
                                             mmwave_math_results_df=mmwave_results_df)

    # complete_constellation_plotter.complete_constellation_sir_snr_sinr_capacity_n_sats()
    # complete_constellation_plotter.complete_constellation_n_interferers_vs_n_sats()
    complete_constellation_plotter.complete_constellation_capacity_vs_n_sats_v2(file_ext='.pdf')


if __name__ == '__main__':
    # single_orbit_sir_vs_nsats_figure_5a()
    # single_orbit_sinr_capacity_figures_5b_5c()
    # coplanar_orbits_sir_figures_6a_6b_6c()
    # coplanar_orbits_sinr_capacity_figures_7a_7b()
    # shifted_orbits_sir_vs_time_figures_8a()
    # shifted_orbits_pdf_figure()
    # shifted_orbits_sinr_figures_8b()
    # shifted_orbits_capacity_figures()
    # shifted_orbits_capacity_n_sats_figure_8c()
    complete_constellation_figure_9()
    # coplanar_orbits_vs_n_sats_figures()
