import os.path
from simulator.file_utils import load_files
from analysis import Analysis, cache_path
from plotter import Plotter
import numpy as np

"""Management script

Main code creating instances of the Analysis and Plotter classes.
The simulations input parameters are defined here.
"""

# THz
thz_input_params = {'p_tx_dbm': 27,  # W = 10 ** ((P(dBm)-30)/10)
                    'bandwidth': 10e9,
                    'T_system': 100,
                    'center_frequency': 130e9,
                    'alpha_list': [1, 3, 5],
                    'extension': '_thz.dat'
                    }

# mmWave
mmWave_input_params = {'p_tx_dbm': 60,  # W = 10 ** ((P(dBm)-30)/10)
                       'bandwidth': 0.4e9,
                       'T_system': 100,
                       'center_frequency': 38e9,
                       'alpha_list': [3, 5, 10, 20, 30, 40],
                       'extension': '_mmWave.dat'
                       }

single_orbit_params = {'h_lims': [300 * 1e3, 2000.1 * 1e3],
                       'n_sats_lims': [10, 201],
                       # Math
                       'n_sats_resolution_math': 1,
                       'h_resolution_math': 10 * 1e3,  # height resolution
                       'math_results_extension': 'single_orbit_math',
                       'math_replace': True,  # Change to False to use cache files, if available
                       # Simulation
                       'n_sats_resolution_sim': 5,
                       'h_resolution_sim': 100 * 1e3,
                       'sim_results_extension': 'single_orbit_simulation',
                       'sim_replace': True  # Change to False to use cache files, if available
                       }

coplanar_orbits_parameters = {'h_low': 500e3,
                              'h_lims': [510 * 1e3, 700.1 * 1e3],
                              'n_sats_list': [20, 40, 50, 100, 200],  # Iterable for number of satellites in orbit
                              'higher_orbit_only': True,
                              # Math
                              'beta_steps_math': 100,
                              'h_resolution_math': 5e3,  # height resolution
                              'math_results_extension': 'coplanar_orbits_math',
                              'math_replace': False,  # Change to False to use cache files, if available
                              # Simulation
                              'beta_steps_sim': 100, # 20 before
                              'h_resolution_sim': 25e3,
                              'sim_results_extension': 'coplanar_orbits_simulation',
                              'sim_replace': False,  # Change to False to use cache files, if available
                              # Higher orbit analysis
                              'math_results_higher_extension': 'coplanar_higher_orbits_math',
                              'sim_results_higher_extension': 'coplanar_higher_orbits_simulation'
                              }

shifted_orbits_parameters = {'h_lims': [500 * 1e3, 500.1 * 1e3],
                             'n_sats_list': [20, 40, 60, 80, 100],  # Iterable for number of satellites in orbit
                             'alpha_lims': [1, 30.1],
                             'inclination_lims': [3, 30.1],
                             'RAA difference': 90,
                             # Math
                             'h_resolution_math': 10e3,
                             'alpha_resolution_math': 0.5,
                             'inclination_resolution_math': 1,
                             'beta_steps_math': 1,
                             'n_timesteps_math': 100,
                             'math_results_extension': 'shifted_orbits_math.dat',
                             'replace_math': False,
                             # Simulation
                             'h_resolution_sim': 10e3,
                             'alpha_resolution_sim': 2,
                             'inclination_resolution_sim': 4,
                             'beta_steps_sim': 1,
                             'n_timesteps_sim': 20,
                             'sim_results_extension': 'shifted_orbits_simulation.dat',
                             'replace_sim': True
                             }

journal_figure_1_parameters = {
    'common_input_params': {'p_tx_dbm': 60,  # W = 10 ** ((P(dBm)-30)/10)
                            'bandwidth': 0.4e9,
                            'T_system': 100,
                            'center_frequency': 38e9,
                            'extension': 'fig1_journal.dat'
                            },
    'single_orbit_params': {'h_lims': [300 * 1e3, 2000.1 * 1e3],
                            'alpha_lims': [1, 90],
                            'n_sats_list': [20, 40, 60, 80, 100],
                            # Math
                            'alpha_resolution_math': 1,
                            'h_resolution_math': 10 * 1e3,  # height resolution
                            'math_results_extension': 'single_orbit_math',
                            'math_replace': True,  # Change to False to use cache files, if available
                            # Simulation
                            'alpha_resolution_sim': 5,
                            'h_resolution_sim': 100 * 1e3,
                            'sim_results_extension': 'single_orbit_simulation',
                            'sim_replace': True  # Change to False to use cache files, if available
                            }
}

results_columns = ['tx_rx_distance', 'rx_power', 'rx_power_linear', 'd_interferer', 'interference_linear',
                   'interference', 'n_interferers', 'sir_linear', 'snr_linear', 'sinr_linear', 'capacity_gbps']


def single_orbit():
    # Create Analysis object
    thz_single_orbit_analysis = Analysis(thz_input_params, single_orbit_params, results_columns)
    mmWave_single_orbit_analysis = Analysis(mmWave_input_params, single_orbit_params, results_columns)

    # Compute or cache Analysis results
    thz_single_orbit_analysis.single_orbit_math()
    thz_single_orbit_analysis.single_orbit_simulation()
    mmWave_single_orbit_analysis.single_orbit_math()
    mmWave_single_orbit_analysis.single_orbit_simulation()

    # Load results
    thz_math_results_path = os.path.join(cache_path,
                                         single_orbit_params['math_results_extension'] + thz_input_params['extension'])
    thz_sim_results_path = os.path.join(cache_path,
                                        single_orbit_params['sim_results_extension'] + thz_input_params['extension'])
    mmWave_math_results_path = os.path.join(cache_path,
                                            single_orbit_params['math_results_extension'] + mmWave_input_params[
                                                'extension'])
    mmWave_sim_results_path = os.path.join(cache_path,
                                           single_orbit_params['sim_results_extension'] + mmWave_input_params[
                                               'extension'])
    thz_single_orbit_math_results_df = load_files(thz_math_results_path)
    thz_single_orbit_sim_results_df = load_files(thz_sim_results_path)
    mmWave_single_orbit_math_results_df = load_files(mmWave_math_results_path)
    mmWave_single_orbit_sim_results_df = load_files(mmWave_sim_results_path)
    '''
    # Create plotter object
    single_orbit_plotter = Plotter(results_columns,
                                   thz_input_params=thz_input_params,
                                   mmWave_input_params=mmWave_input_params,
                                   especial_input_params=single_orbit_params,
                                   thz_math_results_df=thz_single_orbit_math_results_df,
                                   thz_sim_results_df=thz_single_orbit_sim_results_df,
                                   mmWave_math_results_df=mmWave_single_orbit_math_results_df,
                                   mmWave_sim_results_df=mmWave_single_orbit_sim_results_df)

    single_orbit_plotter.journal_sir_snr_sinr_capacity_vs_number_of_satellites(file_ext='.jpg', show=True, h=500e3,
                                                                               thz_alpha_list=[1, 3, 5],
                                                                               mmWave_alpha_list=[1, 20, 30, 40])

    '''
    # Create plotter object
    single_orbit_plotter = Plotter(results_columns,
                                   common_input_params=mmWave_input_params,
                                   especial_input_params=single_orbit_params,
                                   math_results_df=mmWave_single_orbit_math_results_df,
                                   sim_results_df=mmWave_single_orbit_sim_results_df)

    single_orbit_plotter.journal_single_sir_vs_number_of_satellites()


def coplanar_orbits_mmwave():
    # Create Analysis object
    coplanar_orbits_analysis = Analysis(mmWave_input_params, coplanar_orbits_parameters, results_columns)

    #coplanar_orbits_analysis.coplanar_testing_plot_from_above()

    # Compute or cache Analysis results
    coplanar_orbits_analysis.coplanar_orbits_math()
    coplanar_orbits_analysis.coplanar_orbits_simulation()

    # Load results
    math_results_path = os.path.join(cache_path, coplanar_orbits_parameters['math_results_extension'] +
                                     '_higher_orbit_only_' + str(coplanar_orbits_parameters['higher_orbit_only'])
                                     + mmWave_input_params['extension'])
    sim_results_path = os.path.join(cache_path, coplanar_orbits_parameters['sim_results_extension'] +
                                     '_higher_orbit_only_' + str(coplanar_orbits_parameters['higher_orbit_only'])
                                     + mmWave_input_params['extension'])
    coplanar_orbits_math_results_df = load_files(math_results_path)
    coplanar_orbits_sim_results_df = load_files(sim_results_path)

    # Create plotter object
    coplanar_orbits_plotter = Plotter(results_columns, common_input_params=mmWave_input_params,
                                      especial_input_params=coplanar_orbits_parameters,
                                      math_results_df=coplanar_orbits_math_results_df,
                                      sim_results_df=coplanar_orbits_sim_results_df)

    coplanar_orbits_plotter.coplanar_n_interferers_vs_time_plot(separation_index=6, n_sats_list=[20, 40])
    coplanar_orbits_plotter.coplanar_interference_vs_time_plot(separation_index=6, n_sats_list=[20])
    coplanar_orbits_plotter.n_interferers_vs_orbit_separation_plot(n_sats_list=[20, 40])
    coplanar_orbits_plotter.interference_vs_orbit_separation_plot(n_sats_list=[20, 40])
    # coplanar_orbits_plotter.coplanar_interference_vs_time_histogram(alpha_list=mmWave_input_params['alpha_list'])
    coplanar_orbits_plotter.coplanar_sir_snr_sinr_capacity_vs_orbit_separation_plot(alpha_list=mmWave_input_params['alpha_list'])


def coplanar_orbits_thz():
    # Create Analysis object
    coplanar_orbits_analysis = Analysis(thz_input_params, coplanar_orbits_parameters, results_columns)

    # Compute or cache Analysis results
    coplanar_orbits_analysis.coplanar_orbits_math()
    coplanar_orbits_analysis.coplanar_orbits_simulation()

    # Load results
    math_results_path = os.path.join(cache_path, coplanar_orbits_parameters['math_results_extension'] +
                                     '_higher_orbit_only_' + str(coplanar_orbits_parameters['higher_orbit_only'])
                                     + thz_input_params['extension'])
    sim_results_path = os.path.join(cache_path, coplanar_orbits_parameters['sim_results_extension'] +
                                     '_higher_orbit_only_' + str(coplanar_orbits_parameters['higher_orbit_only'])
                                     + thz_input_params['extension'])
    coplanar_orbits_math_results_df = load_files(math_results_path)
    coplanar_orbits_sim_results_df = load_files(sim_results_path)

    # Create plotter object
    coplanar_orbits_plotter = Plotter(results_columns, common_input_params=thz_input_params,
                                      especial_input_params=coplanar_orbits_parameters,
                                      math_results_df=coplanar_orbits_math_results_df,
                                      sim_results_df=coplanar_orbits_sim_results_df)

    coplanar_orbits_plotter.coplanar_interference_vs_time_histogram()
    coplanar_orbits_plotter.coplanar_sir_snr_sinr_capacity_vs_orbit_separation_plot()


def coplanar_orbits_higher_orbit():
    # Create Analysis object
    coplanar_orbits_analysis = Analysis(common_input_params, coplanar_orbits_parameters, results_columns)

    # Compute or cache Analysis results
    coplanar_orbits_analysis.coplanar_higher_orbits_math()
    coplanar_orbits_analysis.coplanar_higher_orbits_simulation()
    coplanar_orbits_analysis.coplanar_orbits_math()
    coplanar_orbits_analysis.coplanar_orbits_simulation()

    # Load results
    math_higher_results_path = os.path.join(cache_path, coplanar_orbits_parameters['math_results_higher_extension'])
    sim_higher_results_path = os.path.join(cache_path, coplanar_orbits_parameters['sim_results_higher_extension'])
    math_lower_results_path = os.path.join(cache_path, coplanar_orbits_parameters['math_results_extension'])
    sim_lower_results_path = os.path.join(cache_path, coplanar_orbits_parameters['sim_results_extension'])

    coplanar_orbits_math_results_df_higher = load_files(math_higher_results_path)
    coplanar_orbits_sim_results_df_higher = load_files(sim_higher_results_path)
    coplanar_orbits_math_results_df_lower = load_files(math_lower_results_path)
    coplanar_orbits_sim_results_df_lower = load_files(sim_lower_results_path)

    # Create plotter object
    coplanar_orbits_comparison_plotter = Plotter(common_input_params, coplanar_orbits_parameters, results_columns,
                                                 math_results_df=coplanar_orbits_math_results_df_lower,
                                                 sim_results_df=coplanar_orbits_sim_results_df_lower,
                                                 high_math_results_df=coplanar_orbits_math_results_df_higher,
                                                 high_sim_results_df=coplanar_orbits_sim_results_df_higher)

    coplanar_orbits_comparison_plotter.coplanar_interference_vs_orbit_separation_higher_vs_lower_paper(file_ext='.jpg',
                                                                                                       show=True,
                                                                                                       alpha_list=[5,
                                                                                                                   30],
                                                                                                       n_sats_list=[50])
    coplanar_orbits_comparison_plotter.coplanar_sir_vs_time_paper_plot(file_ext='.jpg', show=True)
    coplanar_orbits_comparison_plotter.coplanar_interference_vs_orbit_separation_paper_plot(file_ext='.jpg', show=True)
    coplanar_orbits_comparison_plotter.coplanar_sir_vs_time_higher_vs_lower_paper(file_ext='.jpg', show=True,
                                                                                  alpha_list=[5, 30], n_sats_list=[50],
                                                                                  orbit_separation=5e3)


def shifted_orbits():
    # Create Analysis object
    shifted_orbits_analysis = Analysis(common_input_params, shifted_orbits_parameters, results_columns)

    # Compute or cache Analysis results
    shifted_orbits_analysis.shifted_orbits_math()
    shifted_orbits_analysis.shifted_orbits_simulation()

    # Load results
    math_results_path = os.path.join(cache_path, shifted_orbits_parameters['math_results_extension'])
    sim_results_path = os.path.join(cache_path, shifted_orbits_parameters['sim_results_extension'])
    shifted_orbits_math_results_df = load_files(math_results_path)
    shifted_orbits_sim_results_df = load_files(sim_results_path)

    # Create plotter object
    shifted_orbits_plotter = Plotter(common_input_params, shifted_orbits_parameters, results_columns,
                                     math_results_df=shifted_orbits_math_results_df,
                                     sim_results_df=shifted_orbits_sim_results_df)
    shifted_orbits_plotter.shifted_sir_vs_time_plot_paper(file_ext='.jpg', show=True, alpha_list=29)
    shifted_orbits_plotter.shifted_sir_vs_beamwidth_plot_paper(file_ext='.jpg', show=True, inclination=3)
    shifted_orbits_plotter.shifted_sir_vs_inclination_plot_paper(file_ext='.jpg', show=True, alpha_list=29)


if __name__ == '__main__':
    single_orbit()
    # coplanar_orbits_mmwave()
    # coplanar_orbits_thz()
    # shifted_orbits()
    # coplanar_orbits_higher_orbit()
    # journal_beamwidth_figure_single_orbit()
