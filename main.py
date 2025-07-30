from simulator import toolkit as tk
from numpy import random
import random

import numpy as np
from math import log
from datetime import datetime, timedelta
from simulator.link_budget import link_budget
import pandas as pd
from simulator.simulation import Simulation
from simulator.satellite import Constellation
from simulator import Channel


def THz_Ka_dual_band_const_paper():
    # tk.welcome_message()

    """
    Ground Station Setup:
    GS_X : Ground Station in rural area with no internet access
    GS_Y : Ground Station with internet access scattered around the globe.
    GS_Longitude = [-103.5101597, -118.2923, -69.4464, 88.7879 ]
    GS_Latitude = [43.6726477,  36.5785, -14.6312, 30.1534]
    """
    ## Arbitray Locations of GS
    GS_Longitude = [-103.5101597, 88.7879, -119.7871, -78.6382, -95.9345, -106.37, 37.6173, 27.1428, 141.3545, 39.10,
                    36.8219, 51.6660, 44.5152]
    GS_Latitude = [43.6726477, 30.1534, 36.7378, 35.7796, 41.2565, 35.2, 55.7558, 38.4237, 43.0618, 21.42, -1.2921,
                   32.6539, 40.1872]
    GS_Altitude = [1.620,
                   5.011,
                   0.1,
                   0.134,
                   0.362,
                   1.619,
                   0.157,
                   0.029,
                   0.026,
                   0.015,
                   1.79,
                   1.60,
                   1.00]  # in km, from earth.google.com
    GS_Location_Name = ['Custer County, South Dakota ',
                        'China, Tibet',
                        'Fresno, California',
                        'Raleigh North Carolina',
                        'Omaha Nebraska',
                        'Albuquerque ,New Mexico',
                        'Moscow Russia',
                        'Izmir Turkey',
                        'Sapporo Japan',
                        'Jeddah Saudi Arabia ',
                        'Nairobi Kenya',
                        'Esfahan Iran',
                        'yeravan armenia'
                        ]

    number_of_GS_X = 1
    number_of_GS_Y = 1
    GS_Group = tk.setup_GS(GS_Longitude, GS_Latitude, GS_Altitude, GS_Location_Name, len(GS_Altitude), len(GS_Altitude))
    sat_Group = tk.setup_sats_from_tle("TLE.txt")
    print("Number of Satellites: ", len(sat_Group))
    print("Number of Earth-Stations: ", len(GS_Group))

    """
    Model Earth Station with no internet access transmission process as a Poisson process
    """
    sim_number = 1

    for sim_index in range(sim_number):
        simulation_time_period = (60 * 4) * 60  # 2 minutes
        simulation_time_step = 1  # second
        current_time_UTC, year, month, day, hour, minutes, seconds = tk.get_time(0)
        transmission_time = [current_time_UTC]
        lambda_ = 0.001  # packets/per unit time
        # s = 5 # 5 minute interval
        i = 0
        # Random transmit times, modeled as poisson process
        while (transmission_time[i] - current_time_UTC).total_seconds() < (simulation_time_period * 60):
            i = i + 1
            u = random.uniform(0, 1)
            time_diff = -1 * (1 / lambda_) * log(1 - u)
            transmission_time.append(transmission_time[i - 1] + timedelta(minutes=time_diff))

        print("length", len(transmission_time))

        latency = []  # in ms
        data_rates_UL_THz = []
        data_rates_DL_THz = []
        cross_link_Thz = []
        cross_link_KA = []
        data_rates_UL_ku = []
        data_rates_DL_ku = []
        cross_link_thz_avg = []
        cross_link_ka_avg = []
        cross_link_thz_avg_val = 0
        cross_link_ka_avg_val = 0

        packet_drops = 0

        # Transmission of packets at each transmission time
        for i in range(1, len(transmission_time)):
            random_src_trg = random.sample(range(0, len(GS_Altitude)), 2)
            print("Source Location is : " + GS_Group[random_src_trg[0]].name)
            print("Target Location is : " + GS_Group[random_src_trg[1]].name)
            src_gs = GS_Group[random_src_trg[0]]
            target_gs = GS_Group[random_src_trg[1]]
            latency_adder = 0
            print("Transmission time num : ", i, "out of: ", len(transmission_time))
            tx_time_formatted = tk.format_timeadvance(transmission_time[i])
            for current_sat in sat_Group:
                current_sat.update_sat_SSP_predetermined_time(tx_time_formatted)
                for gs_up in GS_Group:
                    gs_up.check_coverage(current_sat)
                    gs_up.update_ULSAT()

            if not src_gs.can_transmit_UL() or not target_gs.can_transmit_UL():
                packet_drops = packet_drops + 1
                print("Packet Drop Number = " + str(packet_drops))
                continue

            # Routing between tx sat_UL and rx sat_UL
            route, trgt_to_gs_distance = tk.short_interlink_route(src_gs.UL_sat, target_gs.UL_sat, src_gs, target_gs,
                                                                  sat_Group)
            if route is None or trgt_to_gs_distance == 0:
                packet_drops = packet_drops + 1
                print("Packet Drop Number = " + str(packet_drops))
                continue

            # map_api.plot_route(route, src_gs, target_gs, sat_Group,str(i))

            # Computing data rates of every CROSS link (through link budget)
            for route_counter in range(len(route.route_distance)):
                if route_counter == 0:  # Ul
                    continue
                power_rec_CL_THZ, data_rate_CL_THZ = link_budget(0.2, route.route_distance[
                    route_counter] * 10 ** 3, 40e9, 47.85 * 2, 295e9, 0, 0)
                power_rec_CL_ka, data_rate_CL_ka = link_budget(5, route.route_distance[
                    route_counter] * 10 ** 3, 2.25e9, 26.88 * 2, 26.375e9, 0, 0)
                cross_link_Thz.append(data_rate_CL_THZ)
                cross_link_KA.append(data_rate_CL_ka)
                cross_link_thz_avg_val = cross_link_thz_avg_val + data_rate_CL_THZ
                cross_link_ka_avg_val = cross_link_ka_avg_val + data_rate_CL_ka
            latency.append(route.total_distance + trgt_to_gs_distance)

            # Computing data rates of UL and DL link (through link budget)
            ele_ul, azi_ul = src_gs.elevation_and_azimuth()
            ele_dl, azi_dl = target_gs.elevation_and_azimuth()

            thz_spreading, thz_UL_abs = Channel.path_loss(src_gs.UL_sat.altitude, [217.5], ele_ul, 0.25,
                                                          src_gs.location[2])
            thz_spreading, thz_DL_abs = Channel.path_loss(target_gs.UL_sat.altitude, [295], ele_dl, 0.25,
                                                          target_gs.location[2])

            thz_power, thz_data_rate_ul = link_budget(0.2, src_gs._min_distance_to_sat * 10 ** 3, 17e9, 69.29 + 45.2,
                                                      217.5e9, thz_UL_abs[0], thz_spreading[0])
            thz_power, thz_data_rate_dl = link_budget(0.2, target_gs._min_distance_to_sat * 10 ** 3, 40e9,
                                                      71.94 + 47.85, 295e9, thz_DL_abs[0], thz_spreading[0])

            ka_power, ka_data_rate_ul = link_budget(5, src_gs._min_distance_to_sat * 10 ** 3, 2.5e9, 51.71 + 27.63,
                                                    28.75e9, 0, thz_spreading[0])
            ka_power, ka_data_rate_dl = link_budget(5, target_gs._min_distance_to_sat * 10 ** 3, 5e9, 54.58 + 30.5,
                                                    40e9, 0, thz_spreading[0])

            data_rates_UL_THz.append(thz_data_rate_ul)
            data_rates_UL_ku.append(ka_data_rate_ul)

            data_rates_DL_THz.append(thz_data_rate_dl)
            data_rates_DL_ku.append(ka_data_rate_dl)

            # latency.append(  ((total_distance_for_route*1000) / (C.SPEED_OF_LIGHT)) *1000   )

        # Results processing
        while len(data_rates_UL_THz) != len(cross_link_Thz):
            data_rates_UL_THz.append(0)
            data_rates_DL_THz.append(0)
            data_rates_UL_ku.append(0)
            data_rates_DL_ku.append(0)
            latency.append(0)
            cross_link_thz_avg.append(0)
            cross_link_ka_avg.append(0)
        results = {'THZ-UL': data_rates_UL_THz,
                   'THZ-DL': data_rates_DL_THz,
                   'THZ-CL': cross_link_Thz,
                   'KA_UL': data_rates_UL_ku,
                   'KA_DL': data_rates_DL_ku,
                   'KA_CL': cross_link_KA,
                   'lat': latency,
                   }

        df = pd.DataFrame(results, columns=['THZ-UL', 'THZ-DL', 'THZ-CL', 'KA_UL', 'KA_DL', 'KA_CL', 'lat'])
        df.to_csv('results/results_' + str(sim_index) + 'simulation' + str(packet_drops) + 'drops.csv', index=False,
                  header=True)


def starlink_test():
    # Simulation initialization
    start_time = datetime(2022, 1, 1)
    sim = Simulation(t_start=start_time)
    np.random.seed(0)

    constellation = Constellation.starlink(sim)
    pairwise_distances = constellation.sat_pairwise_distance()
    print('Minimum and maximum intera-stellite distances at time {}: {}km / {}km'.
          format(sim.t_current.strftime('%m/%d/%Y, %H:%M:%S'), round(min(pairwise_distances), 2),
                 round(max(pairwise_distances), 2)))


if __name__ == '__main__':
    starlink_test()
    # plot_utils.sat_path_image([0,10,20,30],[0,10,20,30], os.path.join(os.getcwd(), 'test.jpg'))
    print('DONE')
