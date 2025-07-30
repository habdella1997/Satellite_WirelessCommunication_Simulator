from numpy import random
from simulator import astrodynamics as AD, constants as C, toolkit as tk
from math import log
import math
from datetime import timedelta
import pandas as pd
import math_util as mu

GS_Longitude = [-71.096851]
GS_Latitude = [42.341869]
GS_Altitude = [1.620] #in km, from earth.google.com
GS_Location_Name = ['Custer County, South Dakota ' ]



number_of_GS_X = 1
number_of_GS_Y = 1
GS_Group = tk.setup_GS(GS_Longitude, GS_Latitude, GS_Altitude,GS_Location_Name, 1, 0 )
Sat_Group = tk.setup_Sat("TLE.txt")

elev = []
azim = []
pl  = []
distance = []
sat_long = []
sat_lat  = []
sat_altitude = []

simulation_time_period =(60*1)*60 #24Hours -> enough for multiple full rotation
simulation_time_step = 1 #second
current_time_UTC, year, month, day, hour, minutes, seconds = tk.get_time(0)
transmission_time = [current_time_UTC]
lambda_ =0.1#packets/per unit time
#s = 5 # 5 minute interval
i = 0
while (transmission_time[i] - (current_time_UTC)).total_seconds() <  (simulation_time_period*60):
    i = i +1
    u = random.uniform(0,1)
    time_diff = -1* (1/lambda_) * log(1-u)
    transmission_time.append(transmission_time[i-1] + timedelta(minutes=time_diff))


time_string = []

for i in range(1,len(transmission_time)):
    tx_time_formatted = tk.format_timeadvance(transmission_time[i])
    tx_time_string = [str(element) for element in tx_time_formatted]
    joined_string = ",".join(tx_time_string)
    time_string.append(joined_string)
    tk.update_sat_SSP_predetermined_time(Sat_Group[0],tx_time_formatted)
    ele, azi = AD.look_up_angles(GS_Group[0].location[0], Sat_Group[0].ssp_lon, GS_Group[0].location[1],
                                 Sat_Group[0].ssp_lat, Sat_Group[0].altitude + C.EARTH_RADIUS_KM)
    ele = math.degrees(ele) * -1
    azim.append(math.degrees(azi))
    elev.append(ele)
    sat_long.append(Sat_Group[0].ssp_lon)
    sat_lat.append(Sat_Group[0].ssp_lat)
    dd = mu.distance_to_sat(GS_Group[0].location[0], Sat_Group[0].ssp_lon, GS_Group[0].location[1],
                            Sat_Group[0].ssp_lat, Sat_Group[0].altitude)
    distance.append(dd)
    spreading,abs = Channel.path_loss(Sat_Group[0].altitude, [217.5], 10, 0.25, GS_Group[0].location[2]) #15 degree inclination an
    lambda_fc = 3e8 / 217.5e9
    spreading_loss = 20*math.log10((4*math.pi)/lambda_fc*(dd*10**3))
    pl.append(spreading_loss+abs[0])
    sat_altitude.append(Sat_Group[0].altitude)


results = {'Time of Transmission': time_string,
           'Satellite SSP Longitude': sat_long,
           'Satellite SSP Latitude': sat_lat,
           'Path Loss dB': pl,
           'Azimuth Degree ': azim,
           'Elevation Degree': elev,
           'Distance to Satellite' : distance,
           'Sat Altitude':sat_altitude
        }


df = pd.DataFrame(results, columns= ['Time of Transmission', 'Satellite SSP Longitude','Satellite SSP Latitude','Path Loss dB', 'Azimuth Degree', 'Elevation Degree', 'Distance to Satellite','Sat Altitude'])
df.to_csv ('results/Michele_'+'simulation_Elevation15deg.csv',index=False, header=True)



#
#
#
# simulation_time_period =(60*24)*60 #24Hours -> enough for multiple full rotation
# simulation_time_step = 1 #second
# current_time_UTC, year, month, day, hour, minutes, seconds = tk.get_time(0)
# transmission_time = [current_time_UTC]
# lambda_ =0.1#packets/per unit time
# #s = 5 # 5 minute interval
# i = 0
# while (transmission_time[i] - (current_time_UTC)).total_seconds() <  (simulation_time_period*60):
#     i = i +1
#     u = random.uniform(0,1)
#     time_diff = -1* (1/lambda_) * log(1-u)
#     transmission_time.append(transmission_time[i-1] + timedelta(minutes=time_diff))
#
# print("length",len(transmission_time))
#
# recieved_power = []
# sat_longitude  = []
# sat_latitide   = []
#
# for i in range(1, len(transmission_time)):
#     src_gs = GS_Group[0]
#     print("Itr num : " , i , "out of: " , len(transmission_time))
#     tx_time_formatted = tk.format_timeadvance(transmission_time[i])
#     for current_sat in sat_Group:
#         tk.update_sat_SSP_predetermined_time(current_sat,tx_time_formatted)
#
#     ele_ul, azi_ul = src_gs.elevation_and_azimuth()
#     ele_dl, azi_dl = target_gs.elevation_and_azimuth()
#
#     thz_spreading,thz_UL_abs = channel.path_loss(src_gs.UL_sat.altitude,[217.5],ele_ul,0.25, src_gs.location[2])
#     thz_spreading,thz_DL_abs = channel.path_loss(target_gs.UL_sat.altitude,[295],ele_dl,0.25, target_gs.location[2])
#
#     thz_power, thz_data_rate_ul = link_budget.link_budget(0.2, src_gs._min_distance_to_sat*10**3, 17e9,69.29+45.2, 217.5e9, thz_UL_abs[0], thz_spreading[0], "8PSK" )
#     thz_power, thz_data_rate_dl = link_budget.link_budget(0.2, target_gs._min_distance_to_sat*10**3, 40e9,71.94+47.85, 295e9, thz_DL_abs[0], thz_spreading[0], "QPSK" )
#
#
#     ka_power, ka_data_rate_ul   = link_budget.link_budget(5, src_gs._min_distance_to_sat*10**3, 2.5e9, 51.71+27.63, 28.75e9, 0, thz_spreading[0], "256QAM" )
#     ka_power, ka_data_rate_dl   = link_budget.link_budget(5, target_gs._min_distance_to_sat*10**3,5e9, 54.58+30.5, 40e9, 0, thz_spreading[0], "256QAM" )
#
#
#     data_rates_UL_THz.append(thz_data_rate_ul)
#     data_rates_UL_ku.append(ka_data_rate_ul)
#
#     data_rates_DL_THz.append(thz_data_rate_dl)
#     data_rates_DL_ku.append(ka_data_rate_dl)
#
#     # latency.append(  ((total_distance_for_route*1000) / (C.SPEED_OF_LIGHT)) *1000   )
#
# while len(data_rates_UL_THz) != len(cross_link_Thz):
#     data_rates_UL_THz.append(0)
#     data_rates_DL_THz.append(0)
#     data_rates_UL_ku.append(0)
#     data_rates_DL_ku.append(0)
#     latency.append(0)
#     cross_link_thz_avg.append(0)
#     cross_link_ka_avg.append(0)
# results = {'THZ-UL': data_rates_UL_THz,
#            'THZ-DL': data_rates_DL_THz,
#            'THZ-CL': cross_link_Thz,
#            'KA_UL': data_rates_UL_ku,
#            'KA_DL': data_rates_DL_ku,
#            'KA_CL': cross_link_KA,
#            'lat' : latency,
#         }
#
#
# df = pd.DataFrame(results, columns= ['THZ-UL', 'THZ-DL','THZ-CL','KA_UL', 'KA_DL', 'KA_CL', 'lat'])
# df.to_csv ('results/resssults_'+str(sim_index)+'simulation' + str(packet_drops)+ 'drops.csv',index=False, header=True)
