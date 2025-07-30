
import math
# from skyfield.api import Loader, EarthSatellite, load
# from skyfield.timelib import Time
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from simulator import toolkit as tk
from simulator import astrodynamics as ad

# def get_sat_const():
#     simulation_time_period =3*60 #2 minutes
#     simulation_time_step = 1 #second
#     current_time_UTC, year, month, day, hour, minutes, seconds = tk.get_time(0)
#     transmission_time = [current_time_UTC]
#     lambda_ =1.25#packets/per unit time
#     #s = 5 # 5 minute interval
#     i = 0
#
#     while (transmission_time[i] - (current_time_UTC)).total_seconds() <  (simulation_time_period*60):
#         i = i +1
#         u = random.uniform(0,1)
#         time_diff = -1* (1/lambda_) * math.log(1-u)
#         transmission_time.append(transmission_time[i-1] + timedelta(minutes=time_diff))
#
#
#     sat_Group = tk.custom_sat(3, 3600) #tk.setup_Sat("TLE.txt")
#     for current_sat in sat_Group:
#         tx_time_formatted = tk.format_timeadvance(transmission_time[random.randrange(len(transmission_time)-1)])
#         tk.update_sat_SSP_predetermined_time(current_sat,tx_time_formatted)
#
#     return sat_Group
'''
sat_Group =tk.custom_sat(3, 215)
sat_Group2 = tk.custom_sat2(3,15)
sat_Group = sat_Group + sat_Group2
distance_errors = []
for satsrc in sat_Group:
    for satrg in sat_Group:
        if satsrc == satrg:
            continue
        xyz1 = tk.get_geo_cords(satsrc, 0)
        xyz2 = tk.get_geo_cords(satrg,0)
        distance = math.sqrt((xyz1[0]-xyz2[0])**2 + (xyz1[1]-xyz2[1])**2 +(xyz1[2]-xyz2[2])**2 )
        # print(distance)
        if distance < 2:
            distance_errors.append(distance)

print(distance_errors)
print(len(distance_errors))
'''
print(ad.period_from_semi_major_axis((6370+600)*1e3)/3600)

# GS_Longitude = [-103.5101597,  88.7879 ]
# GS_Latitude = [43.6726477,  30.1534]
# GS_Altitude = [1.620, 5.011] #in km, from earth.google.com
# GS_Location_Name = ['Custer County, South Dakota ', 'California, Mount Whitney', 'Peru, La Rinconada', 'China, Tibet']
#
# number_of_GS_X = 1
# number_of_GS_Y = 1
# GS_Group = tk.setup_GS(GS_Longitude, GS_Latitude, GS_Altitude,GS_Location_Name, number_of_GS_X, number_of_GS_Y)
# sat_Group = tk.custom_sat(3,155)
# sat_Group2 = tk.custom_sat2(3,20)
# sat_Group = sat_Group + sat_Group2
# long = []
# lat = []
# i = 0
# print(len(sat_Group))
#
# # for sat_src in sat_Group:
# #     if sat_src.argperige + random.uniform(0,0.4):
# #         sat_src.argperige = sat_src.argperige + random.uniform(0,0.4)
# #     else:
# #         sat_src.argperige = 4 + random.uniform(0,0.4)
# #     i=i+1
# i=1
# # while i < 50:
# for sat_src in sat_Group:
#     tk.update_sat_SSP(sat_src, 60*10)
#     long.append(sat_src.ssp_lon)
#     lat.append(sat_src.ssp_lat)
#     i=i+1
#
#
# map_api.satellite_coverage_Map(long, lat, 'teracoverage/newsatconst.png')
#
#
# simulation_time_period =3*60 #2 minutes
# simulation_time_step = 1 #second
# current_time_UTC, year, month, day, hour, minutes, seconds = tk.get_time(0)
# transmission_time = [current_time_UTC]
# lambda_ =1.25#packets/per unit time
# #s = 5 # 5 minute interval
# i = 0
#
# while (transmission_time[i] - (current_time_UTC)).total_seconds() <  (simulation_time_period*60):
#     i = i +1
#     u = random.uniform(0,1)
#     time_diff = -1* (1/lambda_) * math.log(1-u)
#     transmission_time.append(transmission_time[i-1] + timedelta(minutes=time_diff))
#
# for i in range(1, len(transmission_time)):
#     print("Itr num : " , i , "out of: " , len(transmission_time))
#     tx_time_formatted = tk.format_timeadvance(transmission_time[i])
#     for current_sat in sat_Group:
#         tk.update_sat_SSP_predetermined_time(current_sat,tx_time_formatted)
#         for gs_up in GS_Group:
#             gs_up.check_coverage(current_sat)
#             gs_up.update_ULSAT()
#
#
# #
#
#     src_gs = GS_Group[0]
#     target_gs = GS_Group[1]
#     route = tk.short_interlink_route(src_gs.UL_sat, target_gs.UL_sat, src_gs, target_gs, sat_Group)
#     # route = tk.best_link(src_gs.UL_sat,500, src_gs, sat_Group, target_gs)
#     route.print_route()
#     # dis, hops, route2,route_for_plotting = tk.best_link(src_gs.UL_sat,60, src_gs, sat_Group, target_gs)
#     map_api.plot_route(route, src_gs, target_gs, sat_Group,'new_route_link.png')
#     # map_api.plot_route(route2, src_gs, target_gs, sat_Group,'new_route2_link.png')
#     break
#






#     for gs_up in GS_Group:
#         gs_up.check_coverage(current_sat)
#         gs_up.update_ULSAT()
# src_gs = GS_Group[0]
# target_gs = GS_Group[3]
# route = tk.short_interlink_route(src_gs.UL_sat, target_gs.UL_sat, src_gs, target_gs, sat_Group)
# # dis, hops, route2,route_for_plotting = tk.best_link(src_gs.UL_sat,60, src_gs, sat_Group, target_gs)
# map_api.plot_route(route, src_gs, target_gs, sat_Group,'new_route_link.png')
# map_api.plot_route(route2, src_gs, target_gs, sat_Group,'new_route2_link.png')
# xyz = []
# for sat in sat_Group:
#     xyz.append(tk.get_geo_cords(sat,0))
# ultimate_lowest =  999999999999999999999999
# ultimate_longest = -1
# i = 0
# j= 0
# for sat_src in sat_Group:
#     rep = 0
#     xyz1 = xyz[i]
#     shortest_distance  = 999999999999999999999999
#     longest = -1
#     j=0
#     for sat_target in sat_Group:
#         xyz2 = xyz[j]
#         if sat_src == sat_target:
#             rep = rep + 1
#             j = j+1
#             continue
#         if rep > 1:
#             print("Error rep")
#             exit()
#         distance = math.sqrt((xyz1[0]-xyz2[0])**2 + (xyz1[1]-xyz2[1])**2 +(xyz1[2]-xyz2[2])**2 )
#         if distance < shortest_distance:
#             shortest_distance = distance
#         if distance < ultimate_lowest:
#             ultimate_lowest = distance
#         j=j+1
#     i=i+1
#     print("\n\nUltimare Distance Final: " , shortest_distance)


# long = []
# lat = []
# i = 0
# # while i < 50:
# for sat_src in sat_Group:
#     tk.update_sat_SSP(sat_src, 0)
#     long.append(sat_src.ssp_lon)
#     lat.append(sat_src.ssp_lat)
#     # i=i+1
#
# map_api.satellite_coverage_Map(long, lat, 'teracoverage/EDIT2obritalplanes600sat.png')

# print(sat_src.num_of_rev)
# exit()
# while i<(10000*60*60):
#     i = i + 60
#     for jk in range(909999):
#         jk = jk - 1
#         jk = jk + 2
#     tk.update_sat_SSP(sat_src, i)
#     tk.update_sat_SSP(sat_trg, 0)
#     print("Altitude of Sat1 : " , sat_src.altitude)
#     print("Altitude of Sat2 : " , sat_trg.altitude)
#
#     long1.append(sat_src.ssp_lon)
#     lat1.append(sat_src.ssp_lat)
#     long2.append(sat_trg.ssp_lon)
#     lat2.append(sat_trg.ssp_lat)
#     xyz1 = tk.get_geo_cords(sat_src, 0)
#     xyz2 = tk.get_geo_cords(sat_trg,0)
#     distance = math.sqrt((xyz1[0]-xyz2[0])**2 + (xyz1[1]-xyz2[1])**2 +(xyz1[2]-xyz2[2])**2 )
#     xyz1_true = tk.get_correct_GEO_cords(sat_trg, 0)
#
#     # print("Correct Geo: " ,xyz1_true)
#     print("Diff Error GEO: ", [abs(xyz2[0] -xyz1_true[0]) , abs(xyz2[1] -xyz1_true[1]), abs(xyz2[2] -xyz1_true[2])])
#     # print("correct GEO:" , xyz1_true)
#     # print("Distance: " , distance)
#     # print("-----------\n")
#     # print(sat_src.ssp_lon)
#     # print(sat_trg.ssp_lat)
#     # print(sat_trg.ssp_lon)
#     # print(sat_trg.ssp_lat)
#     print("-----\n")
# map_api.sat_path_image(long1, lat1, "Ali-link.png")
# map_api.sat_path_image(long2, lat2, "Ali-link2.png")

    # xyz1 = tk.get_geo_cords(sat_src, 0)
    # # xyz2 = tk.get_correct_GEO_cords(sat_src,0)
    # print("My XYZ: " , xyz1)
    # print("Skyfield : ", xyz2)
    # print("------------------------------------------\n")
#
# xyz1 = sat_src.xyz_r
# xyz2 = sat_trg.xyz_r
# distance = math.sqrt((xyz1[0]-xyz2[0])**2 + (xyz1[1]-xyz2[1])**2 +(xyz1[2]-xyz2[2])**2 )
#
#
# sat_src = sat_Group[100]
# sat_trg = sat_Group[250]
#
#
# simulation_time_period =100*60 #2 minutes
# simulation_time_step = 1 #second
# current_time_UTC, year, month, day, hour, minutes, seconds = tk.get_time(0)
# transmission_time = [current_time_UTC]
# lambda_ =0.0005 #packets/per unit time
# #s = 5 # 5 minute interval
# i = 0
#
# while (transmission_time[i] - (current_time_UTC)).total_seconds() <  (simulation_time_period*60):
#     i = i +1
#     u = random.uniform(0,1)
#     time_diff = -1* (1/lambda_) * math.log(1-u)
#     transmission_time.append(transmission_time[i-1] + timedelta(minutes=time_diff))
# sat_Group = tk.custom_sat(2, 600)
# for sat in sat_Group:
#     print(sat.altitude)
# ultimate_lowest = 999999999999999999999999
# for sat_src in sat_Group:
#     shortest_distance = 999999999999999999999999
#     num_repeat = 0
#     for sat_trg in sat_Group:
#         if sat_src == sat_trg:
#             num_repeat = num_repeat +1
#             print("Continue , " ,num_repeat)
#             continue
#         # tx_time_formatted = tk.format_timeadvance(transmission_time[1])
#         # tk.update_sat_SSP_predetermined_time(sat_src,tx_time_formatted)
#         # tk.update_sat_SSP_predetermined_time(sat_trg,tx_time_formatted)
#         xyz1 = tk.get_geo_cords(sat_src, 0)
#         xyz2 = tk.get_geo_cords(sat_trg, 0)
#         distance = math.sqrt((xyz1[0]-xyz2[0])**2 + (xyz1[1]-xyz2[1])**2 +(xyz1[2]-xyz2[2])**2 )
#         if distance < shortest_distance:
#             shortest_distance = distance
#         if distance < ultimate_lowest:
#             ultimate_lowest = distance
#     print("Distance is: ", shortest_distance)
#     print("ultimate_lowest Distance is: ", ultimate_lowest)
#
# print("\n\nultimate_lowest Distance Final is: ", ultimate_lowest)

# ##PaPer calc for x,y,z:::
# eccentric_anomly1, mean_anomly1, time_diff1 = tk.get_eccentric_mean_anamoly(sat_src, 0)
# eccentric_anomly2, mean_anomly2, time_diff2 = tk.get_eccentric_mean_anamoly(sat_trg, 0)
#
# v1 = 2 * math.atan( math.sqrt((1+sat_src.eccentricity)/(1-sat_src.eccentricity)) * math.tan(eccentric_anomly1/2))
# v2 = 2 * math.atan( math.sqrt((1+sat_trg.eccentricity)/(1-sat_trg.eccentricity)) * math.tan(eccentric_anomly2/2))
#
# r1 = (sat_src.semi_major_axis * (1-sat_src.eccentricity**2)) / (1+ (sat_src.eccentricity*math.cos(v1)))
# r2 = (sat_trg.semi_major_axis * (1-sat_trg.eccentricity**2)) / (1+ (sat_trg.eccentricity*math.cos(v2)))
#
# print("R1 = " , r1)
# geo_cords = tk.get_geo_cords(sat_src, 0)
# print("Semi-Major-Axis a = " , sat_src.semi_major_axis)
#
# omega = math.radians(sat_src.argperige)
# sigma = math.radians( sat_src.right_asc)
# i     = math.radians(sat_src.inc)
# numer = math.sqrt(398600) * 0.00108263 * 6378**2
# denom = (1-sat_src.eccentricity**2)**2 * sat_src.semi_major_axis**(7/2)
# sigma_new = sigma + ((-3/2) * (numer/denom) * math.cos(i))*time_diff1
# inter_omega = (-1)*(3/2)*((math.sqrt(398600)*0.00108263*6378**2)/((1-sat_src.eccentricity**2)**2 * sat_src.semi_major_axis**(7/2))) *(5/2 * math.sin(i)**2 -2)
# omega_new   = omega + inter_omega*time_diff1
# sigma = sigma_new
# omega = omega_new
# x = r1 * ( (math.cos(sigma)*math.cos(omega+v1)) - (math.sin(sigma)*math.sin(omega+v1)*math.cos(i))   )
# y = r1 * ( (math.sin(sigma)*math.cos(omega+v1)) + (math.cos(sigma)*math.sin(omega+v1)*math.cos(i))   )
# z = r1 * (  math.sin(omega+v1)*math.sin(i)   )
# print("Calculated xyz from paper = " , [x,y,z])
# print("correct geo = " , geo_cords)
# print("Skyfield geo cords = " , tk.get_correct_GEO_cords(sat_src,0) )
#
#





























"""
Todays Work:
1) Clean up yesterdays code
2) Write map utility functions to plot locations of all ground-stations, and satellite FootPrint
3) Write function to plot best route (or best 10 routes to check if algorithm actually works. )
4) Write up the final simulation project in a file called simulation_1
    4-A) Put each type of simulation in its own file.

"""

# x, y = lb.link_budget(2, 500e3, 100e9, 80, 100e9, 6, 150, "16PSK" )
# print([x,y])



# sat = tk.setup_Sat()


# sat1 = sat[0]
# sat2 = sat[1]
# xyz1, xy1 = tk.update_sat_SSP(sat1, 0)
# xyz2, xy2 = tk.update_sat_SSP(sat2,0)
# coor1 = tk.get_coorect_SSP_cords(sat1,0)
# coor2 = tk.get_coorect_SSP_cords(sat2,0)
# print("Coorssp1: " , coor1)
# print("Coorssp2: " , coor2)
# print("Sat-1 SSP: " , [sat1.ssp_lon, sat1.ssp_lat])
# print("Sat-2 SSP: " , [sat2.ssp_lon, sat2.ssp_lat])
# print("XYZ-1: " ,xyz1)
# print("XYZ-2: " ,xyz2)
# print("XY-1: " ,xy1)
# print("XY-2: " ,xy2)
# d1 = math.sqrt((xyz1[0]-xyz2[0])**2 + (xyz1[1]-xyz2[1])**2 + (xyz1[2]-xyz2[2])**2)
# d2 = math.sqrt((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2 )
# print(d1)
# print(d2)
# print(tk.sat_to_sat_visibility(xyz1, xyz2))



# x1 = 3500
# y1 = 3300
# z1 = 3200
# x2 = 4502
# y2 = 3000
# z2 = 3000
# r1 = [x1 ,y1, z1];
# r2 = [x2 ,y2 ,z2];
# norm_r1 = math.sqrt(x1**2 + y1**2 + z1**2)
# norm_r2 = math.sqrt(x2**2 + y2**2 + z2**2)
# dot_prod = x1*x2 + y1*y2 + z1*z2
# vis_test = math.acos(6371 / norm_r1) + math.acos(6371/ norm_r2) - math.acos( dot_prod / (norm_r1*norm_r2) )
# lat,long,bs = astrod.SSP_cord(x1, y1, z1)
# print(vis_test)
# print(lat,long)





#[-103.5101597, -87.03400638561939, 43.6726477, 31.04565228669573, 356.0798432582151]
# Le = 43.6726477
# Ls = (31.04565228669573)
# le = -103.5101597
# ls = (-87.034006385619391)
# h_diff  = 356
# rs = 6730 + h_diff
# print("distance",math_util.long_lat_distance(Le, Ls, le, ls))
# x,y = astrod.look_up_angles(le, ls, Le, Ls, rs)
# Le = math.radians(Le)
# Ls = math.radians(Ls)
# le = math.radians(le)
# ls = math.radians(ls)
#
# print("bound: ", math.acos(6370/rs))
#
#
#
# print("elevation", math.degrees(x))
# print(math.degrees(y))
#
# angle = math.acos(math.cos(Le)*math.cos(Ls)*math.cos(ls-le)+math.sin(Ls)*math.sin(Le))
# re = 6370
# rs = re + h_diff
# d = math.sqrt(re**2 + rs**2 - 2*re*rs * math.cos(angle))
# print("distance_etod: ", d)
# print("ce" , math_util.distance_to_sat(le, ls, Le, Ls, h_diff))
#p = math.asin(re/rs)

# nadir = math.atan((math.sin(p)*math.sin(angle)) / (1-math.sin(p)*math.cos(angle)) )
# D = re * (math.sin(angle)/math.sin(nadir))
# print(D)
#
# print("dcheck: ",math_util.distance_to_sat(le, ls, Le, Ls, h_diff))

# print(datetime.utcnow())

# str = datetime.utcnow()
# print(str)
#
# current_time_UTC = datetime.utcnow()
#
# year    = current_time_UTC.year
# month   = current_time_UTC.month
# day     = current_time_UTC.day
# hour    = current_time_UTC.hour
# minutes = current_time_UTC.minute
# seconds = current_time_UTC.second
#
# min_after_midnight = hour*60 + minutes + seconds/60
#
# # calculate Julian Day
# y = year
# m = month
# d = day
#
# if month == 1 or month ==2:
#     y = y-1
#     m = m + 12
#
# A  = y/100
# B  = A/4
# C  = 2-A+B
# E  = 365.25 * (y+4716)
# F  = 30.6001 * (m+1)
# JD = C + d + E + F -1524.5
#
# print(JD)

##Example 1:
# E = astrod.eccentric_anomly(1.2037,0.0002282)
# x_0 , y_0 , z_0 = astrod.polor_cord_orbital_system(6783,0.0002282,E)
# M  = astrod.orbital_to_GEC_transformation_matrix(77.2230, 184.0276, 51.6491)
# angle = astrod.angle_Between_GEC_Rot_System(2456591,324.98)
# M2 = astrod.GEC_to_Rotating_System_Transformation_Matrix(angle)
# y = astrod.sat_cord_in_rot_system(M2, M, x_0, y_0, z_0)
