import simulator.satellite as sat
import numpy as np
import random
import simulator.ground_station as GS
import simulator.astrodynamics as astrod
from datetime import datetime, timedelta
import math
# from skyfield.api import load,EarthSatellite,wgs84
# from art import *
import simulator.plot_utils
# from PIL import Image
import simulator.weather
import simulator.route
import copy
import simulator.constants as C
from simulator.simulation import data_path
import os


from math import sin, cos, acos, sqrt, floor, atan2, radians

'''
Orbital Coordinates systems

XYZ_r--> Initial coordinates system. Earth centered. Z is the Earth's rotation axis. X and Y arbitrary, perpendicular with
Z and in the equatorial plane. Rotates with the Earth. 

X0Y0Z0 --> Orbital plane coordinate system. Z0 will always be 0 for a satellite rotating in the orbital plane used as 
reference. Also called perifocal coordinate system

R0Phi0 --> Polar coordinates in the orbital plane. Two dimensional polar coordinates assumed to be used in the orbital
plane of reference. Phi0 is called True Anomaly.

GEC (XiYiZi) --> Geocentric Equatorial Coordinate system. Does not rotate with Earth. Zi is Earth rotation axis. Xi  from the 
center of the Earth to the first point of Aries (center of the sun at vernal equinox)
    RA --> right ascension. Angle measured from Xi.
    w --> Argument of perigee west. Angle between the RA of the ascending node and the perigee.
    i --> Inclination. Angle between orbital plane and equatorial plane.


Some names:
Perigee/periapsis --> closest point of the orbit to the Earth.
Apogee/apoapsis --> furthest point of the orbit to the Earth.
E --> Eccentric Anomaly. Angle from X0 axis to the line joining C (center of orbit) to A (vertical projection of
    satellite position on the circumscribed circle).
M --> Mean anomaly. Arc length (radians) that satellite would have traveled since the perigee if it were moving on the 
    circumscribed circle at mean angular velocity.
SSP --> Sub Satellite Point
UTC --> Universal Time Coordinates or zulu time.
Julian dates/days --> December 31st 1899 --> 241 5020 juian day.
Osculating orbit --> Keplerian orbit the spacecraft would follow if all perturbing forces were removed at that time, 
    with orbital elements (a, e, tp, omega, i w)
Anomalistic period --> elapsed time between successive perigee passages.

Quantities required to fully describe satellite position at time t:
    Eccentricity e
    semimajor axis a
    time of perige tp (or mean anomaly M at a given time)
    RA of ascending node Omega
    Inclinatioin i
    Argument of perige w
    
Look Angles
    Azimuth --> angle from geographic nord to the projection of the satellite.
    Elevation --> Angle from local horizontal plane to the satellite path.
'''


def welcome_message():
    print("------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------")
    print("                                                  UN-LAB                       ")
    print("Tera-Hertz Satellite Communication")
    print("------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------")
    print("By: ")
    print("                Dr. Jornet")
    print("                Ali Al Qaraghuli")
    print("                Hussam Abdellatif")
    print("------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------")
    print("Setting Up simulator")


def main_user_controller(sat_Group, GS_Group):
    while True:
        ret = uio.main_console()
        if ret == '1':
            sat_IO = uio.command_1(sat_Group)
            ##Call function to calculate SSP of selected satellite
            plot_utils.sat_path_image(Sat_Foot_Prints_Sat_Long, Sat_Foot_Prints_Sat_Lat, "Whatever-IDC.png")
            im = Image.open(r"Whatever-IDC.png")
            im.show()
        elif ret == '2':
            uio.command_2(GS_Group)


## Setup the Ground Stations as an object
def setup_GS(GS_Longitude, GS_Latitude, GS_Altitude, GS_Location_Name, number_of_GS_X, number_of_GS_Y):
    total_GS = number_of_GS_X
    GS_Group = np.empty(total_GS, dtype=GS.GS)
    ### Intialize ALL GS_Radios starting from GS_0 -> GS_<total_GS-1> , first <number_of_GS_X> are GS_X
    for i in range(total_GS):
        gs_name = "GS_" + str(i)
        location_temp = [GS_Longitude[i], GS_Latitude[i], GS_Altitude[i]]
        temp, humidity, pressure = weather.get_weather_stats(GS_Latitude[i], GS_Longitude[i])
        gs_temp = GS.GS(gs_name, GS_Location_Name[i], location_temp, 2, 100, humidity, pressure, temp)
        if i < number_of_GS_X:
            gs_temp.internet_access = False
        GS_Group[i] = gs_temp
    return GS_Group


def custom_sat2(orbital_planes, num_sat):
    # This function will take a templte tle and create a new constellation from that templete.
    templete_sat = setup_Sat('TLE_templete.txt')[0]
    inc_list = [53, 53.001, 53.002, 55.4, 55.5, 55.6, 55.7, 55.8]  # custom inclination angles - tweak as needed
    omega_list = []  # a change by 0.5 will lead to around 60km seperation between satellites in same orbit considering other parms kept the same
    right_asc = np.arange(0, 320, 10).tolist()
    sat_constellation = []
    templete_date = templete_sat.tle_date
    templete_eccentricity = templete_sat.eccentricity
    templete_mean_anonly = templete_sat.mean_anomaly  # random.uniform(20, 300) #templete_sat.mean_anomaly
    temple_num_of_rev = 15.93596695
    templete_altitude = astrod.sat_height(float(temple_num_of_rev))
    templete_period = ((86400) / float(temple_num_of_rev)) ** 2
    templte_semi_major_axis = astrod.apogee_new(templete_period) / (10 ** 3)
    omega_start = 30
    omega_stepsize = 0.2
    sat_num = 1
    # print(templete_date)
    # time_sat_TLE_capture = templete_date
    # year_ = 2000 + int(time_sat_TLE_capture[0:2])
    # decimal_days1 = float(time_sat_TLE_capture[2:])
    # decimal_days2 = float(time_sat_TLE_capture[2:]) + 0.0002
    # tempdatetime = datetime(year_, 1, 1) + timedelta(decimal_days1 - 1)
    # tempdatetime2 = datetime(year_, 1, 1) + timedelta(decimal_days2 - 1)
    # time_diff = (tempdatetime2) - tempdatetime
    # print(time_diff.total_seconds())
    for i in range(orbital_planes):
        incline_angle = inc_list[i]
        for j in range(len(right_asc)):
            raan = right_asc[j]
            for k in range(num_sat):
                sat_constellation.append(sat.satellite('TeraSatbb-' + str(sat_num),
                                                       str(float(templete_date) + random.uniform(0, 2)),
                                                       incline_angle,
                                                       raan,
                                                       templete_eccentricity,
                                                       random.uniform(20, 320),
                                                       random.uniform(20, 320),
                                                       temple_num_of_rev,
                                                       templete_altitude,
                                                       templete_period,
                                                       templte_semi_major_axis,
                                                       'DontCare',
                                                       'DontCare2'))
                sat_num = sat_num + 1

    # for i in range(orbital_planes):
    #     incline_angle = inc_list[i]
    #     omega_start = 10
    #     for j in range(math.floor(num_sat/len(right_asc))):
    #         omega_start = omega_start + random.uniform(0.1, 0.2)
    #         if omega_start >= 355:
    #             omega_start = 10 + random.uniform(0.1, 0.2)
    #         for k in range(len(right_asc)):
    #             sat_constellation.append(sat.satellite('TeraSat-' + str(sat_num) ,
    #                                                     templete_date,
    #                                                     incline_angle,
    #                                                     random.uniform(20,330),
    #                                                     templete_eccentricity,
    #                                                     random.uniform(20,330),
    #                                                     random.uniform(20,330),
    #                                                     temple_num_of_rev,
    #                                                     templete_altitude,
    #                                                     templete_period,
    #                                                     templte_semi_major_axis,
    #                                                     'DontCare',
    #                                                     'DontCare2'))
    #             sat_num = sat_num +1

    return sat_constellation


def custom_sat(orbital_planes, num_sat):
    # This function will take a templte tle and create a new constellation from that templete.
    templete_sat = setup_sats_from_tle('../data/TLE_templete.txt')[0]
    inc_list = [50, 60, 70, 80, 55.5, 55.6, 55.7, 55.8]  # custom inclination angles - tweak as needed
    omega_list = []  # a change by 0.5 will lead to around 60km seperation between satellites in same orbit considering other parms kept the same
    right_asc = np.arange(0, 320, 20).tolist()
    sat_constellation = []
    templete_date = templete_sat.tle_date
    templete_eccentricity = templete_sat.eccentricity
    templete_mean_anonly = templete_sat.mean_anomaly  # random.uniform(20, 300) #templete_sat.mean_anomaly
    temple_num_of_rev = 15.93596695
    templete_altitude = astrod.sat_height(float(temple_num_of_rev))
    templete_period = ((86400) / float(temple_num_of_rev)) ** 2
    templte_semi_major_axis = astrod.apogee_new(templete_period) / (10 ** 3)
    omega_start = 30
    omega_stepsize = 0.2
    sat_num = 1
    # print(templete_date)
    # time_sat_TLE_capture = templete_date
    # year_ = 2000 + int(time_sat_TLE_capture[0:2])
    # decimal_days1 = float(time_sat_TLE_capture[2:])
    # decimal_days2 = float(time_sat_TLE_capture[2:]) + 0.0002
    # tempdatetime = datetime(year_, 1, 1) + timedelta(decimal_days1 - 1)
    # tempdatetime2 = datetime(year_, 1, 1) + timedelta(decimal_days2 - 1)
    # time_diff = (tempdatetime2) - tempdatetime
    # print(time_diff.total_seconds())
    for i in range(orbital_planes):
        incline_angle = inc_list[i]
        for j in range(len(right_asc)):
            raan = right_asc[j]
            for k in range(num_sat):
                sat_constellation.append(sat.satellite('TeraSat-' + str(sat_num),
                                                       str(float(templete_date) + random.uniform(0, 0.1)),
                                                       incline_angle,
                                                       raan,
                                                       templete_eccentricity,
                                                       random.uniform(20, 320),
                                                       templete_mean_anonly,
                                                       temple_num_of_rev,
                                                       templete_altitude,
                                                       templete_period,
                                                       templte_semi_major_axis,
                                                       'DontCare',
                                                       'DontCare2'))
                sat_num = sat_num + 1

    # for i in range(orbital_planes):
    #     incline_angle = inc_list[i]
    #     omega_start = 10
    #     for j in range(math.floor(num_sat/len(right_asc))):
    #         omega_start = omega_start + random.uniform(0.1, 0.2)
    #         if omega_start >= 355:
    #             omega_start = 10 + random.uniform(0.1, 0.2)
    #         for k in range(len(right_asc)):
    #             sat_constellation.append(sat.satellite('TeraSat-' + str(sat_num) ,
    #                                                     templete_date,
    #                                                     incline_angle,
    #                                                     random.uniform(20,330),
    #                                                     templete_eccentricity,
    #                                                     random.uniform(20,330),
    #                                                     random.uniform(20,330),
    #                                                     temple_num_of_rev,
    #                                                     templete_altitude,
    #                                                     templete_period,
    #                                                     templte_semi_major_axis,
    #                                                     'DontCare',
    #                                                     'DontCare2'))
    #             sat_num = sat_num +1

    return sat_constellation


def setup_sats_from_tle(file_name, sim=None):
    """file_name --> TLE file for the satellite/constellation as a .txt in /data"""
    sat_name = []
    sat_date = []
    sat_inc = []
    sat_right_ascension = []
    sat_eccentricity = []
    sat_argofperigee = []
    sat_mean_anamoly = []
    sat_num_rev_per_day = []
    sat_altitude = []
    sat_period = []
    sat_semi_major_axis = []
    TLE_line1 = []
    TLE_line2 = []
    with open(os.path.join(data_path, file_name), 'r') as reader:
        i = 0
        for line in reader:
            if i == 0:
                sat_name.append(line.rstrip())
            elif i == 1:
                TLE_line1.append(line)
                col_num = 1
                day = ''
                year = '20'
                for col in line:
                    if 19 <= col_num <= 20:
                        year = year + col
                    elif 21 <= col_num <= 32:
                        day = day + col
                    col_num = col_num + 1
                year = int(year)
                day = float(day)
                date_time = datetime(year, 1, 1) + timedelta(days=day)
                sat_date.append(date_time)
            else:
                TLE_line2.append(line)
                col_num = 1
                inclination = ""
                right_asc = ""
                ecent = "0."
                argperige = ""
                manamoly = ""
                num_of_rev = ""

                for col in line:
                    if 9 <= col_num <= 16:
                        inclination = inclination + col
                    elif 18 <= col_num <= 25:
                        right_asc = right_asc + col
                    elif 27 <= col_num <= 33:
                        ecent = ecent + col
                    elif 35 <= col_num <= 42:
                        argperige = argperige + col
                    elif 44 <= col_num <= 51:
                        manamoly = manamoly + col
                    elif 53 <= col_num <= 63:
                        num_of_rev = num_of_rev + col
                    else:
                        col_num = col_num  # do nothing
                    col_num = col_num + 1
                sat_inc.append(float(inclination))
                sat_right_ascension.append(float(right_asc))
                sat_eccentricity.append(float(ecent))
                sat_argofperigee.append(float(argperige))
                sat_mean_anamoly.append(float(manamoly))
                sat_num_rev_per_day.append(float(num_of_rev))
                sat_altitude.append(astrod.sat_height(float(num_of_rev)))
                # period_temp = ((86400) / float(num_of_rev))
                # real day: 23h 56min 4.09s
                period_temp = (23 * 3600 + 56 * 60 + 4.09) / float(num_of_rev)
                sat_period.append(period_temp)
                sat_semi_major_axis.append(astrod.semi_major_axis_from_period(period_temp))
            i = (i + 1) % 3
    # print(len(sat_name))
    # sat_Group = np.empty(len(sat_name), dtype=sat.Satellite)
    sat_group = []
    for i in range(0, len(sat_name)):
        satellite_temp = sat.Satellite(sat_name[i],
                                       sat_date[i],
                                       sat_inc[i],
                                       sat_right_ascension[i],
                                       sat_eccentricity[i],
                                       sat_argofperigee[i],
                                       sat_mean_anamoly[i],
                                       sat_num_rev_per_day[i],
                                       sat_altitude[i],
                                       sat_period[i],
                                       sat_semi_major_axis[i],
                                       TLE_line1[i],
                                       TLE_line2[i],
                                       sim)
        # sat_Group[i] = satellite_temp
        sat_group.append(satellite_temp)

    return sat_group


def get_time(seconds_in_future):
    time_advance = timedelta(seconds=seconds_in_future)
    current_time_UTC = datetime.utcnow() + time_advance
    ###Calculate Current Time
    year = current_time_UTC.year
    month = current_time_UTC.month
    day = current_time_UTC.day
    hour = current_time_UTC.hour
    minutes = current_time_UTC.minute
    seconds = current_time_UTC.second
    return current_time_UTC, year, month, day, hour, minutes, seconds


def format_timeadvance(current_time_UTC):
    year = current_time_UTC.year
    month = current_time_UTC.month
    day = current_time_UTC.day
    hour = current_time_UTC.hour
    minutes = current_time_UTC.minute
    seconds = current_time_UTC.second
    return [current_time_UTC, year, month, day, hour, minutes, seconds]


def time_in_seconds(hour, minutes, seconds):
    return hour * 3600 + minutes * 60 + seconds


def minutes_after_midnight(hour, minutes, seconds):
    return hour * 60 + minutes + seconds / 60


def julianDay(y, m, d):
    if m == 1 or m == 2:
        y = y - 1
        m = m + 12

    A = y / 100
    B = A / 4
    C = 2 - A + B
    E = 365.25 * (y + 4716)
    F = 30.6001 * (m + 1)
    return C + d + E + F - 1524.5


def update_sat_SSP(current_satellite, seconds_in_future):
    current_time_UTC, year, month, day, hour, minutes, seconds = get_time(seconds_in_future)
    JD = julianDay(year, month, day)
    min_after_midnight = minutes_after_midnight(hour, minutes, seconds)
    # Calculate avg_angular_velocity:
    avg_angular_velocity_calculated = astrod.avg_angular_velocity(current_satellite.semi_major_axis * (10 ** 3))
    # Calculate Current Mean Anamoly (based on current time - time data recorded)
    time_sat_TLE_capture = current_satellite.tle_date
    year_ = 2000 + int(time_sat_TLE_capture[0:2])
    decimal_days_ = float(time_sat_TLE_capture[2:])
    tempdatetime = datetime(year_, 1, 1) + timedelta(decimal_days_ - 1)
    # M2 = n * ((t_2-t_1) + M1/n) <-- t1 is datetime of TLE record, M1 is TLE Mean anamoly, t1 is current time
    time_diff = current_time_UTC - tempdatetime
    mean_anomly_calculated_rads = (time_diff.total_seconds() + (math.radians(
        current_satellite.mean_anomaly) / avg_angular_velocity_calculated)) * avg_angular_velocity_calculated
    mean_anomly_calculated_deg_checked = check_angle(math.degrees(mean_anomly_calculated_rads))

    # Solve for Eccentric anomaly
    E_solved = astrod.eccentric_anomly(math.radians(mean_anomly_calculated_deg_checked), current_satellite.eccentricity)
    E_solved_deg_checked = check_angle(math.degrees(E_solved))

    # Find Polar Coordinates
    x_0, y_0, z_0, garbage = astrod.polar_cord_orbital_system(current_satellite.semi_major_axis,
                                                              current_satellite.eccentricity,
                                                              math.radians(E_solved_deg_checked))
    # Solve for Transformation Matrices M1, M2
    M1 = astrod.orbital_to_GEC_transformation_matrix(current_satellite.argperige, current_satellite.right_asc,
                                                     current_satellite.inc)
    angle_gec = astrod.angle_Between_GEC_Rot_System(JD, min_after_midnight)
    M2 = astrod.GEC_to_Rotating_System_Transformation_Matrix(angle_gec)
    xyz_r = astrod.sat_cord_in_rot_system(M2, M1, x_0, y_0, z_0)
    SSP_Lat_Deg, SSP_Long_Deg, case = astrod.SSP_cord(xyz_r[0], xyz_r[1], xyz_r[2])
    # print("From tk: ", [SSP_Lat_Deg, SSP_Long_Deg])
    current_satellite.ssp_lat = SSP_Lat_Deg
    current_satellite.ssp_lon = SSP_Long_Deg
    return xyz_r, [x_0, y_0]


def get_eccentric_mean_anamoly(current_satellite, seconds_in_future):
    current_time_UTC, year, month, day, hour, minutes, seconds = get_time(seconds_in_future)
    ###Calculate avg_angular_velocity:
    avg_angular_velocity_calculated = astrod.avg_angular_velocity(None, current_satellite.semi_major_axis * (10 ** 3))
    ###Calculate Current Mean Anamoly (based on current time - time data recorded)
    time_sat_TLE_capture = current_satellite.tle_date
    year_ = 2000 + int(time_sat_TLE_capture[0:2])
    decimal_days_ = float(time_sat_TLE_capture[2:])
    tempdatetime = datetime(year_, 1, 1) + timedelta(decimal_days_ - 1)
    # M2 = n * ((t_2-t_1) + M1/n) <-- t1 is datetime of TLE record, M1 is TLE Mean anamoly, t1 is current time
    time_diff = current_time_UTC - tempdatetime
    mean_anomly_calculated_rads = (time_diff.total_seconds() + (math.radians(
        current_satellite.mean_anomaly) / avg_angular_velocity_calculated)) * avg_angular_velocity_calculated
    mean_anomly_calculated_deg_checked = check_angle(math.degrees(mean_anomly_calculated_rads))

    ######## Solve for Eccentric anomly
    E_solved = astrod.eccentric_anomly(math.radians(mean_anomly_calculated_deg_checked), current_satellite.eccentricity)
    E_solved_deg_checked = check_angle(math.degrees(E_solved))
    return E_solved, mean_anomly_calculated_rads, time_diff.total_seconds()


def get_geo_cords(current_satellite, seconds_in_future):
    current_time_UTC, year, month, day, hour, minutes, seconds = get_time(seconds_in_future)
    ###Calculate avg_angular_velocity:
    avg_angular_velocity_calculated = astrod.avg_angular_velocity(None, current_satellite.semi_major_axis * (10 ** 3))
    ###Calculate Current Mean Anamoly (based on current time - time data recorded)
    time_sat_TLE_capture = current_satellite.tle_date
    year_ = 2000 + int(time_sat_TLE_capture[0:2])
    decimal_days_ = float(time_sat_TLE_capture[2:])
    tempdatetime = datetime(year_, 1, 1) + timedelta(decimal_days_ - 1)
    # M2 = n * ((t_2-t_1) + M1/n) <-- t1 is datetime of TLE record, M1 is TLE Mean anamoly, t1 is current time
    time_diff = current_time_UTC - tempdatetime
    mean_anomly_calculated_rads = (time_diff.total_seconds() + (math.radians(
        current_satellite.mean_anomaly) / avg_angular_velocity_calculated)) * avg_angular_velocity_calculated
    mean_anomly_calculated_deg_checked = check_angle(math.degrees(mean_anomly_calculated_rads))

    ######## Solve for Eccentric anomly
    E_solved = astrod.eccentric_anomly(math.radians(mean_anomly_calculated_deg_checked), current_satellite.eccentricity)
    E_solved_deg_checked = check_angle(math.degrees(E_solved))

    ######## Find Polar Cordinates
    x_0, y_0, z_0, r_0 = astrod.polar_cord_orbital_system(current_satellite.semi_major_axis,
                                                          current_satellite.eccentricity,
                                                          math.radians(E_solved_deg_checked))
    ##### Solve for Transformation Matrices M1, M2
    M1 = astrod.orbital_to_GEC_transformation_matrix(current_satellite.argperige, current_satellite.right_asc,
                                                     current_satellite.inc, time_diff.total_seconds(),
                                                     current_satellite.eccentricity, current_satellite.semi_major_axis)
    return np.matmul(M1, np.transpose(np.array([x_0, y_0, z_0])))


def get_correct_GEO_cords(current_satellite, seconds_in_future):
    ts = load.timescale()

    current_time_UTC, year, month, day, hour, minutes, seconds = get_time(seconds_in_future)
    satellite_skyfield = EarthSatellite(current_satellite.TLE_Line1, current_satellite.TLE_Line2, 'DONTCARE', ts)
    t0_skyfield = ts.utc(year, month, day, hour, minutes, seconds)  # ts.utc(ts.now() + timedelta(minutes=x))
    time_skyfield = ts.now()
    geocentric_skyfield = satellite_skyfield.at(time_skyfield)
    return geocentric_skyfield.position.km


def get_coorect_SSP_cords(current_satellite, seconds_in_future):
    ts = load.timescale()
    current_time_UTC, year, month, day, hour, minutes, seconds = get_time(seconds_in_future)
    satellite_skyfield = EarthSatellite(current_satellite.TLE_Line1, current_satellite.TLE_Line2, 'DONTCARE', ts)
    t0_skyfield = ts.utc(year, month, day, hour, minutes, seconds)  # ts.utc(ts.now() + timedelta(minutes=x))
    time_skyfield = ts.now()
    geocentric_skyfield = satellite_skyfield.at(time_skyfield)
    subpointskyfield = wgs84.subpoint(geocentric_skyfield)
    return [subpointskyfield.longitude._degrees, subpointskyfield.latitude._degrees]


def sat_to_sat_visibility(sat_src, sat_trg):
    xyz1 = sat_src.xyz_r
    xyz2 = sat_trg.xyz_r
    r1 = sat_src.r_0
    r2 = sat_trg.r_0
    r_dot = xyz1[0] * xyz2[0] + xyz1[1] * xyz2[1] + xyz1[2] * xyz2[2]
    # Equation 6 from https://benthamopen.com/contents/pdf/TOAAJ/TOAAJ-5-26.pdf
    R = (C.EARTH_RADIUS_M ** 2 * ((r1 ** 2 + r2 ** 2) - 2 * r_dot)) - ((r1 ** 2) * (r2 ** 2)) + (r_dot) ** 2
    if R >= 0:  # or  sat_to_sat_disance(xyz1,xyz2) > 500:
        return False
    return True


def sat_to_sat_disance(xyz1, xyz2):
    return math.sqrt((xyz1[0] - xyz2[0]) ** 2 + (xyz1[1] - xyz2[1]) ** 2 + (xyz1[2] - xyz2[2]) ** 2)


def possible_sats(sat, sat_Group):
    visibile_sats = []
    for sat_cands in sat_Group:
        if sat_cands != sat:
            if sat_to_sat_visibility(sat, sat_cands):
                visibile_sats.append(sat_cands)
    return visibile_sats


def check_gs_visibility(sat, GS_Group):
    shortest_link_to_gs = 999999999999999999999999
    gs = None
    for i in range(1, len(GS_Group)):
        vis = math_util.check_visibility(GS_Group[i].location[0], sat.ssp_lon, GS_Group[i].location[1], sat.ssp_lat,
                                         sat.altitude)
        if vis == False:
            continue
        distance = math_util.distance_to_sat(GS_Group[i].location[0], sat.ssp_lon, GS_Group[i].location[1],
                                             sat.ssp_lat, sat.altitude)
        if distance < shortest_link_to_gs:
            shortest_link_to_gs = distance
            gs = GS_Group[i]
    return shortest_link_to_gs, gs


def short_interlink_route(sat_src, sat_target, gs_source, gs_target, sat_group):
    init_distance = math_util.distance_to_sat(gs_source.location[0], sat_src.ssp_lon, gs_source.location[1],
                                              sat_src.ssp_lat, sat_src.altitude)
    starting_route = route.route_option(sat_src, gs_source, init_distance, 0)
    final_distance = 0  # distance to DL for latency calculation
    num_itr = 0
    while True:
        # print("Itr # : " , num_itr)
        # print("#of hops: ",len(starting_route.route_link))
        if num_itr >= 900:
            return None, 0
            break
        num_itr = num_itr + 1
        possible_next_hop = possible_sats(starting_route.get_last_node(), sat_group)
        possible_next_hop.sort(key=lambda dis: sat_to_sat_disance(dis.xyz_r, starting_route.get_last_node().xyz_r))

        print("\n ----------------------------------------------------------------------------------------\n")
        for next_hop2 in possible_next_hop:
            distance_to_check = sat_to_sat_disance(starting_route.get_last_node().xyz_r, next_hop2.xyz_r)
            print("Distance from sat: " + starting_route.get_last_node().name + " to " + next_hop2.name + " is " + str(
                distance_to_check))

        for next_hop in possible_next_hop:
            if starting_route.sat_in_route(next_hop):  # Check if next hop is already in the route
                continue
            # source_tonext = sat_to_sat_disance(starting_route.get_last_node().xyz_r , next_hop.xyz_r)
            # if source_tonext < 50:
            #     continue
            if sat_to_sat_disance(next_hop.xyz_r, sat_target.xyz_r) < sat_to_sat_disance(
                    starting_route.get_last_node().xyz_r, sat_target.xyz_r):
                print(next_hop.name + "  is picked\n")
                starting_route.add_hop(next_hop)
                print("\n ----------------------------------------------------------------------------------------\n")
                break
        if starting_route.get_last_node().name == sat_target.name:
            # starting_route.print_route()
            final_distance = math_util.distance_to_sat(gs_target.location[0], sat_target.ssp_lon,
                                                       gs_target.location[1], sat_target.ssp_lat, sat_target.altitude)
            return starting_route, final_distance
    return starting_route, final_distance


def best_link(sat_coverage, max_hops, starting_gs, sat_group, gs_target):
    shortest_route = None
    shortest_total_distance = 999999999999999999999999
    route_for_plotting = []
    satellites_explored = {}  # name of satellite, total distance to get there........
    possible_routes = []
    possible_routes2 = []
    init_distance = math_util.distance_to_sat(starting_gs.location[0], sat_coverage.ssp_lon, starting_gs.location[1],
                                              sat_coverage.ssp_lat, sat_coverage.altitude)
    starting_route = route.route_option(sat_coverage, starting_gs, init_distance, 0)
    possible_routes.append(starting_route)
    # satellites_explored[sat_coverage.name] = starting_route
    for i in range(0, max_hops):
        print("hop number: ", i, "out of: ", max_hops)
        for j in range(len(possible_routes)):
            print("itr number : ", j, "out of: ", len(possible_routes))
            parent_node = possible_routes[j].get_last_node()
            parent_route = possible_routes[j]
            parent_node_next_hop_options = possible_sats(parent_node, sat_group)

            for possible_next_hop in parent_node_next_hop_options:
                new_route = copy.deepcopy(parent_route)
                if sat_to_sat_disance(possible_next_hop.xyz_r, gs_target.UL_sat.xyz_r) > sat_to_sat_disance(
                        new_route.get_last_node().xyz_r, gs_target.UL_sat.xyz_r):
                    continue
                if possible_next_hop.name in new_route.route_name:
                    # print("Sat: " , possible_next_hop.name , " already in my route. So continuing")
                    continue
                new_route.add_hop(possible_next_hop)
                if possible_next_hop.name in satellites_explored:
                    # print("Already Explored a route that lead to this satellite: ", possible_next_hop.name)
                    ## Path already acheived... lets check if this new path leads to a less total distance.....
                    if new_route.get_avg_dis() < satellites_explored[possible_next_hop.name]:
                        # print("My new route has a shorter path. I will remove the previous paths.")
                        for every_route in possible_routes2:
                            if every_route.sat_in_route(possible_next_hop.name):
                                # print("Removing this path: ")
                                # every_route.print_route()
                                possible_routes2.remove(every_route)
                    else:
                        continue

                if possible_next_hop.name == gs_target.UL_sat.name:
                    # route_for_plotting.append(new_route)
                    if new_route.get_avg_dis() < shortest_total_distance:
                        new_route.print_route()
                        shortest_total_distance = new_route.get_avg_dis()
                        shortest_route = new_route
                        # print("This Route is removed - target -LN")
                        # new_route.print_route()
                    continue

                satellites_explored[possible_next_hop.name] = new_route.get_avg_dis()
                possible_routes2.append(new_route)
                # print("New Route Added to My Possible Routes: ")
                # new_route.print_route()

        possible_routes = possible_routes2
        possible_routes2 = []
    return shortest_total_distance + gs_target._min_distance_to_sat, shortest_route.get_hops(), shortest_route, route_for_plotting


def elevation_and_azimuth(gs_location, sat_SSP):
    ele, azi = astrod.look_up_angles(gs_location[0], sat_SSP.ssp_lon, gs_location[1], sat_SSP.ssp_lat,
                                     sat_SSP.altitude + C.EARTH_RADIUS_KM)
    ele = math.degrees(ele)
    azi = math.degrees(azi)
    return ele, azi


def check_angle(angle):
    if angle > 360:
        angle = angle - (floor(angle / 360) * 360)
    return angle


def long_lat_distance(lat1, lat2, long1, long2):
    # Calculated via haversine formula
    # This uses the ‘haversine’ formula to calculate the great-circle distance between two points –
    # that is, the shortest distance over the earth’s surface (in KM)
    delta_lat = radians(lat2 - lat1)
    delta_long = radians(long2 - long1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    long1 = radians(long1)
    long2 = radians(long2)

    a = sin(delta_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(delta_long / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return (const.EARTH_RADIUS_KM * 10 ** 3 * c) / (10 ** 3)


# Evaluates wheter the station can or cannot see the sat, considering 0deg min elevation angle
def check_visibility(gs_lon, sat_lon, gs_lat, sat_lat, sat_altitude):
    gs_lon = radians(gs_lon)
    sat_lon = radians(sat_lon)
    gs_lat = radians(gs_lat)
    sat_lat = radians(sat_lat)
    magic_angle = acos(cos(gs_lat) * cos(sat_lat) * cos(sat_lon - gs_lon) + sin(gs_lat) * sin(sat_lat))
    if magic_angle <= acos(const.EARTH_RADIUS_KM / (sat_altitude + const.EARTH_RADIUS_KM)):
        return True
    return False


# Computes distance from gs to satellite without taking into account gs altitude
def distance_to_sat(gs_lon, sat_lon, gs_lat, sat_lat, sat_h):
    gs_lat = radians(gs_lat)
    sat_lat = radians(sat_lat)
    gs_lon = radians(gs_lon)
    sat_lon = radians(sat_lon)

    angle = acos(cos(gs_lat) * cos(sat_lat) * cos(sat_lon - gs_lon) + sin(sat_lat) * sin(gs_lat))
    re = const.EARTH_RADIUS_KM
    rs = re + sat_h
    d = sqrt(re ** 2 + rs ** 2 - 2 * re * rs * cos(angle))
    # alpha = acos(cos(Le)*cos(Ls)*cos(ls-le)+sin(Ls)*sin(Le))
    # num   = sin(alpha)
    # r_s = h_diff + const.EARTH_RADIUS_KM
    # denom = sqrt(1 + ((const.EARTH_RADIUS_KM/r_s)**2) - (2*(const.EARTH_RADIUS_KM/r_s) * cos(alpha)))
    # elevation = acos(num/denom)
    # d = const.EARTH_RADIUS_KM * (sqrt((r_s/const.EARTH_RADIUS_KM)**2 - (cos(elevation)**2))-sin(elevation))
    return d


# Computes angle of the triangle formed by three satellites
def angle_pointing(sat_v1, sat_vertice, sat_v2):
    direct_los = sat_v1.xyz_r - sat_vertice.xyz_r
    interference_los = sat_v2.xyz_r - sat_vertice.xyz_r
    dot = np.dot(direct_los, interference_los)
    d_direct = sat_to_sat_disance(sat_v1.xyz_r, sat_vertice.xyz_r)
    d_interference = sat_to_sat_disance(sat_v2.xyz_r, sat_vertice.xyz_r)
    angle_rad = np.arccos(dot / (d_direct * d_interference))
    return np.rad2deg(angle_rad)


# Checks if i_tx_sat causes interference to rx_sat when pointing to i_rx_sat
def check_interference(rx_sat, tx_sat, i_rx_sat, i_tx_sat, half_cone_angle=0.5):
    if not sat_to_sat_visibility(rx_sat, i_tx_sat):
        return False
    elif abs(angle_pointing(tx_sat, rx_sat, i_tx_sat)) > half_cone_angle:
        return False
    elif abs(angle_pointing(i_rx_sat, i_tx_sat, rx_sat)) > half_cone_angle:
        return False
    else:
        return True
