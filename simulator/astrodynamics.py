"""
Pyhton Tool for implementing helper functions for any geometrical related problems necessary for satellite tracking, satellite orbit, and earth coverage.
"""
import simulator.constants as C
import simulator.toolkit as tk
import numpy as np
import math



def sin(x):
    return math.sin(x)


def cos(x):
    return math.cos(x)


def period_of_revolution(h):
    """
    Calculates the period of revolution given h_diff in Km (height from Earth surface).
    Equation: p^2 = 4pir^3 / GM
    Constants:
        G = Constant of Gravitation = 6.67259*10^-11 Nm^2/kg^2
        M = Mass of Earth = 5.972 × 10^24 kg
    Variables:
        r: Distance from earth's center. r=R+h_diff. For Satellite orbiting earth in an eleptical fashion, r stands for semi-major axis (a) of the orbit.

    Back-Ground:
        Generally, the point where the satellite is the closest to the central body is called the periapsis, with the length to the central body usually denoted as r_p.
        The point where the satellite is the farthest away is called the apoapsis, and has the associated length r_a.
        The semi-major axis is on the line segment between the periapsis and the apoapsis, and it is half of the distance between them, that is a = (r_p+r_a)/2
    """
    return math.sqrt(
        (4 * math.pi ** 2 * ((C.EARTH_RADIUS_KM + h) * 10 ** 3) ** 3) / (C.GRAVITATIONAL_CONSTANT * C.EARTH_MASS))


def velocity_of_satellite(r_p, r_a):
    """
    Calculates and returns the velocity at perigee, and apogee
    """
    r_p = (r_p + C.EARTH_RADIUS_KM) * 10 ** 3
    r_a = (r_a + C.EARTH_RADIUS_KM) * 10 ** 3
    v_p = math.sqrt(2 * C.GRAVITATIONAL_CONSTANT * C.EARTH_MASS * r_a / (r_p * (r_a + r_p)))
    v_a = math.sqrt(2 * C.GRAVITATIONAL_CONSTANT * C.EARTH_MASS * r_p / (r_a * (r_a + r_p)))
    return v_p, v_a


def eccentricity_of_orbit(r_p, v_p):
    """
    The eccentricity of an ellipse refers to how flat or round the shape of the ellipse is.
    The more flattened the ellipse is, the greater the value of its eccentricity.
    The more circular, the smaller the value or closer to zero is the eccentricity. The eccentricity ranges between one and zero.
    """
    return ((r_p * v_p ** 2) / (C.GRAVITATIONAL_CONSTANT * C.EARTH_MASS)) - 1


def periapsis_distance(a, e):
    """
    Given semi-major axis a, and eccentricity of ellipse e, return the periapsis distance:
    """
    return a * (1 - e)


def apoapsis_distance(a, e):
    """
    Given semi-major axis a, and eccentricity of ellipse e, return the apoapsis distance:
    """
    return a * (1 + e)


def altitude_at_periapsis_apoasis(v1, r1, zenith_angle1):
    """
    A quadratic equation that returns two values:
        -> Smaller value is sat altitude at periapsis
        -> Larger value is sat altitude atapoasis
    This function takes in three parameters:
        **** the 1 denotes that the input variables are initial launch values
        v1: velocity of the satellite
        r1: satellite distance from the center of the earth
        zenith_angle1: the angle between the position and the velocity vectors
    Warning: This function will have a 1-3 km error
    """
    r1 = (r1 + C.EARTH_RADIUS_KM) * (10 ** 3)
    zenith_radian1 = math.radians(zenith_angle1)
    c = (2 * C.GRAVITATIONAL_CONSTANT * C.EARTH_MASS) / (r1 * v1 ** 2)
    ##print(const)
    B = c ** 2 - 4 * (1 - c) * (-1 * math.sin(zenith_radian1) ** 2)
    B = math.sqrt(B)
    RpRa1 = (((-1 * c + B) / (2 * (1 - c))) * r1) / 1000
    RpRa2 = (((-1 * c - B) / (2 * (1 - c))) * r1) / 1000
    return min(RpRa1, RpRa2) - C.EARTH_RADIUS_KM, max(RpRa1, RpRa2) - C.EARTH_RADIUS_KM


def eccentricity_from_launch_values(v1, r1, zenith_angle1):
    r1 = (r1 + C.EARTH_RADIUS_KM) * (10 ** 3)
    zenith_radian1 = math.radians(zenith_angle1)
    a = ((r1 * v1 ** 2 / (C.GRAVITATIONAL_CONSTANT * C.EARTH_MASS)) - 1) ** 2
    return math.sqrt(a * math.sin(zenith_radian1) ** 2 + math.cos(zenith_radian1) ** 2)


def true_anomaly_rad(v1, r1, zenith_angle1):
    """
    To pin down a satellite's orbit in space, we need to know the angle , the true anomaly, from the periapsis point to the launch point. This angle is given by true anamoly
    This function returns the value in radians
    EX: Calculates the angle from perigee point to launch point for the satellite
    """
    r1 = (r1 + C.EARTH_RADIUS_KM) * (10 ** 3)
    zenith_radian1 = math.radians(zenith_angle1)

    num = (r1 * v1 ** 2 / (C.GRAVITATIONAL_CONSTANT * C.EARTH_MASS)) * math.sin(zenith_radian1) * math.cos(
        zenith_radian1)
    den = (r1 * v1 ** 2 / (C.GRAVITATIONAL_CONSTANT * C.EARTH_MASS)) * math.sin(zenith_radian1) ** 2 - 1
    return (math.atan(num / den))


def true_anomaly_rad_2(e, E):
    inter = math.sqrt((1 + e) / (1 - e)) * math.tan(E / 2)
    return math.atan(inter) * 2


def true_anomaly_deg(v1, r1, zenith_angle1):
    return math.degrees(true_anomaly_rad(v1, r1, zenith_angle1))


def semi_major_axis(r, v):
    r = (r + C.EARTH_RADIUS_KM) * (10 ** 3)
    return 1 / ((2 / r) - (v ** 2 / (C.GRAVITATIONAL_CONSTANT * C.EARTH_MASS)))


def avg_angular_velocity(a, T = None):
    # a in meters
    if T:
        return (2 * math.pi) / T  # rads/sec
    else:
        mu = C.GRAVITATIONAL_CONSTANT * C.EARTH_MASS
        return math.sqrt(mu) / (a ** (3 / 2))


def mean_anomly(n, t, t_p):
    # Determines the location of satellite at arbitrary time t>t_p  (IN SECONDS), where t_p is time at perigee.
    # n is avg angular velocity (Can be calculated with knowing T(period of revolution) or a (semi-major-axis))
    N = n * (t - t_p)
    if N / (2 * math.pi) > 1:  # might need to check bounds for when it equals one (do you set to zero??)
        ratio = math.floor(N / (2 * math.pi))
        N = N - (ratio * 2 * math.pi)
    return N  # radians


def eccentric_anomly(M, e):
    # M is mean anolmly , e is eccentricity , M = E - e *sin(E)
    # solve by iteration

    E = M + e * math.sin(M)

    max_iters = 200
    iters = 0
    while iters < max_iters:
        dM = M - (E - e * math.sin(E))
        dE = dM / (1 - e * math.cos(E))
        E = E + dE
        if abs(dE) < 1e-14:
            return E
        iters += 1
    else:
        raise ValueError('Failed to converge')
    # E_prev = M
    # E = M
    # itr = 0
    # while (M != (E - e * math.sin(E))):
    #     E = M + e * math.sin(E_prev)
    #     E_prev = E
    #     itr = itr + 1
    #     if itr == 900:
    #         #print("YOUR ALGORITHM DID NOT CONVERGE Restart Program.... :( ")
    #         return E
    # print("Eccentric Anamoly")
    # print(math.degrees(E))


#  XYZ coordinates relative to orbital plane Z=0
def polar_cord_orbital_system(a, e, E):
    r_0 = a * (1 - e * (math.cos(E)))
    if e == 0:
        true_anomaly = E
    else:
        num = a * (1 - e ** 2) - r_0
        denom = e * r_0
        true_anomaly = math.acos(num / denom)
    # true_anomaly = math.radians(mu.check_angle(math.degrees(true_anomaly)))
    phi_1 = 2 * math.pi - true_anomaly

    # Whys use of delta_1, delta_2? Shouldn't true anomaly (cos and sin) give you the sign already?
    delta_1 = abs(true_anomaly - E)
    delta_2 = abs(phi_1 - E)

    phi_2 = 0

    if delta_1 > delta_2:
        phi_2 = phi_1
    else:
        phi_2 = true_anomaly

    return r_0 * math.cos(phi_2), r_0 * math.sin(phi_2), 0, r_0


# M1 and M2 matrices (wrong ones) in Prof. Ivica Kostanic, lecture 3 minute 18:35
# Correct M1 matrix in https://www.youtube.com/watch?v=ZiLxfVevkI8&ab_channel=CyberneticSystemsandControls minute 22:11
# M1 is the matrix to pas from GEC to orbital coordinate system, or Earth Centered Inertial system (ECI) to perfiocal system PQW
def orbital_to_GEC_transformation_matrix(omega, w, i):
    omega = math.radians(omega)  # RA ascending node
    w = math.radians(w)  # argument of perigee
    i = math.radians(i)

    element_1_1 = cos(omega) * cos(w) - sin(w) * cos(i) * sin(omega)
    element_1_2 = -1 * cos(omega) * sin(w) - sin(omega) * cos(i) * cos(w)
    element_1_3 = sin(omega) * sin(i)

    element_2_1 = sin(omega) * cos(w) + cos(omega) * cos(i) * sin(w)
    element_2_2 = -1 * sin(omega) * sin(w) + cos(omega) * cos(i) * cos(w)
    element_2_3 = -cos(omega) * sin(i)

    element_3_1 = sin(i) * sin(w)
    element_3_2 = sin(i) * cos(w)
    element_3_3 = cos(i)

    return np.array([[element_1_1, element_1_2, element_1_3], [element_2_1, element_2_2, element_2_3],
                     [element_3_1, element_3_2, element_3_3]])


def GEC_to_Rotating_System_Transformation_Matrix(arg):
    return np.array([[cos(arg), sin(arg), 0], [-1 * sin(arg), cos(arg), 0], [0, 0, 1]])


def angle_Between_GEC_Rot_System(JD, t):
    t_c = (JD - 2415020) / 36525
    alpha = 99.6909833 + (36000.7689 * t_c) + (0.00038708 * (t_c ** 2))
    # alpha = mu.check_angle(alpha)
    angle = alpha + 0.25068447 * t
    angle = tk.check_angle(angle)
    return math.radians(angle)


def sat_cord_in_rot_system(M2, M1, x_0, y_0, z_0):
    M = np.matmul(M2, M1)
    r = np.array([x_0, y_0, z_0])
    r = np.transpose(r)
    return np.matmul(M, r)


def SSP_cord(x_r, y_r, z_r):
    # lat = (math.pi / 2) - math.acos(z_r / (math.sqrt(x_r ** 2 + y_r ** 2 + z_r ** 2)))
    lat = math.asin(z_r / (math.sqrt(x_r ** 2 + y_r ** 2 + z_r ** 2)))
    long = 0
    case = 0
    if x_r >= 0 and y_r >= 0:
        case = 1
        # print("Case1")
        long = (-math.atan2(y_r, x_r))
    elif x_r < 0 and y_r >= 0:
        case = 2
        # print("Case2")
        long = math.pi + math.atan2(y_r, abs(x_r))
    elif x_r < 0 and y_r < 0:
        case = 3
        # print("Case3")
        long = (-1*math.pi / 2) - math.atan(abs(y_r) / abs(x_r))
    elif x_r > 0 and y_r < 0:
        case = 4
        # print("Case4")
        long = -1 * math.atan2(abs(y_r), x_r)
    else:
        case = case
        # print("SSP CORD function Unexpected Error Line 187")

    return math.degrees(lat), math.degrees(math.atan2(y_r, x_r)), case


def look_up_angles(gs_lon, sat_lon, gs_lat, sat_lat, r_sat):
    """
    r_s: distance to satellite in km
    le = longitude of earth Station
    ls = longitude of SSP
    Le = latitude of earth Station
    Ls = latitude of SSP
    """
    gs_lon = math.radians(gs_lon)
    sat_lon = math.radians(sat_lon)
    gs_lat = math.radians(gs_lat)
    sat_lat = math.radians(sat_lat)
    alpha = math.acos(cos(gs_lat) * cos(sat_lat) * cos(sat_lon - gs_lon) + sin(gs_lat) * sin(sat_lat))
    print("alpha", alpha)
    num = sin(alpha)
    denom = math.sqrt(1 + ((6370 / r_sat) ** 2) - (2 * (6370 / r_sat) * cos(alpha)))
    # print("num.denom\n")
    # print(num/denom)
    elevation = math.acos(num / denom)
    azimuth_angle_rads = 0

    lA = 0
    lB = 0
    LA = 0
    LB = 0
    A = "dummy"
    B = "dummy"

    if gs_lat or sat_lat >= 0:
        # Atleast one point is in the northern hemisphere!

        # B chosen to be north of A
        if sat_lat > gs_lat:
            # SSP is north of Earth-Station
            LB = sat_lat
            LA = gs_lat
            lB = sat_lon
            lA = gs_lon
            B = "SSP"
            A = "ES"
        else:
            LB = gs_lat
            LA = sat_lat
            lB = gs_lon
            lA = sat_lon
            B = "ES"
            A = "SSP"

        CC = abs(lA - lB)
        if math.degrees(CC) > 180:
            CC = math.radians(360 - math.degrees(CC))
        else:
            CC = CC

        alpha = math.atan2(sin(0.5 * (LB - LA)), (math.tan(0.5 * CC) * cos(0.5 * (LB + LA))))
        beta = math.atan2(cos(0.5 * (LB - LA)), (math.tan(0.5 * CC) * sin(0.5 * (LB + LA))))
        x = beta - alpha
        y = alpha + beta

        # In this case, B has to equal ES (that is the only option) but I added the and condition for clarity
        # (removing it wont affect anything)
        if A == "SSP" and B == "ES":
            if lA <= lB:  # A west of B
                print("case-1")
                azimuth_angle_rads = math.radians(360) - y
            else:  # B west of A
                print("case-2")
                azimuth_angle_rads = y

        elif A == "ES" and B == "SSP":
            if lA <= lB:
                # print("case-2")
                print("case3")
                azimuth_angle_rads = math.radians(360) - x
            elif lB < lA:
                print("case4")
                azimuth_angle_rads = math.radians(360) - x

    else:
        # Both Earth-Station and SSP are on the Souther Hemishphere

        # B chosen to be south of A
        if sat_lat < gs_lat:
            # SSP is north of Earth-Station
            LB = sat_lat
            LA = gs_lat
            lB = sat_lon
            lA = gs_lon
            B = "SSP"
            A = "ES"
        else:
            LB = gs_lat
            LA = sat_lat
            lB = gs_lon
            lA = sat_lon
            B = "ES"
            A = "SSP"

        CC = abs(lA - lB)
        if CC > math.radians(180):
            CC = 360 - CC
        else:
            CC = CC
            # print("Look Up Angle C-Calculation Line 235-238 Unexpected Error")

        alpha = math.atan2(sin(math.floor(0.5 * (abs(LB) - abs(LA)))),
                           (math.tan(0.5 * CC) * cos(0.5 * (abs(LB) + abs(LA)))))
        beta = math.atan2(cos(math.floor(0.5 * (abs(LB) - abs(LA)))),
                          (math.tan(0.5 * CC) * sin(0.5 * (abs(LB) + abs(LA)))))
        x = beta - alpha
        y = alpha + beta

        # In this case, B has to equal ES (that is the only option) but I added the and condition for clarity
        # (removing it wont affect anything)
        if A == "SSP" and B == "ES":
            if lA <= lB:  # A west of B
                print("case5")
                azimuth_angle_rads = math.radians(180) + y
            else:  # B west of A
                print("case6")
                azimuth_angle_rads = math.radians(180) - y
        elif A == "ES" and B == "SSP":
            if lA <= lB:
                print("case7")
                azimuth_angle_rads = math.radians(180) - x
            elif lB < lA:
                print("case8")
                azimuth_angle_rads = math.radians(180) + x

    return elevation, azimuth_angle_rads


def sat_height(rev_per_day):
    orbital_period = (86400 / rev_per_day)
    altitude = ((C.GRAVITATIONAL_CONSTANT * C.EARTH_MASS * orbital_period ** 2) / (4 * math.pi ** 2)) ** (1 / 3)
    altitude = altitude / (10 ** 3)
    altitude = altitude - C.EARTH_RADIUS_KM
    return altitude


def semi_major_axis_from_period(T):
    return ((C.EARTH_MASS * C.GRAVITATIONAL_CONSTANT / (4 * math.pi ** 2)) * (T ** 2)) ** (1 / 3)


def period_from_semi_major_axis(a):
    # A in meters
    return math.sqrt(4 * (math.pi ** 2) * (a ** 3)/(C.EARTH_MASS * C.GRAVITATIONAL_CONSTANT))


def revs_per_day(T=None, a=None):
    if T is not None:
        return 24 * 3600/T
    elif a is not None:
        T = period_from_semi_major_axis(a)
        return 24 * 3600 / T
    else:
        raise ValueError('Neither T or a was introduced')
