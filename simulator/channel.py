import numpy as np
import math
import simulator.constants as c


def atm_stand(h):
    """
    # ----------- Reference Standard Atmospheres (Rec. ITU-R P.835) -----------
    #
    # This code provides expressions and data for reference standard atmospheres
    # required for the calculation of gaseous attenuation on Earth-space paths.
    #
    # -- Inputs:
    #            h_diff: Geometric height [km]
    # -- Outputs:
    #            T: Temperature [K]
    #            P: Pressure [hPa]
    #            rho: Water vapor density [g/m^3]
    #            e: Water vapor pressure [hPa]

    """
    h = np.array(h)
    hh = np.divide(np.multiply(6356.766, h), np.add(6356.766, h))
    N = len(h)

    H = [0, 11, 20, 32, 47, 51, 71, 84.852]
    L = [-6.5, 0, 1, 2.8, 0, -2.8, -2]
    TL = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65]
    PL = [1013.25, 226.3226, 54.74980, 8.680422, 1.109106, 0.6694167, 0.03956649]
    a = [1.340543e-6, -4.789660e-4, 6.424731e-2, -4.011801, 95.571899]

    H = np.array(H)
    L = np.array(L)
    TL = np.array(TL)
    PL = np.array(PL)
    a = np.array(a)

    T = np.zeros(N)  # Temperature [K]
    P = np.zeros(N)  # Pressure [hPa]

    for n in range(N):  # might cause issue, Matlab starts with index 1 and python starts with index 0
        if (hh[n] >= H[0]) and (hh[n] <= H[1]):
            T[n] = TL[0] + L[0] * (hh[n] - H[0])
            P[n] = PL[0] * math.pow((TL[0] / T[n]), 34.1632 / L[0])
        elif (hh[n] > H[1]) and (hh[n] <= H[2]):
            T[n] = TL[1] + L[1] * (hh[n] - H[1])
            P[n] = PL[1] * math.exp(-34.1632 * (hh[n] - H[1]) / TL[1])
        elif (hh[n] > H[2]) and (hh[n] <= H[3]):
            T[n] = TL[2] + L[2] * (hh[n] - H[2])
            P[n] = PL[2] * math.pow((TL[2] / T[n]), (34.1632 / L[2]))
        elif (hh[n] > H[3]) and (hh[n] <= H[4]):
            T[n] = TL[3] + L[3] * (hh[n] - H[3])
            P[n] = PL[3] * math.pow((TL[3] / T[n]), (34.1632 / L[3]))
        elif (hh[n] > H[4]) and (hh[n] <= H[5]):
            T[n] = TL[4] + L[4] * (hh[n] - H[4])
            P[n] = PL[4] * math.exp(-34.1632 * (hh[n] - H[4]) / TL[4])
        elif (hh[n] > H[5]) and (hh[n] <= H[6]):
            T[n] = TL[5] + L[5] * (hh[n] - H[5])
            P[n] = PL[5] * math.pow((TL[5] / T[n]), (34.1632 / L[5]))
        elif (hh[n] > H[6]) and (hh[n] <= H[7]):
            T[n] = TL[6] + L[6] * (hh[n] - H[6])
            P[n] = PL[6] * math.pow((TL[6] / T[n]), (34.1632 / L[6]))
        elif (h[n] >= 86) and (h[n] <= 91):
            T[n] = 186.8673
            P[n] = math.exp(np.polyval(a, h[n]))
        else:
            T[n] = 263.1905 - 76.3232 * math.sqrt(1 - ((h[n] - 91) / 19.9429) ** 2)
            P[n] = math.exp(np.polyval(a, h[n]))

    ## -- Water vapor density and pressure
    rho0 = 7.5  ## Standard ground-level water-vapor density [g/m^3]
    h0 = 2  ## Scale height [km]

    rho = rho0 * np.exp(-h / h0)  ## Water vapor density [g/m^3]
    e = rho * T / 216.7  ## Water vapor pressure [hPa] (Rec. ITU-R P.453)
    r = (e / P < 2e-6)  ## Mixing ratio
    r = r * 1  # convert from [True True False ....] to [1 1 0 ....]

    counter = 0
    for bool_check in r:
        if r[counter] == 1:
            e[counter] = 2e-6 * P[counter]
            rho[counter] = e[counter] * 216.7 / T[counter]
        counter = counter + 1

    return T, P, rho, e


def ATMDRY_ITU(f, p, e, T):
    """
    # ------------ Oxygen specific attenuation (Rec. ITU-R P.676) -------------
    #
    # This code provides the specific attenuation of oxygen.
    #
    # -- Inputs:
    #            f: Frequency [GHz]
    #            p: Dry air pressure [hPa]
    #            e: Water vapor partial pressure [hPa]
    #            T: Temperature [K]
    # -- Outputs:
    #            gamma_oxy: Oxygen specific attenuation [dB/km]

    # Spectroscopic data for oxygen attenuation
    """
    f0 = [50.474214, 50.987745, 51.503360, 52.021429, 52.542418, 53.066934, 53.595775, 54.130025, 54.671180, 55.221384,
          55.783815, 56.264774, 56.363399, 56.968211, 57.612486, 58.323877, 58.446588, 59.164204, 59.590983, 60.306056,
          60.434778, 61.150562, 61.800158, 62.411220, 62.486253, 62.997984, 63.568526, 64.127775, 64.678910, 65.224078,
          65.764779, 66.302096, 66.836834, 67.369601, 67.900868, 68.431006, 68.960312, 118.750334, 368.498246,
          424.763020, 487.249273, 715.392902, 773.839490, 834.145546]
    a1 = [0.975, 2.529, 6.193, 14.320, 31.240, 64.290, 124.600, 227.300, 389.700, 627.100, 945.300, 543.400, 1331.800,
          1764.600, 2120.100, 2363.700, 1442.100, 2379.900, 2090.700, 2103.400, 2438.000, 2479.500, 2275.900, 1915.400,
          1503.000, 1490.200, 1078.000, 728.700, 461.300, 274.000, 153.000, 80.400, 39.800, 18.560, 8.172, 3.397, 1.334,
          940.300, 67.400, 637.700, 237.400, 98.100, 572.300, 183.100]
    a2 = [9.651, 8.653, 7.709, 6.819, 5.983, 5.201, 4.474, 3.800, 3.182, 2.618, 2.109, 0.014, 1.654, 1.255, 0.910,
          0.621, 0.083, 0.387, 0.207, 0.207, 0.386, 0.621, 0.910, 1.255, 0.083, 1.654, 2.108, 2.617, 3.181, 3.800,
          4.473, 5.200, 5.982, 6.818, 7.708, 8.652, 9.650, 0.010, 0.048, 0.044, 0.049, 0.145, 0.141, 0.145]
    a3 = [6.690, 7.170, 7.640, 8.110, 8.580, 9.060, 9.550, 9.960, 10.370, 10.890, 11.340, 17.030, 11.890, 12.230,
          12.620, 12.950, 14.910, 13.530, 14.080, 14.150, 13.390, 12.920, 12.630, 12.170, 15.130, 11.740, 11.340,
          10.880, 10.380, 9.960, 9.550, 9.060, 8.580, 8.110, 7.640, 7.170, 6.690, 16.640, 16.400, 16.400, 16.000,
          16.000, 16.200, 14.700]
    a5 = [2.566, 2.246, 1.947, 1.667, 1.388, 1.349, 2.227, 3.170, 3.558, 2.560, -1.172, 3.525, -2.378, -3.545, -5.416,
          -1.932, 6.768, -6.561, 6.957, -6.395, 6.342, 1.014, 5.014, 3.029, -4.499, 1.856, 0.658, -3.036, -3.968,
          -3.528, -2.548, -1.660, -1.680, -1.956, -2.216, -2.492, -2.773, -0.439, 0.000, 0.000, 0.000, 0.000, 0.000,
          0.000]
    a6 = [6.850, 6.800, 6.729, 6.640, 6.526, 6.206, 5.085, 3.750, 2.654, 2.952, 6.135, -0.978, 6.547, 6.451, 6.056,
          0.436, -1.273, 2.309, -0.776, 0.699, -2.825, -0.584, -6.619, -6.759, 0.844, -6.675, -6.139, -2.895, -2.590,
          -3.680, -5.002, -6.091, -6.393, -6.475, -6.545, -6.600, -6.650, 0.079, 0.000, 0.000, 0.000, 0.000, 0.000,
          0.000]

    f0 = np.array(f0)
    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)
    a5 = np.array(a5)
    a6 = np.array(a6)

    th = 300 / T;

    # Line strength
    S = a1 * 1e-7 * p * th ** 3 * np.exp(a2 * (1 - th))
    # Width of the line
    df = a3 * 1e-4 * (p * th ** 0.8 + 1.1 * e * th)
    # Zeeman splitting of oxygen lines
    df = np.sqrt(df ** 2 + 2.25e-6)
    # Correction factor due to interferences effects in oxygen lines
    delta = (a5 + a6 * th) * 1e-4 * (p + e) * th ** 0.8
    # Line-shape factor
    fmins = f0 - f
    fplus = f0 + f
    Fmins = (df - delta * fmins) / (fmins ** 2 + df ** 2)
    Fplus = (df - delta * fplus) / (fplus ** 2 + df ** 2)
    F = f / f0 * (Fmins + Fplus)
    # Oxygen imaginary part refractivities
    Noxy = np.matmul(S, np.transpose(F))

    # Width parameter for Debye spectrum
    d = 5.6e-4 * (p + e) * th ** 0.8
    # Non-resonant Debye spectrum of oxygen below 10 GHz
    abs_debye = 6.14e-5 / (d * (1 + (f / d) ** 2))
    # Pressure-induced nitrogen attenuation above 100 GHz
    abs_nitro = 1.4e-12 * p * th ** 1.5 / (1 + 1.9e-5 * f ** 1.5)
    # Dry air continuum
    Nd = f * p * th ** 2 * (abs_debye + abs_nitro)

    # Oxygen specific attenuation [dB/km]
    gamma_oxy = 0.1820 * f * (Noxy + Nd)
    return gamma_oxy


def atmwvp_itu(f, p, e, T):
    """
    # ---------- Water vapor specific attenuation (Rec. ITU-R P.676) ----------
    #
    # This code provides the specific attenuation of water vapor.
    #
    # -- Inputs:
    #            f: Frequency [GHz]
    #            p: Dry air pressure [hPa]
    #            e: Water vapor partial pressure [hPa]
    #            T: Temperature [K]
    # -- Outputs:
    #            gamma_wvp: Water vapor specific attenuation [dB/km]
    """

    f0 = [22.235080, 67.803960, 119.995940, 183.310087, 321.225630, 325.152888, 336.227764, 380.197353, 390.134508,
          437.346667, 439.150807, 443.018343, 448.001085, 470.888999, 474.689092, 488.490108, 503.568532, 504.482692,
          547.676440, 552.020960, 556.935985, 620.700807, 645.766085, 658.005280, 752.033113, 841.051732, 859.965698,
          899.303175, 902.611085, 906.205957, 916.171582, 923.112692, 970.315022, 987.926764, 1780.000000]
    b1 = [0.1079, 0.0011, 0.0007, 2.273, 0.0470, 1.514, 0.0010, 11.67, 0.0045, 0.0632, 0.9098, 0.1920, 10.41, 0.3254,
          1.260, 0.2529, 0.0372, 0.0124, 0.9785, 0.1840, 497.0, 5.015, 0.0067, 0.2732, 243.4, 0.0134, 0.1325, 0.0547,
          0.0386, 0.1836, 8.400, 0.0079, 9.009, 134.6, 17506.0]
    b2 = [2.144, 8.732, 8.353, 0.668, 6.179, 1.541, 9.825, 1.048, 7.347, 5.048, 3.595, 5.048, 1.405, 3.597, 2.379,
          2.852, 6.731, 6.731, 0.158, 0.158, 0.159, 2.391, 8.633, 7.816, 0.396, 8.177, 8.055, 7.914, 8.429, 5.110,
          1.441, 10.293, 1.919, 0.257, 0.952]
    b3 = [26.38, 28.58, 29.48, 29.06, 24.04, 28.23, 26.93, 28.11, 21.52, 18.45, 20.07, 15.55, 25.64, 21.34, 23.20,
          25.86, 16.12, 16.12, 26.00, 26.00, 30.86, 24.38, 18.00, 32.10, 30.86, 15.90, 30.60, 29.85, 28.65, 24.08,
          26.73, 29.00, 25.50, 29.85, 196.3]
    b4 = [0.76, 0.69, 0.70, 0.77, 0.67, 0.64, 0.69, 0.54, 0.63, 0.60, 0.63, 0.60, 0.66, 0.66, 0.65, 0.69, 0.61, 0.61,
          0.70, 0.70, 0.69, 0.71, 0.60, 0.69, 0.68, 0.33, 0.68, 0.68, 0.70, 0.70, 0.70, 0.70, 0.64, 0.68, 2.00]
    b5 = [5.087, 4.930, 4.780, 5.022, 4.398, 4.893, 4.740, 5.063, 4.810, 4.230, 4.483, 5.083, 5.028, 4.506, 4.804,
          5.201, 3.980, 4.010, 4.500, 4.500, 4.552, 4.856, 4.000, 4.140, 4.352, 5.760, 4.090, 4.530, 5.100, 4.700,
          5.150, 5.000, 4.940, 4.550, 24.15]
    b6 = [1.00, 0.82, 0.79, 0.85, 0.54, 0.74, 0.61, 0.89, 0.55, 0.48, 0.52, 0.50, 0.67, 0.65, 0.64, 0.72, 0.43, 0.45,
          1.00, 1.00, 1.00, 0.68, 0.50, 1.00, 0.84, 0.45, 0.84, 0.90, 0.95, 0.53, 0.78, 0.80, 0.67, 0.90, 5.00]
    th = 300 / T

    f0 = np.array(f0)
    b1 = np.array(b1)
    b2 = np.array(b2)
    b3 = np.array(b3)
    b4 = np.array(b4)
    b5 = np.array(b5)
    b6 = np.array(b6)

    # Line strength
    S = b1 * 0.1 * e * (th ** 3.5) * np.exp(b2 * (1 - th))
    # Width of the line
    df = b3 * 1e-4 * (p * th ** b4 + b5 * e * th ** b6)
    # Doppler broadening of water vapor lines
    df = 0.535 * df + np.sqrt(0.217 * df ** 2 + 2.1316e-12 * f0 ** 2 / th)
    # print(df)
    # Line-shape factor
    fmins = f0 - f
    fplus = f0 + f
    Fmins = df / (fmins ** 2 + df ** 2)
    # print(Fmins)
    Fplus = df / (fplus ** 2 + df ** 2)
    F = f / f0 * (Fmins + Fplus)
    # Water vapor imaginary part refractivities
    Nwvp = np.matmul(S, np.transpose(F))

    # Water vapor specific attenuation [dB/km]
    gamma_wvp = 0.1820 * f * Nwvp
    return gamma_wvp


def path_loss(r, f, a, dh=0.1, h_starting=0):
    """
    # This code calculates atmopheric specific attenuation.
    #
    # -- Inputs:
    #            - r: Satellite altitude [km]
    #            - f: Frequency [GHz]
    #            - a: Elevation angle [deg]
    #            - dh: Height step [km]
    #            Standard atmospheric profiles which can be obtained through
    #            selecting one of the following functions (Rec. ITU-R P.835):
    #            - ATM_STAND: Reference standard atmosphere
    #            - ATM_LOW_LAT: Low-latitude reference atmosphere
    #            - ATM_SUM_MID: Summer-mid latitude
    #            - ATM_WIN_MID: Winter-mid latitude
    #            - ATM_SUM_HIGH: Summer-high latitude
    #            - ATM_WIN_HIGH: Winter-high latitude
    #            The atmospheric profiles provide the following:
    #              - T: Temperature [K]
    #              - P: Pressure [hPa]
    #              - rho: Water vapor density [g/m^3]
    #              - e: Water vapor pressure [hPa]
    #
    # -- Outputs:
    #            - Lspr(r,f,a): Free space transmission loss [dB]
    #            - Labs(r,f,a): Atmospheric absorption loss [dB]
    """
    # Discard a(a == 0) = 0.1; -> Elevation angle will never be 0
    R = c.EARTH_RADIUS_KM  # Earth radius [km]
    if not isinstance(f, list):
        f = [f]
    f = np.array(f)
    Nf = len(f)

    Lspr = np.zeros(Nf)
    Labs = np.zeros(Nf)

    h = np.linspace(h_starting, min(r, 100), math.floor(min(r, 100) / dh) + 1)
    Nh = len(h)

    # Propagation distance [m] (Rec. ITU-R S.1257)
    d = math.sqrt(R ** 2 * math.sin(math.radians(a)) ** 2 + 2 * R * r + r ** 2) - R * math.sin(math.radians(a))

    # Atmospheric profile (Rec. ITU-R P.835)
    T, P, dont_care, e = atm_stand(h)

    # Atmospheric refractive index (Rec. ITU-R P.453)
    N0 = 315  # Average value of atmospheric refractivity extrapolated to sea level
    h0 = 7.35  # Scale height [km]
    ref = 1 + N0 * 1e-6 * np.exp(-h / h0)

    gamma = np.zeros((Nf, Nh))  # Specific attenuation by atmospheric gases [dB/km]
    gamma_oxy = np.zeros((Nf, Nh))  # Oxygen specific attenuation [dB/km]
    gamma_wvp = np.zeros((Nf, Nh))  # Water vapor specific attenuation [dB/km]
    slant = np.zeros(Nh)  # Slant

    for nf in range(0, Nf):
        Lspr[nf] = 92.45 + 20 * math.log10(f[nf] * d)
        for nh in range(0, Nh):
            gamma_oxy[nf, nh] = ATMDRY_ITU(f[nf], P[nh] - e[nh], e[nh], T[nh])

            gamma_wvp[nf, nh] = atmwvp_itu(f[nf], P[nh] - e[nh], e[nh], T[nh])
            slant[nh] = math.sqrt(1 - (R / (R + h[nh]) * ref[0] / ref[nh] * math.cos(math.radians(a))) ** 2)
            gamma[nf, nh] = (gamma_oxy[nf, nh] + gamma_wvp[nf, nh]) / slant[nh]
        Labs[nf] = np.trapz(gamma[nf]) * dh

    return Lspr, Labs
