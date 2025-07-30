import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import math
import scipy as sp
from simulator.plot_utils import get_ax, templateFormat
import scipy.integrate as integrate
import simulator.constants as constants
from functools import partial

# Constants
cache_path = '../atmospheric_turbulence/cache'
plots_path = '../atmospheric_turbulence/plots'
font_line_sizes = templateFormat()

sim_params = {# 'h_list': np.arange(1, 2e4, 1e2),
              'h_list': np.logspace(0, 5.7, num=1000), #(0 to 500km)
              'rms_wind_speed_list': [10, 21, 30],
              'ground_Cn2_list': [1.7e-13, 1.7e-14],
              'center_frequencies_list': [140e9, 600e12]  # 140GHz, 0.5um laser
              }

"""
Gaussian beam parameters for:
- Plane Wave approximation: Theta = 1, Lambda = 0
- Spherical Wave approximation: Theta = Lambda = 0
"""


def hufnagel_valley_model(h_list, w=21, A=1.7e-14, f=None):
    """
    Hufnagel-Valley model of the atmospheric refraction index structure parameter as a function of altitude
    Args:
        w: rms windspeed in [m/s], typically 21m/s
        A: Nominal value of Cn2(0) at the ground in [m^2/3], Cn2(0m), typically 1.7e-14m^2/3 for IR
        h_list: Array of altitude values
        f: Frequency at which the model is applied, in Hz

    Returns:
        Array with Cn2 at every altitude
    """
    Cn2 = 0.00594 * (w / 27) ** 2 * (1e-5 * h_list) ** 10 * np.exp(- h_list / 1000) \
        + 2.7e-16 * np.exp(- h_list / 1500) + A * np.exp(- h_list / 100)

    if not f:
        return Cn2
    else:
        # Wavelength at which the HV model was developed (infrared laser 0.5um)
        lambda_ir = 0.5e-6
        lambda_thz = 300 / (f * 1e-6)

        # (dn_thz(T)/dT)^2 / (dn_ir(T)/dT)^2
        conversion_factor = (1 + 7.52e-3 / lambda_thz**2) ** 2 / (1 + 7.52e-3 / lambda_ir**2) ** 2

        return Cn2 * conversion_factor


def cn2_integral(w, A, h0, H):
    return integrate.quad(lambda h: hufnagel_valley_model(w, A, h), h0, H)


def rytov_variance(h0, H, fc, z, w=None, A=None):
    """
    Computes the Rytov variance given the Cn2 profile between h0 and H at frequency fc and zenith angle z
    Args:
        h0: height above ground level of the ground station [m]
        H: satellite altitude [m]
        fc: carrier frequency [Hz]
        z: zenith angle (angle from ground station zenith to satellite nadir) [deg]
    Returns:
        r_var: Rytov variance (float)
    """
    lambda_c = 300 / (fc * 1e-6)
    k = 2 * np.pi / lambda_c

    Cn2_integral = integrate.quad(lambda h: hufnagel_valley_model(h, f=fc, w=w, A=A) * (h-h0)**(5/6), h0, H)[0]

    r_var = 2.25 * k**(7/6) * (1/np.cos(np.deg2rad(z)))**(11/6) * Cn2_integral

    return r_var


def gamma_gamma_pdf(r_var):
    # Gamma-gamma distribution parameters
    alfa = 1 / (np.exp(0.49 * r_var / (1 + 1.11 * r_var ** (12 / 5)) ** (7 / 6)) - 1)
    beta = 1 / (np.exp(0.51 * r_var / (1 + 0.69 * r_var ** (12 / 5)) ** (5 / 6)) - 1)

    normalized_irradiance = np.linspace(0, 3, 1000)
    term1 = 2 * (alfa * beta)**((alfa + beta)/2) / (math.gamma(alfa) * math.gamma(beta))
    bessel_argument = 2 * np.sqrt(alfa * beta * normalized_irradiance)
    term2 = sp.special.kv(alfa - beta, bessel_argument)



if __name__ == '__main__':
    h_list = sim_params['h_list']
    rms_wind_speed_list = sim_params['rms_wind_speed_list']
    ground_Cn2_list = sim_params['ground_Cn2_list']
    fc_list = sim_params['center_frequencies_list']

    print(r'$\mu_0$ = {}'.format(cn2_integral(rms_wind_speed_list[0], ground_Cn2_list[0], h_list[0], h_list[-1])))

    fig = get_ax()
    plt.grid()
    file_ext = '.jpg'

    colors = ['r', 'b']
    # linestyles = ['solid', 'dashdot', 'dashed']

    # Cn2 model
    for i, fc in enumerate(fc_list):
        for j, A in enumerate(ground_Cn2_list):
            for k, w in enumerate(rms_wind_speed_list):
                y_values = hufnagel_valley_model(h_list, w=w, A=A, f=fc)
                plt.plot(h_list * 1e-3, y_values, color=colors[i],
                         label='A={}'.format(A) + r'm$^{2/3}$' + ', w={}m/s, f={:.2f}THz'.format(w, fc*1e-12),
                         markersize=5, markevery=None)

    plt.title(r'Hufnagel-Valley model of $C_n^2$', fontsize=font_line_sizes['subTitleSize'])
    plt.xlabel(r'Altitude $h [km]$', fontsize=font_line_sizes['axisLabelSize'])
    plt.ylabel(r'$C_n^2$', fontsize=font_line_sizes['axisLabelSize'])
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([1e-40, 1e-11])

    plt.legend()
    extension = 'Cn2_Hufnagel_Valley_model' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
    plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
    plt.show()
    plt.close()

    # Downlink Irradiance pdf in moderate to strong turbulence conditions from sea level to sat_h
    # Irradiance normalised variance (scintillation index)
    sat_h = 500e3  # Satellite altitude
    elevation_angle_list = np.linspace(0, 90)
    zenith_angles_list = 90-elevation_angle_list

    fig = get_ax()
    plt.grid()
    file_ext = '.jpg'

    A = ground_Cn2_list[-1]
    w = rms_wind_speed_list[1]

    for i, f in enumerate(fc_list):
        r_var = rytov_variance(0, sat_h, f, zenith_angles_list, w=w, A=A)
        # i_var = np.exp(0.49 * r_var / (1+1.11 * r_var**(12/5))**(7/6) + 0.51 * r_var / (1 + 0.69 * r_var**(12/5))**(5/6)) - 1
        i_var = r_var
        plt.plot(elevation_angle_list, i_var, color=colors[i],
                 label='f={:.2f}THz'.format(f * 1e-12),
                 markersize=5, markevery=None)
    plt.suptitle(r'Downlink Rytov variance $\sigma_R^2$', fontsize=font_line_sizes['subTitleSize'])
    plt.title(r'A={}'.format(A) + r'm$^{2/3}$' + ', w={}m/s'.format(w))
    plt.xlabel(r'Elevation angle $[^\circ]$', fontsize=font_line_sizes['axisLabelSize'])
    plt.ylabel(r'$\sigma_R^2$', fontsize=font_line_sizes['axisLabelSize'])

    plt.legend()
    extension = 'scintillation_index' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + file_ext
    plt.savefig(os.path.join(plots_path, extension.replace(' ', '_')))
    plt.show()
    plt.close()

    # Normalized irradiance pdf
    elev_angles_list2 = [10, 90]

    for i, f in enumerate(fc_list):
        for j, e in enumerate(elev_angles_list2):
            zenith_angle = 90-e
            r_var = rytov_variance(0, sat_h, f, zenith_angle, w=w, A=A)














    # Coherence radius
    # Downlink
    w = rms_wind_speed_list[0]
    A = ground_Cn2_list[0]
    mu_0 = cn2_integral(w, A, h_list[0], h_list[-1])
    zenith_angle = 0

    for fc in fc_list:
        k = 2 * np.pi * constants.SPEED_OF_LIGHT / fc

    # Uplink
