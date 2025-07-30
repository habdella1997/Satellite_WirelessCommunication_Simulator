import math
import simulator.constants as C
import numpy as np


def link_budget(p_tx, distance, max_bandwidth, antenna_gain, f_c, abs_loss, mod):
    p_tx = 10 * math.log10(p_tx)
    lambda_fc = C.SPEED_OF_LIGHT / f_c

    L_mixer = 7
    L_misc = 0

    spreading_loss = 20 * math.log10((4 * math.pi) / lambda_fc * distance)
    L_total = L_mixer + L_misc + abs_loss + spreading_loss

    NF_mixer = 6
    NF_LNA = 1
    G_LNA = 35

    NF = 10 * math.log10(10 ** (NF_mixer / 10) + (10 ** (NF_LNA / 10) - 1) / 10 ** (G_LNA / 10))

    B_efficiencies = [1, 2, 3, 4, 6, 8]
    Eb_N0_min = [10.6, 10.6, 14, 14.5, 18.8, 23]
    # if mod == "BPSK" or mod == "QPSK":
    #     Eb_N0 = Eb_N0_min[0]
    #     if mod == "BPSK":
    #         B_eff = B_eff_permod[0]
    #     else:
    #         B_eff = B_eff_permod[1]
    # elif mod == "8PSK":
    #     Eb_N0 = Eb_N0_min[1]
    #     B_eff = B_eff_permod[2]
    # elif mod == "16QAM":
    #     Eb_N0 = Eb_N0_min[2]
    #     B_eff = B_eff_permod[3]
    # elif mod == "64QAM":
    #     Eb_N0 = Eb_N0_min[3]
    #     B_eff = B_eff_permod[4]
    # elif mod == "256QAM":
    #     Eb_N0 = Eb_N0_min[4]
    #     B_eff = B_eff_permod[5]
    # else:
    #     print(mod)
    #     print("Not Supported, will go with 64 QAM")
    #     Eb_N0 = Eb_N0_min[3]
    #     B_eff = B_eff_permod[4]
    data_rates = []
    max_data_rate = -1
    p_r = 0
    for i in range(len(B_efficiencies)):
        B_eff = B_efficiencies[i]
        Eb_N0 = Eb_N0_min[i]
        Eb_N0_lin = 10 ** (Eb_N0 / 10)
        SNR_out = 10 * math.log10(Eb_N0_lin * B_eff)
        SNR_in = SNR_out + NF

        p_rx = p_tx + antenna_gain - L_total
        N = p_rx - SNR_in
        B = 10 ** (N / 10) / C.K / C.T
        if (B > max_bandwidth):
            B = max_bandwidth
        data_rate = B * B_eff
        if data_rate > max_data_rate:
            max_data_rate = data_rate
            p_r = p_rx
    return p_r, max_data_rate / (10 ** 6)


# p_tx and p_rx in dBm!!!
def rx_power(p_tx, distance, f_c, g_tx, l_abs=0, g_rx=None):
    if g_rx is None:
        g_rx = g_tx

    lambda_fc = C.SPEED_OF_LIGHT / f_c
    l_spreading = 20 * math.log10((4 * math.pi) / lambda_fc * distance)

    p_rx = p_tx + g_tx - l_spreading - l_abs + g_rx
    return p_rx


def snr(p_tx, distance, f_c, g_tx, bw, T, l_abs=0, g_rx=None, nf=0):
    p_rx = rx_power(p_tx, distance, f_c, g_tx, l_abs, g_rx)

    p_n = 10 * np.log10(C.BOLTZMAN * T * bw)

    snr = p_rx - p_n - nf

    return snr
