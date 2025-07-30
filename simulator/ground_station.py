import simulator.constants as C
import math
import simulator.astrodynamics as AD
import simulator.channel
import simulator.link_budget as link_budget
import simulator.toolkit as tk
import simulator.channel as channel


class GroundStation:
    def __init__(self, name, lat, lon, tx_power, gain, humidty_per, pressure, temp):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.internet_access = True
        self.tx_power = tx_power
        self.gain = gain
        self.humidty_per = humidty_per
        self.pressure = pressure
        self.temp = temp
        self._sats_in_coverage = []
        self._min_distance_to_sat = 99999999999999
        self.UL_sat = None

    def check_coverage(self, sat):
        # Check if satellite is within-coverage
        # sat_to_SSP_Distance = math_util.long_lat_distance(self.lon, sat.ssp_lat, self.lat, sat.ssp_lon)
        dd = tk.distance_to_sat(self.lat, sat.ssp_lon, self.lon, sat.ssp_lat, sat.altitude)
        if sat == self.UL_sat:
            self._min_distance_to_sat = dd
        if tk.check_visibility(self.lat, sat.ssp_lon, self.lon, sat.ssp_lat, sat.altitude):
            if sat not in self._sats_in_coverage:
                self._sats_in_coverage.append(sat)
            if dd < self._min_distance_to_sat:
                self._min_distance_to_sat = dd
                self.UL_sat = sat
        else:
            if sat in self._sats_in_coverage:
                self._sats_in_coverage.remove(sat)
                if sat == self.UL_sat:
                    self._min_distance_to_sat = 99999999999999
                    if len(self._sats_in_coverage) == 0:
                        self.UL_sat = None
                self.update_ULSAT()

    # Updates GS UL satellite from the available satellites in self._sats_in_coverage. RN chooses closest.
    def update_ULSAT(self):
        if self.UL_sat is not None:
            dd = tk.distance_to_sat(self.lat, self.UL_sat.ssp_lon, self.lon,
                                           self.UL_sat.ssp_lat, self.UL_sat.altitude)
            self._min_distance_to_sat = dd
        for sat in self._sats_in_coverage:
            dd = tk.distance_to_sat(self.lat, sat.ssp_lon, self.lon, sat.ssp_lat, sat.altitude)
            if dd < self._min_distance_to_sat:
                self._min_distance_to_sat = dd
                self.UL_sat = sat

    def elevation_and_azimuth(self):
        if self.UL_sat != None:
            ele, azi = AD.look_up_angles(self.lat, self.UL_sat.ssp_lon, self.lon, self.UL_sat.ssp_lat,
                                         self.UL_sat.altitude + C.EARTH_RADIUS_KM)
            ele = math.degrees(ele)
            azi = math.degrees(azi)
            return ele, azi
        else:
            return 666, 666

    # Not able to transmit UL if there is no self.UL_sat
    def can_transmit_UL(self):
        self.update_ULSAT()
        if self.UL_sat == None:
            return False
        return True

    def transmit_UL(self, antenna_tx_power, antenna_tx_gain, frequency, bandwidth, mod):
        """
        After checking that UL transmission is possible (i.e. Groun-station is within some satellites coverage), this function will simulate the
        communication link between the earth station and satellite.
        Step-1: Based on satellite location/footprint, calculate earth station antenna elevation angle and azimuth angle
        Step-2: Calculate Pathloss from calculated elevation angle in Step-1, latest-updated satellite altitude value, and frequency
        Step-3: calculate recieved power, and date_rate.
        """
        ele, azi = AD.look_up_angles(self.lat, self.UL_sat._SSP_Long, self.lon, self.UL_sat._SSP_Lat,
                                     (C.EARTH_RADIUS_KM + self.UL_sat.altitude))
        spread_L, abs_L = channel.path_loss(self.UL_sat.altitude, [frequency], math.degrees(ele), 0.25)
        print("spreading")
        p_rx, data_rate = link_budget.link_budget(antenna_tx_power, self._min_distance_to_sat, bandwidth,
                                                  antenna_tx_gain, frequency, abs_L, spread_L)
        return p_rx, data_rate, self._min_distance_to_sat
#     def check_sat_coverage(self,sat):
#         i = 0
#         for existing_sat in self._sats_in_coverage:
#             if sat.name == existing_sat.name:
#                 self._sats_in_coverage[i] = sat
#                 return True
#         return False
# #    location_temp = [GS_Longitude[i], GS_Latitude[i], GS_Altitude[i]]
#     def check_distance(self):
#         for sat in self._sats_in_coverage:
#             Le = math.radians(self.lon)
#             Ls = math.radians(sat.ssp_lat)
#             le = math.radians(self.lat)
#             ls = math.radians(sat.ssp_lon)
#             angle = math.acos(math.cos(Le)*math.cos(Ls)*math.cos(ls-le)+math.sin(Ls)*math.sin(Le))
#             re = 6370
#             h_diff  = 500
#             rs = re + h_diff
#             d = math.sqrt(re**2 + rs**2 - 2*re*rs * math.cos(angle))
#             if d < self._min_distance_to_sat:
#                 self._min_distance_to_sat = d
#                 self.UL_sat       = sat
#
#     def add_sat(self,sat):
#         self._sats_in_coverage.append(sat)
#         self.check_distance()
#     def coverage(self,sat,cover_rad):
#         distance = math_util.long_lat_distance(math.radians(self.lon),
#                                                math.radians(sat.ssp_lat),
#                                                math.radians(self.lat),
#                                                math.radians(sat.ssp_lon))
#         check_cov = self.check_sat_coverage(sat)
#         if distance < cover_rad:
#             if check_cov == False:
#                 self.add_sat(sat)
#             else:
#                 self.check_distance()
#         else:
#             if check_cov:
#                 for existing_sat in self._sats_in_coverage:
#                     if sat.name == existing_sat.name:
#                         self._sats_in_coverage.remove(existing_sat)
#                 self.check_distance()
#                 #self.UL_sat  = None
