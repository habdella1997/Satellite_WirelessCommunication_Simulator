from simulator.simulation import SimulationObject, Simulation
import simulator.toolkit as tk
from datetime import datetime, timedelta, timezone
import simulator.astrodynamics as astrod
import math
import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import imageio
from tqdm import tqdm
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from simulator.link_budget import link_budget
import simulator.constants as c

from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from itertools import cycle, islice
import warnings
import time
import networkx as nx

'''
Quantities required to fully describe satellite position at time t:
    Eccentricity e
    semimajor axis a
    time of perigee tp (or mean anomaly M at a given time)
    RA of ascending node Omega
    Inclination i
    Argument of perigee w
'''


class Satellite(SimulationObject):
    def __init__(self, inc, right_asc, eccentricity, argperige, mean_anomly, revs_per_day, altitude, period,
                 semi_major_axis, date=datetime.now(), name=None, TLE_Line1=None, TLE_Line2=None, simulation=None,
                 orbit=None, fake=False):
        super().__init__(simulation, name)

        self.location = []  # [xcor, ycord, altitude]
        self.coverage_gs = []
        self.distance_left = 0
        self.distance_right = 0
        self.tle_date = date
        self.inc = inc
        self.right_asc = right_asc  # Omega, in degrees (right ascension of ascending node)
        self.eccentricity = eccentricity
        self.argperige = argperige  # In degrees
        self.mean_anomaly = mean_anomly
        self.revs_per_day = revs_per_day
        self.altitude = altitude
        self.period = period
        self.semi_major_axis = semi_major_axis
        self.TLE_Line1 = TLE_Line1
        self.TLE_Line2 = TLE_Line2
        self._SSP_Lat = 0
        self._SSP_Long = 0
        self._r_0 = 0
        self._x_0 = 0
        self._y_0 = 0
        self._xyr = 0
        self.xyz_r0 = 0
        self.xyz_rgec = 0
        self.orbit = orbit

        # Epoch time as datetime
        self.tle_datetime = date

    @property
    def ssp_lat(self):
        return self._SSP_Lat

    @ssp_lat.setter
    def ssp_lat(self, a):
        self._SSP_Lat = a

    @property
    def ssp_lon(self):
        return self._SSP_Long

    @ssp_lon.setter
    def ssp_lon(self, a):
        self._SSP_Long = a

    @property
    def r_0(self):
        return self._r_0

    @r_0.setter
    def r_0(self, a):
        self._r_0 = a

    @property
    def y_0(self):
        return self._y_0

    @y_0.setter
    def y_0(self, a):
        self._y_0 = a

    @property
    def x_0(self):
        return self._x_0

    @x_0.setter
    def x_0(self, a):
        self._x_0 = a

    @property
    def xyz_r(self):
        return self._xyr

    @xyz_r.setter
    def xyz_r(self, a):
        self._xyr = a

    # SSP  --> Sub Satellite Point. Time is a datetime object
    def update_SSP_predetermined_time(self, time):
        time_UTC = time
        year = time.year
        month = time.month
        day = time.day
        hour = time.hour
        minutes = time.minute
        seconds = time.second
        JD = tk.julianDay(year, month, day)
        min_after_midnight = tk.minutes_after_midnight(hour, minutes, seconds)

        # Calculate avg_angular_velocity:
        avg_angular_velocity = astrod.avg_angular_velocity(self.semi_major_axis)

        # Calculate Current Mean Anomaly (based on current time - time data recorded)
        t_perigee = self.tle_date
        # M2 = n * ((t_2-t_1) + M1/n) <-- t1 is datetime of TLE record, M1 is TLE Mean anomaly, t1 is current time
        time_diff = time_UTC - t_perigee
        mean_anomaly = (time_diff.total_seconds() + (math.radians(self.mean_anomaly) / avg_angular_velocity)) \
                       * avg_angular_velocity
        mean_anomaly_deg_checked = tk.check_angle(math.degrees(mean_anomaly))

        # Solve for Eccentric anomaly
        E_solved = astrod.eccentric_anomly(math.radians(mean_anomaly_deg_checked), self.eccentricity)
        E_solved_deg_checked = tk.check_angle(math.degrees(E_solved))

        # Find Polar Coordinates
        x_0, y_0, z_0, r_0 = astrod.polar_cord_orbital_system(self.semi_major_axis,
                                                              self.eccentricity,
                                                              math.radians(E_solved_deg_checked))
        # Solve for Transformation Matrices M1, M2
        M1 = astrod.orbital_to_GEC_transformation_matrix(self.right_asc, self.argperige, self.inc)
        xyz_geo = np.matmul(M1, np.transpose(np.array([x_0, y_0, z_0])))
        angle_gec = astrod.angle_Between_GEC_Rot_System(JD, min_after_midnight)
        M2 = astrod.GEC_to_Rotating_System_Transformation_Matrix(angle_gec)
        xyz_r = astrod.sat_cord_in_rot_system(M2, M1, x_0, y_0, z_0)
        SSP_Lat_Deg, SSP_Long_Deg, case = astrod.SSP_cord(xyz_r[0], xyz_r[1], xyz_r[2])
        # print('Mean anomaly: {}'.format(mean_anomaly_deg_checked))
        # print('E: {}'.format(E_solved_deg_checked))
        # print('OmegaTe: {}'.format(angle_gec))
        # print('x0y0z0 (r0): {}, {}, {}  ({})'.format(x_0, y_0, z_0, r_0))
        # print('T: {}'.format(time.strftime('%m/%d/%Y, %H:%M:%S')))
        # print('SSP_Lat_Deg/SSP_Long_Deg: {}/{}'.format(SSP_Lat_Deg, SSP_Long_Deg))
        self.ssp_lat = SSP_Lat_Deg
        self.ssp_lon = SSP_Long_Deg
        self.xyz_r = xyz_geo
        self.r_0 = r_0
        self.y_0 = y_0
        self.x_0 = x_0
        self.xyz_r0 = np.array([x_0, y_0, z_0])
        self.xyz_rgec = np.matmul(M1, np.transpose(self.xyz_r0))
        # print("From tk: ", [self.ssp_lat, self.ssp_lon], "\n")


class Constellation(SimulationObject):
    _ID = 0

    def __init__(self, simulation=None, name="Deffault name", orbits=None, satellites=None):
        super().__init__(simulation, name)
        self.satellites = []
        if orbits is None:
            self.orbits = []
            if satellites:
                for sat in satellites:
                    self.satellites.append(sat)
        else:
            self.orbits = orbits
            for orbit in orbits:
                for sat in orbit.satellites:
                    self.satellites.append(sat)

        Constellation._ID = Constellation._ID + 1

    @staticmethod
    def starlink(sim):
        const = Constellation(sim, 'Starlink')
        const.satellites = tk.setup_sats_from_tle('starlink_TLE.txt', sim)
        return const

    @staticmethod  # Starlink phase 1 according to wikipedia as of June 3rd 2022
    def starlink_phase1(sim=None):
        const = Constellation(sim, 'Starlink')
        for omega in np.arange(0, 360, 360 / 72):
            orbit = Orbit(sim, e=0.0, a=c.EARTH_RADIUS_M + 550e3, omega=omega, i=53, w=0)
            orbit.add_satellites(n_sats=22)
            const.add_orbit(orbit)

        for omega in np.arange(0, 360, 360 / 72):
            orbit = Orbit(sim, e=0.0, a=c.EARTH_RADIUS_M + 540e3, omega=omega, i=53.2, w=0)
            orbit.add_satellites(n_sats=22)
            const.add_orbit(orbit)

        for omega in np.arange(0, 360, 360 / 36):
            orbit = Orbit(sim, e=0.0, a=c.EARTH_RADIUS_M + 570e3, omega=omega, i=70, w=0)
            orbit.add_satellites(n_sats=20)
            const.add_orbit(orbit)

        for omega in np.arange(0, 360, 360 / 6):
            orbit = Orbit(sim, e=0.0, a=c.EARTH_RADIUS_M + 560e3, omega=omega, i=97.6, w=0)
            orbit.add_satellites(n_sats=58)
            const.add_orbit(orbit)

        for i, omega in enumerate(np.arange(0, 360, 360 / 10)):
            orbit = Orbit(sim, e=0.0, a=c.EARTH_RADIUS_M + 560e3, omega=omega, i=97.6, w=0)
            if i <= 5:
                orbit.add_satellites(n_sats=58)
            else:
                orbit.add_satellites(n_sats=43)
            const.add_orbit(orbit)

        return const

    def add_orbit(self, orbit):
        self.orbits.append(orbit)
        for sat in orbit.satellites:
            self.satellites.append(sat)

    def update_SSPs(self, time=None):
        if time is None:
            time = self.simulation.t_current
        for sat in self.satellites:
            sat.update_SSP_predetermined_time(time)

    def sat_pairwise_distance(self):
        """Computes distances between all satellites in the constellation at the current simulation time"""
        coords = []
        for sat in self.satellites:
            sat.update_SSP_predetermined_time(self.simulation.t_current)
            coords.append(sat.xyz_r)
        return pdist(coords, 'minkowski', p=3)

    def plot_constellation(self, show=False, llcrnrlon=-180, urcrnrlon=180, llcrnrlat=-90, urcrnrlat=90, time=None,
                           step=1, G=None, highlights=None, filename=None, orbits=False, track=False,
                           color_by_orbit=False, annotate=False, equator=False, file_extension='.jpg'):
        """Plots and updates the constellation at timestep t
        step: portion of satellites not plotted for visualization purposes
        """
        if time is None:
            time = self.simulation.t_current

        fig = plt.figure(figsize=(12, 6))
        m = fig.add_subplot(projection=ccrs.PlateCarree())
        # map zoom
        m.set_extent([llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat], ccrs.PlateCarree())

        # Map features
        # m.stock_img()
        # m.coastlines()
        m.add_feature(cfeature.LAND, facecolor='0.6')
        m.add_feature(cfeature.OCEAN, facecolor='0.9')
        # m.add_feature(cfeature.COASTLINE)
        # m.add_feature(cfeature.BORDERS, linestyle=':')
        # m.add_feature(cfeature.LAKES, alpha=0.5)
        # m.add_feature(cfeature.RIVERS)

        if equator:
            m.axhline(y=0, color='k', linestyle='dotted', linewidth=3)

        # plt.title('T: {}'.format(time.strftime('%m/%d/%Y, %H:%M:%S')))

        self.update_SSPs(time=time)

        # If network graph is provided
        if G is not None:
            pos = {}
            for node in G.nodes():
                pos[node] = np.array([self.satellites[node].ssp_lon, self.satellites[node].ssp_lat])
            # pos = spring_layout(G)

            alpha = 0.01 if len(self.satellites) > 1e3 else 0.3
            edges = nx.draw_networkx_edges(G, pos, alpha=alpha, width=0.5)
            eig_cent = nx.eigenvector_centrality(G)
            # nodes = nx.draw_networkx_nodes(G, pos, node_shape='>', node_color='r', edgecolors='k', node_size=[(eig_cent[node] - 0.95*min(eig_cent.values()))*4000 for node in G.nodes]) #Eigenvector centrality proportional 1652
            # nodes = nx.draw_networkx_nodes(G, pos, node_shape='>', node_color='r', edgecolors='k', node_size=[(eig_cent[node] - 0.95 * min(eig_cent.values())) * 1500 for node in G.nodes])  # Eigenvector centrality proportional 1652
            nodes = nx.draw_networkx_nodes(G, pos, node_shape='>', node_color='r', edgecolors='k',
                                           node_size=[np.exp((degree[1] - min(dict(G.degree).values()) + 1) / 20) for
                                                      degree in G.degree])  # Degree centrality proportional 1652
            # nodes = nx.draw_networkx_nodes(G, pos, node_shape='>', node_color='r', edgecolors='k', node_size=[np.exp((degree[1] - min(dict(G.degree).values()) + 1)/4) for degree in G.degree]) #Degree centrality proportional 166
            if highlights:
                aux_pos = {}
                for node in highlights:
                    aux_pos[node] = pos[node]
                degrees = [G.degree[node] for node in highlights]
                nx.draw_networkx_nodes(G, aux_pos, highlights, node_shape='>', node_color='c', edgecolors='k',
                                       node_size=[np.exp((degree - min(dict(G.degree).values()) + 1) / 20) for degree in
                                                  degrees])
                # nx.draw_networkx_nodes(G, aux_pos, highlights, node_shape='>', node_color='c', edgecolors='k', node_size=[np.exp((degree - min(dict(G.degree).values()) + 1)/4) for degree in degrees])

        else:
            if track:
                past_ts = [self.simulation.t_start + i * self.simulation.t_step for i in
                           range(int((time - self.simulation.t_start) / self.simulation.t_step))]
                for t in past_ts:
                    self.update_SSPs(time=t)
                    for sat in self.satellites[0::step]:
                        m.scatter(sat.ssp_lon, sat.ssp_lat, marker='o', color='k', s=1, zorder=99,
                                  transform=ccrs.PlateCarree())
            if color_by_orbit and self.orbits:
                # Plot satellites colored by the orbit the pertain
                _, _, colors = tk.templateColors(len(self.orbits))
                for i, orbit in enumerate(self.orbits):
                    for sat in orbit.satellites:
                        m.scatter(sat.ssp_lon, sat.ssp_lat, marker='o', color=colors[i], s=40, zorder=101,
                                  edgecolor='k',
                                  transform=ccrs.PlateCarree())

                if highlights:
                    for sat in highlights:
                        m.scatter(sat.ssp_lon, sat.ssp_lat, marker='o', color='m', s=40, zorder=101, edgecolor='k',
                                  transform=ccrs.PlateCarree())
            else:
                # plot satellites normally
                for sat in self.satellites[0::step]:
                    m.scatter(sat.ssp_lon, sat.ssp_lat, marker='o', color='r', s=40, zorder=100, edgecolor='k',
                              transform=ccrs.PlateCarree())
                if highlights:
                    _, _, colors = tk.templateColors(len(highlights))
                    for n in highlights:
                        sat = self.satellites[n]
                        m.scatter(sat.ssp_lon, sat.ssp_lat, marker='o', color=colors[n], s=40, zorder=101,
                                  edgecolor='k',
                                  transform=ccrs.PlateCarree())

            if annotate and self.orbits:
                const_id = 0
                for orbit in self.orbits:
                    for j, sat in enumerate(orbit.satellites):
                        m.annotate('Const_ID: {} \nOrb_ID:{}'.format(const_id, j), (sat.ssp_lon, sat.ssp_lat + 1),
                                   zorder=102, fontsize=7, weight='bold', ha='center', rotation='vertical', color='w')
                        const_id += 1

        if self.orbits and orbits:
            # colors = ['orange', 'green', 'yellow']
            for i, orbit in enumerate(self.orbits):
                lats, lons = orbit.get_trajectory()
                m.scatter(lons[::3], lats[::3], color='k', zorder=99, marker='o', s=10, transform=ccrs.PlateCarree())
                # m.scatter(lons[::3], lats[::3], color=colors[i], zorder=99, marker='o', s=10, transform=ccrs.PlateCarree())
                # m.plot(lons, lats, color='k', zorder=99,
                #       transform=ccrs.PlateCarree(), linestyle='dotted')

        # Miscelaneous
        m.axhline(y=50, color='k', linestyle='--', linewidth=2, zorder=100)
        m.axhline(y=-50, color='k', linestyle='--', linewidth=2, zorder=100)

        # Save to results folder
        if filename is not None:
            path = os.path.join(self.simulation.figures_folder, filename + file_extension)
        else:
            path = os.path.join(self.simulation.figures_folder,
                                "const_plot_{}".format(time.strftime("%Y%m%d%H%M%S")) + file_extension)
        plt.savefig(path, bbox_inches='tight')
        if show:
            plt.show()

        plt.close(fig)
        # return m

    def adjacency_matrix(self, weighted=False):
        n_sats = len(self.satellites)
        A = np.zeros((n_sats, n_sats))

        for i in range(n_sats):
            for j in range(n_sats):
                if tk.sat_to_sat_visibility(self.satellites[i], self.satellites[j]) and i != j:
                    A[i, j] = 1

        if not weighted:
            return A
        else:
            non_zero_elements_indices = np.transpose(np.nonzero(A))

            for sat_pair in non_zero_elements_indices:
                sat1 = self.satellites[sat_pair[0]]
                sat2 = self.satellites[sat_pair[1]]
                distance = tk.sat_to_sat_disance(sat1.xyz_r, sat2.xyz_r)
                _, datarate = link_budget(0.2, distance * 10 ** 3, 40e9, 47.85 * 2, 295e9, 0, None)
                A[sat_pair[0], sat_pair[1]] = datarate

            return A

    # Not working, ignore
    def identify_orbits(self):
        """identifies the orbits present in a constellation"""
        columns = ['sat name', 'inclination', 'RA asc. node', 'Arg. of perigee', 'eccentricity', 'semimajor axis',
                   'altitude']
        data = pd.concat(
            [pd.DataFrame(
                [[sat.name, sat.inc, sat.right_asc, sat.argperige, sat.eccentricity, sat.semi_major_axis, sat.altitude]]
                , columns=columns) for sat in self.satellites], ignore_index=True)

        # hist = data[columns[1:]].hist(bins=100)

        # separating features
        x = data[columns[1:]]
        # Scaling before PCA
        x = StandardScaler().fit_transform(x)
        # PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(x)
        data['PC1'] = principal_components[:, 0]
        data['PC2'] = principal_components[:, 1]

        # clustering algorithm comparison
        df = data.copy().to_numpy(), None
        # ============
        # Set up cluster parameters
        # ============
        plt.figure(figsize=(9 * 2 + 3, 10))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.05)
        plot_num = 1

        default_base = {'quantile': .3,
                        'eps': .3,
                        'damping': .9,
                        'preference': -200,
                        'n_neighbors': 10,
                        'n_clusters': 72 * 2 + 36 + 6,
                        'min_samples': 20,
                        'xi': 0.2,
                        'min_cluster_size': 0.1}

        # Modify algorithms parameters for each dataset if necessary
        datasets = [((data[['PC1', 'PC2']], None), {}, 'all')]

        for i_dataset, (dataset, algo_params, dataset_name) in enumerate(datasets):
            # update parameters with dataset-specific values
            params = default_base.copy()
            params.update(algo_params)

            X, y = dataset

            # estimate bandwidth for mean shift
            bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(
                X, n_neighbors=params['n_neighbors'], include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)

            # ============
            # Create cluster objects
            # ============
            ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
            two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
            ward = cluster.AgglomerativeClustering(
                n_clusters=params['n_clusters'], linkage='ward',
                connectivity=connectivity)
            spectral = cluster.SpectralClustering(
                n_clusters=params['n_clusters'], eigen_solver='arpack',
                affinity="nearest_neighbors")
            dbscan = cluster.DBSCAN(eps=params['eps'])
            optics = cluster.OPTICS(min_samples=params['min_samples'],
                                    xi=params['xi'],
                                    min_cluster_size=params['min_cluster_size'])
            affinity_propagation = cluster.AffinityPropagation(
                damping=params['damping'], preference=params['preference'])
            average_linkage = cluster.AgglomerativeClustering(
                linkage="average", affinity="cityblock",
                n_clusters=params['n_clusters'], connectivity=connectivity)
            birch = cluster.Birch(n_clusters=params['n_clusters'])
            gmm = mixture.GaussianMixture(
                n_components=params['n_clusters'], covariance_type='full')

            clustering_algorithms = (
                ('MiniBatchKMeans', two_means),
                ('AffinityPropagation', affinity_propagation),
                ('MeanShift', ms),
                ('SpectralClustering', spectral),
                ('Ward', ward),
                ('AgglomerativeClustering', average_linkage),
                ('DBSCAN', dbscan),
                ('OPTICS', optics),
                ('Birch', birch),
                ('GaussianMixture', gmm)
            )

            for i_algo, (name, algorithm) in enumerate(clustering_algorithms):
                t0 = time.time()

                # catch warnings related to kneighbors_graph
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="the number of connected components of the " +
                                "connectivity matrix is [0-9]{1,2}" +
                                " > 1. Completing it to avoid stopping the tree early.",
                        category=UserWarning)
                    warnings.filterwarnings(
                        "ignore",
                        message="Graph is not fully connected, spectral embedding" +
                                " may not work as expected.",
                        category=UserWarning)
                algorithm.fit(X)

                t1 = time.time()
                if hasattr(algorithm, 'labels_'):
                    y_pred = algorithm.labels_.astype(int)
                else:
                    y_pred = algorithm.predict(X)

                ### PCA subplot ###
                # plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
                # plt.subplot(2, len(clustering_algorithms) / 2, plot_num)
                plt.subplot(3, len(clustering_algorithms), plot_num)
                if i_dataset == 0:
                    plt.title(name, fontsize=10)

                # display dataset name at y axis first column
                if i_algo == 0:
                    plt.ylabel('PCA')

                colors = np.array(list(islice(cycle(['#000080', '#00ff00', '#ff0000',
                                                     '#f781bf', '#a65628', '#984ea3',
                                                     '#999999', '#e41a1c', '#dede00']),
                                              int(max(y_pred) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                plt.scatter(data['PC1'], data['PC2'], s=10, color=colors[y_pred],
                            alpha=0.4)

                # plt.xlim(-2.5, max(X[:, 0])+0.5)
                # plt.ylim(-2.5, max(X[:, 1])+0.5)
                plt.xticks(())
                plt.yticks(())
                plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                         transform=plt.gca().transAxes, size=15,
                         horizontalalignment='right')

                ### Inclination vs RA asc. node ###
                plt.subplot(3, len(clustering_algorithms), plot_num + len(clustering_algorithms))
                # display dataset name at y axis first column
                if i_algo == 0:
                    plt.ylabel('Avg data rate vs. Intergroup const.')
                plt.scatter(data['RA asc. node'], data['inclination'], s=10, color=colors[y_pred],
                            alpha=0.4)
                plt.xticks(())
                plt.yticks(())

                ### RA asc. node vs eccentricity ###
                plt.subplot(3, len(clustering_algorithms), plot_num + 2 * len(clustering_algorithms))
                # display dataset name at y axis first column
                if i_algo == 0:
                    plt.ylabel('Avg data rate vs. Intragroup const.')
                plt.scatter(data['RA asc. node'], data['eccentricity'], s=10, color=colors[y_pred],
                            alpha=0.4)
                plt.xticks(())
                plt.yticks(())

                plot_num += 1

        plt.show()


class Orbit(SimulationObject):
    def __init__(self, simulation, e, a, omega, i, w, initial_anomaly=0):
        super().__init__(simulation)
        self.e = e  # eccentricity
        self.a = a  # semimajor axis
        self.omega = omega  # Right ascension of ascending node, in degrees
        self.i = i  # inclination
        self.w = w  # argument of perigee, in degrees
        self.altitude = a - c.EARTH_RADIUS_M
        self.satellites = []
        self.orbital_period = astrod.period_from_semi_major_axis(a)
        self.revs_per_day = astrod.revs_per_day(T=self.orbital_period)
        self.initial_anomaly = initial_anomaly  # In degrees

    def add_satellites(self, n_sats=1):
        angular_separation = 360 / n_sats
        for j in range(n_sats):
            sat = Satellite(self.i, self.omega, self.e, self.w, self.initial_anomaly + j * angular_separation,
                            self.revs_per_day,
                            self.altitude, self.orbital_period, self.a, simulation=self.simulation,
                            date=self.simulation.t_start)
            self.satellites.append(sat)

    def get_trajectory(self):
        """Returns trajectory as a list of SSPs"""
        fake_sat = Satellite(self.i, self.omega, self.e, self.w, 0, self.revs_per_day, self.altitude,
                             self.orbital_period, self.a, simulation=self.simulation, date=self.simulation.t_start)

        # ts = np.arange(self.simulation.t_start, self.simulation.t_start + timedelta(seconds=self.orbital_period), timedelta(seconds=self.orbital_period/200),
        #                dtype='datetime64[ms]')
        fake_anomalies = np.arange(0, 360, 360 / 200)

        lats = []
        lons = []

        for m_a in fake_anomalies:
            fake_sat.mean_anomaly = m_a
            fake_sat.update_SSP_predetermined_time(time=self.simulation.t_start)
            lats.append(fake_sat.ssp_lat)
            lons.append(fake_sat.ssp_lon)

        # for t in ts:
        #     t_datetime = t.item()
        #     fake_sat.tle_date = t_datetime
        #     fake_sat.update_SSP_predetermined_time(t_datetime)
        #     lats.append(fake_sat.ssp_lat)
        #     lons.append(fake_sat.ssp_lon)

        return lats, lons


# RESIDUAL CODE FROM HERE
def test_video():
    constellation = Constellation.starlink(
        Simulation(t_start=datetime.now(), t_end=datetime.now() + timedelta(hours=1), t_step=timedelta(minutes=1)))
    images = []
    for t in tqdm(constellation.simulation.ts):
        constellation.plot_constellation(show=False, time=t)
        images.append(imageio.imread(os.path.join(constellation.simulation.figures_folder,
                                                  "const_plot_{}.jpg".format(t.strftime("%Y%m%d%H%M%S")))))
    imageio.mimsave(os.path.join(constellation.simulation.figures_folder, 'Video_{}_fps.mp4'.format(1)), images,
                    fps=6)


def detect_outliers():
    constellation = Constellation.starlink(
        Simulation(t_start=datetime(year=2021, month=12, day=2, hour=17),
                   t_end=datetime(year=2021, month=12, day=2, hour=17) + timedelta(hours=1),
                   t_step=timedelta(minutes=1)))
    prev_lons = [-181 for sat in constellation.satellites]
    for t in tqdm(constellation.simulation.ts[0:2]):
        constellation.update_SSPs(t)
        diffs = [[index, sat.ssp_lon - prev_lons[index]] for index, sat in enumerate(constellation.satellites)]
        outliers = list(filter(lambda diff: -100 < diff[1] < 0, diffs))
        prev_lons = [sat.ssp_lon for sat in constellation.satellites]
        for outlier in outliers:
            print('Inclination outlier {}: {}'.format(outlier[0], constellation.satellites[outlier[0]].inc))
        pass


if __name__ == '__main__':
    # sat = tk.setup_sats_from_tle('TLE_templete.txt')[0]
    # sat = tk.setup_sats_from_tle('satcom_course_example.txt')[0]
    const = Constellation.starlink_phase1()
    print('Starlink phase 1 constellation loaded correctly')
