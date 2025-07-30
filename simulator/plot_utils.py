import requests
import simulator.constants as C
# from mpl_toolkits.basemap import Basemap
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from random import randrange
from itertools import cycle, islice


def sat_path_image(Longitude, Latitude, image_name):
    path_str = 'color:0x0000ff|weight:5|'
    for i in range(0, len(Longitude)):
        if i == len(Longitude) - 1:
            path_str = path_str + str(Latitude[i]) + ',' + str(Longitude[i])
        else:
            path_str = path_str + str(Latitude[i]) + ',' + str(Longitude[i]) + '|'

    payload = {'center': "40.52,34.34", 'zoom': str(1), 'size': '640x640', 'path': path_str, 'key': C.GOOFLE_MAP_API}
    r = requests.get("https://maps.googleapis.com/maps/api/staticmap?", params=payload)
    print(r)
    f = open(image_name, 'wb')
    f.write(r.content)
    f.close()


def satellite_coverage_Map(long, lat, image_name):
    fig = plt.figure(figsize=(12, 9))
    m = Basemap(projection='mill',
                llcrnrlat=-90,
                urcrnrlat=90,
                llcrnrlon=-180,
                urcrnrlon=180,
                resolution='const')
    m.drawcoastlines()
    m.drawparallels(np.arange(-90, 90, 10), labels=[True, False, False, False])
    m.drawmeridians(np.arange(-180, 180, 30), labels=[0, 0, 0, 1])
    m.scatter(long, lat, latlon=True, s=250, c='red', marker='.', alpha=0.7, edgecolor='k', linewidth=1, zorder=2)
    plt.title('Basemap tutorial', fontsize=20)
    fig.savefig(image_name)


"""
https://maps.googleapis.com/maps/api/staticmap?center=63.259591,-144.667969&zoom=6&size=400x400
&markers=color:blue%7Clabel:S%7C62.107733,-145.541936&markers=size:tiny%7Ccolor:green%7CDelta+Junction,AK
&markers=size:mid%7Ccolor:0xFFFF00%7Clabel:C%7CTok,AK"&key=YOUR_API_KEY
"""


def satellite_earthStation_Map(Sat_Group, Earth_Station, image_name):
    fig = plt.figure(figsize=(12, 9))
    m = Basemap(projection='mill',
                llcrnrlat=-90,
                urcrnrlat=90,
                llcrnrlon=-180,
                urcrnrlon=180,
                resolution='const')
    m.drawcoastlines()
    m.drawparallels(np.arange(-90, 90, 10), labels=[True, False, False, False])
    m.drawmeridians(np.arange(-180, 180, 30), labels=[0, 0, 0, 1])
    es_lat = []
    es_long = []
    sat_cov_lat = []
    sat_cov_long = []
    sat_cov = []
    sat_lat = []
    sat_long = []

    for es in Earth_Station:
        es_lat.append(es.location[1])
        es_long.append(es.location[0])
        if es.UL_sat != None:
            sat_cov_lat.append(es.UL_sat.ssp_lat)
            sat_cov_long.append(es.UL_sat.ssp_lon)
            sat_cov.append(es.UL_sat)
    for sat in Sat_Group:
        if sat in sat_cov:
            continue
        sat_lat.append(sat.ssp_lat)
        sat_long.append(sat.ssp_lon)

    m.scatter(es_long, es_lat, latlon=True, s=250, c='red', marker='.', alpha=1, edgecolor='k', linewidth=1, zorder=2)
    m.scatter(sat_cov_long, sat_cov_lat, latlon=True, s=250, c='blue', marker='.', alpha=1, edgecolor='k', linewidth=1,
              zorder=2)
    m.scatter(sat_long, sat_lat, latlon=True, s=250, c='black', marker='.', alpha=1, edgecolor='k', linewidth=1,
              zorder=2)

    plt.title('Basemap tutorial', fontsize=20)
    fig.savefig(image_name)

    # plt.show()


def plot_route(route, src_gs, target_gs, sat_Group, fig_name):
    fig = plt.figure(figsize=(12, 9))
    m = Basemap(projection='mill',
                llcrnrlat=-90,
                urcrnrlat=90,
                llcrnrlon=-180,
                urcrnrlon=180,
                resolution='const')
    m.drawcoastlines()
    m.drawparallels(np.arange(-90, 90, 10), labels=[True, False, False, False])
    m.drawmeridians(np.arange(-180, 180, 30), labels=[0, 0, 0, 1])
    route_cords_long = []
    route_cords_lat = []
    sat_long = []
    sat_lat = []
    #
    for sat in route.route_link:
        route_cords_long.append(sat.ssp_lon)
        route_cords_lat.append(sat.ssp_lat)

    for sat in sat_Group:
        if sat in route.route_link:
            continue
        sat_long.append(sat.ssp_lon)
        sat_lat.append(sat.ssp_lat)

    m.scatter(route_cords_long, route_cords_lat, latlon=True, s=250, c='red', marker='.', alpha=1, edgecolor='k',
              linewidth=1, zorder=2)
    m.scatter(sat_long, sat_lat, latlon=True, s=100, c='orange', marker='.', alpha=0.05, edgecolor='k', linewidth=1,
              zorder=2)
    m.scatter([src_gs.location[0], target_gs.location[0]], [src_gs.location[1], target_gs.location[1]], latlon=True,
              s=250, c='blue', marker='.', alpha=1, edgecolor='k', linewidth=1, zorder=2)

    plt.title('Route Link', fontsize=20)
    fig.savefig('Testing_resl/Routes/' + fig_name)


def plot_routes(routes, sat_constellation):
    print(len(routes))
    fig = plt.figure(figsize=(12, 9))
    m = Basemap(projection='mill',
                llcrnrlat=-90,
                urcrnrlat=90,
                llcrnrlon=-180,
                urcrnrlon=180,
                resolution='const')
    m.drawcoastlines()
    m.drawparallels(np.arange(-90, 90, 10), labels=[True, False, False, False])
    m.drawmeridians(np.arange(-180, 180, 30), labels=[0, 0, 0, 1])
    colors_ = list(colors._colors_full_map.values())
    colors_.remove('#56ae57')
    sats = sat_constellation
    for route in routes:
        color = colors_[randrange(0, len(colors_))]
        colors_.remove(color)
        size = randrange(150, 300)
        alp = randrange(1, 5) / 10
        long = []
        lat = []
        for sat in route.route_link:
            long.append(sat.ssp_lon)
            lat.append(sat.ssp_lat)
            for burger in sats:
                if burger.name == sat.name:
                    sats.remove(burger)

        m.scatter(long, lat, latlon=True, s=size, c=color, marker='.', alpha=alp, edgecolor='k', linewidth=1, zorder=2)
    long2 = []
    lat2 = []
    for satellite in sats:
        long2.append(satellite.ssp_lon)
        lat2.append(satellite.ssp_lat)
    print("HEllo")
    m.scatter(long2, lat2, latlon=True, s=250, c='red', marker='.', alpha=0.1, edgecolor='k', linewidth=1, zorder=2)
    plt.title('Route Link', fontsize=20)
    fig.savefig("all_routes.png")


def plot_2sats(sat1, sat2, sat_group):
    fig = plt.figure(figsize=(12, 9))
    m = Basemap(projection='mill',
                llcrnrlat=-90,
                urcrnrlat=90,
                llcrnrlon=-180,
                urcrnrlon=180,
                resolution='const')
    m.drawcoastlines()
    m.drawparallels(np.arange(-90, 90, 10), labels=[True, False, False, False])
    m.drawmeridians(np.arange(-180, 180, 30), labels=[0, 0, 0, 1])
    long = []
    lat = []
    for sat in sat_group:
        if sat == sat1 or sat == sat2:
            continue
        long.append(sat.ssp_lon)
        lat.append(sat.ssp_lat)
    m.scatter([sat1.ssp_lon, sat2.ssp_lon], [sat1.ssp_lat, sat2.ssp_lat], latlon=True, s=250, c='red', marker='.',
              alpha=0.9, edgecolor='k', linewidth=1, zorder=2)
    m.scatter(long, lat, latlon=True, s=200, c='blue', marker='.', alpha=0.6, edgecolor='k', linewidth=1, zorder=2)
    plt.title('Satellite to Satellite Visibility Test', fontsize=20)
    fig.savefig("satsloc.png")


def get_ax(figsize=(10, 6)):
    fig = plt.figure(figsize=figsize)  # inches, [7.2, 4] perfect for docs
    ax = fig.add_subplot(1, 1, 1)
    return ax


def get_fig_and_ax(figsize=(10, 6)):
    fig = plt.figure(figsize=figsize)  # inches, [7.2, 4] perfect for docs
    ax = fig.add_subplot(1, 1, 1)
    return fig, ax


# function to print subscript
def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)


# function to print superscript
def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)


# Template data structure for colors and fontsize
def templateColors(iterable_size=None, reverse=False, id=0):
    if id==0:
        colors = {"Red": '#e41a1c',
                  "Blue": '#377eb8',
                  "Green": '#4daf4a',
                  "Purple": '#984ea3',
                  "Orange": '#ff7f00',
                  "Pink": '#f781bf',
                  "Olive": '#808000',
                  "Brown": '#a65628',
                  "Gray": '#999999'}
    elif id==1:
        colors = {"Red": '#cc0036',
                   "Blue": '#007db3',
                   "Green": '#00CD6C',
                   "Purple": '#AF58BA',
                   "Yellow": '#FFC61E',
                   "Orange": '#F28522',
                   "Pink": '#f781bf',
                   }
    elif id==2:
        colors = {"Red": '#e31a1c',
                   "Blue": '#1f78b4',
                   "Green": '#00CD6C',
                   "Purple": '#AF58BA',
                   "Yellow": '#FFC61E',
                   "Orange": '#F28522',
                   "Pink": '#f781bf',
                   }

    if iterable_size:
        iterable = np.array(list(islice(cycle(list(colors.values())), iterable_size + 1)))
        if reverse:
            iterable = np.flip(iterable)
    else:
        iterable = None

    return colors, list(colors.keys()), iterable


def templateColorsSequential(iterable_size=None):
    colors = {"0": [0 / 255, 63 / 255, 92 / 255],
              "1": [47 / 255, 75 / 255, 124 / 255],
              "3": [102 / 255, 81 / 255, 145 / 255],
              "4": [160 / 255, 81 / 255, 149 / 255],
              "5": [212 / 255, 80 / 255, 135 / 255],
              "6": [249 / 255, 93 / 255, 106 / 255],
              "7": [245 / 255, 97 / 255, 99 / 255],
              "8": [255 / 255, 124 / 255, 67 / 255],
              "9": [255 / 255, 166 / 255, 0 / 255]}

    if iterable_size:
        iterable = np.array(list(islice(cycle(list(colors.values())), iterable_size + 1)))
    else:
        iterable = None

    return colors, list(colors.keys()), iterable


def templateColorsPairs(iterable_size=None):
    colors = {"Red_0": '#fb9a99',
              "Red_1": '#e31a1c',
              "Green_0": '#b2df8a',
              "Green_1": '#33a02c',
              "Blue_0": '#a6cee3',
              "Blue_1": '#1f78b4',
              }

    if iterable_size:
        iterable = np.array(list(islice(cycle(list(colors.values())), iterable_size + 1)))
    else:
        iterable = None

    return colors, list(colors.keys()), iterable


# Template data structure for font and Line sizes
def templateFormat():
    sizes = {'linewidth': 3,
             'ticksLabelSize': 20,
             'axisLabelSize': 25,
             'legendSize': 18,
             'titleSize': 30,
             'subTitleSize': 18,
             'fontname': 'Times New Roman',
             }

    return sizes
