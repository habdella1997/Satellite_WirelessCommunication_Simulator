import satellite
import ground_station
from prettytable import PrettyTable


def main_console():
    print("Control System")
    print("System Is Running In The Background By Main Thread")
    print("**************************************************************************")
    print("1 - Display Satellite FootPrint")
    print("2 - Display Earth-Station Look Up Table")
    print("3 - Simulate Communication System With Space Sattelite")
    print("4 - Change Earth-Station Locations and Properites")
    print("5 - Change/Add/Remove Satellite")
    print("6 - Check Satellite Coordinates")
    num = input("Please Enter A Command From The List Above [1-6]\n")
    return num

def command_1(sat_constellation):
    print("The Current Satellites In Your Constellation Are: ")
    sat_name_i_dic = {}
    i = 0
    for sat in sat_constellation:
        print(sat.name)
        sat_name_i_dic[sat.name] = i
        i = i+1
    sat_choice = input("Which Satellite Would You Like To Use To Plot Its Foot-Print: \n")
    while sat_choice in sat_name_i_dic == False:
        print("Satellite Does Not Exist.")
        sat_choice = input("Which Satellite Would You Like To Use To Plot Its Foot-Print: \n")
    current_satellite = sat_constellation[sat_name_i_dic[sat_choice]]
    return current_satellite

def command_2(GS_Group):
    x = PrettyTable()
    x.field_names = ["GS Name",
                     "GS Coordinates Long/Lat",
                     "Internet Access",
                     "Governing Satellite Name",
                     "Satellite SSP Long/Lat",
                     "Distance To Satellite",
                     "Humidty",
                     "Temperature",
                     "Pressure",
                     "Altitude"]
    for i in range(len(GS_Group)):
        cur_gs = GS_Group[i]
        sat_UL = "No Coverage Right Now"
        if cur_gs.UL_sat != None:
            x.add_row([cur_gs.name,
                       str(cur_gs.location[0]) +"/" + str(cur_gs.location[1]),
                       cur_gs.internet_access,
                       cur_gs.UL_sat.name,
                       str(cur_gs.UL_sat.ssp_lon) + "/" + str(cur_gs.UL_sat.ssp_lat),
                       cur_gs._min_distance_to_sat,
                       cur_gs.humidty_per,
                       cur_gs.temp,
                       cur_gs.pressure,
                       cur_gs.location[2]
                       ])
        else:
            x.add_row([cur_gs.name,
                       str(cur_gs.location[0])+"/"+str(cur_gs.location[1]),
                       cur_gs.internet_access,
                       sat_UL,
                       "N/A",
                       "N/A",
                       cur_gs.humidty_per,
                       cur_gs.temp,
                       cur_gs.pressure,
                       cur_gs.location[2]
                       ])

    print(x)
