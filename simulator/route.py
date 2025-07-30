import simulator.toolkit as tk


class route_option:
    def __init__(self,starting_satellite,starting_gs,route_distance_init,parent_index):
        self.starting_satellite = starting_satellite
        self.starting_gs = starting_gs
        self.route_link = []
        self.route_link.append(starting_satellite)
        self.route_name = []
        self.route_name.append(starting_satellite.name)
        self.route_distance = []
        self.route_distance.append(route_distance_init)
        self.number_of_hops = 0
        self.parent_index = parent_index
        self.total_distance = route_distance_init
    def add_hop(self,sat):
        self.route_link.append(sat)
        self.route_name.append(sat.name)
        prev_sat = self.route_link[len(self.route_link)-2]
        route_distance_hop = tk.sat_to_sat_disance(sat.xyz_r,prev_sat.xyz_r)
        self.route_distance.append(route_distance_hop)
        self.number_of_hops = self.number_of_hops +1
        self.total_distance = self.total_distance +route_distance_hop
    def get_hops(self):
        return self.number_of_hops
    def get_total_distance(self):
        return sum(self.route_distance)
    def get_last_node(self):
        return self.route_link[len(self.route_link)-1]
    def pop_node(self):
        self.route_link.pop()
        self.route_name.pop()
        self.route_distance.pop()
        self.number_of_hops = self.number_of_hops - 1
    def print_route(self):
        print("Satellite Route: ", self.route_name)
        print("Distance Route: ", self.route_distance)
    def get_avg_dis(self):
        sum = 0
        for route in self.route_distance:
            sum = sum + route
        return sum/len(self.route_distance)
    def sat_in_route(self, sat_name):
        for satellites in self.route_link:
            if satellites.name == sat_name:
                return True
        return False
