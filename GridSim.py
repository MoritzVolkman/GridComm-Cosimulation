import itertools
import json
import time
import numpy as np
import pandapower
import socket
import os
import math
import pandas as pd
import simbench as sb
import FDIA as fdia
import matplotlib.pyplot as plt
from pandapower.plotting.plotly import simple_plotly, pf_res_plotly
import Network
import time


def main():
    # Load the Simbench data and configure the grid
    sb_code = "1-LV-semiurb4--0-sw"
    net = sb.get_simbench_net(sb_code)

    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    for i in range(3): # normally should be 96, 3 just for testing
        # Do a power flow calculation for each time step
        apply_absolute_values(net, profiles, i)
        net.trafo.tap_pos = 1
        pandapower.runpp(net, calculate_voltage_angles=True)
        # Prints the head of the Powerflow results dataframe
        # print(net.res_bus.head())
        # create the measurement data for the time step
        SMGW_data = create_measurement(net)
        # parse measurements and run state estimation to see effect of FDIA
        send_to_network_sim(SMGW_data, i)
        received_data = receive_from_network_sim()
        parse_measurement(json.loads(received_data), net)
        run_state_estimation(net)
        correct_data = net.res_bus_est
        # Conduct FDIA on the measurement data
        attack_buses = [40, 41, 42]
        fdia.random_fdia(attack_buses, SMGW_data)
        fdia.plot_attack(net, attack_buses)
        # send_to_network_sim(SMGW_data, i)
        # parse the measurement data from the network simulator SMGW_data will be replaced by GO_data
        parse_measurement(SMGW_data, net) #replace SMGW_data with GO_data to incorporate network sim
        run_state_estimation(net)
        fdia.plot_differences(correct_data, net.res_bus_est)


def run_state_estimation(net):
    # Detects if Bad Data was detected and removes it
    try:
        bad_data = pandapower.estimation.chi2_analysis(net, init="slack")
        if bad_data:
            print("Bad Data Detected")
    except AttributeError:
        print("No Bad Data Detected")
    pandapower.estimation.remove_bad_data(net, init="slack")
    # Runs the state estimation
    try:
        pandapower.estimation.estimate(net, init="slack", calculate_voltage_angles=True, zero_injection="aux_bus")
    except ValueError:
        print("State Estimation Failed")


def parse_measurement(measurements, net):
    # Parse the measurement data from the network simulator
    # The measurement data is in the form of a json object
    # Extract the values Voltage, ActivePower and ReactivePower
    # Set as measurement for the according bus
    for measurement in measurements:
        voltage = measurement["MeasurementData"]["Voltage"]
        active_power = measurement["MeasurementData"]["ActivePower"]
        reactive_power = measurement["MeasurementData"]["ReactivePower"]
        bus = measurement["UserInformation"]["ConsumerID"]
        if voltage is not None:
            pandapower.create_measurement(net, "v", element_type="bus", value=voltage, std_dev=0.1, element=bus, side=0)
        if active_power is not None:
            pandapower.create_measurement(net, "p", element_type="bus", value=active_power, std_dev=0.1, element=bus, side=0)
        if reactive_power is not None:
            pandapower.create_measurement(net, "q", element_type="bus", value=reactive_power, std_dev=0.1, element=bus, side=0)
    # print(net.measurement)


def create_measurement(net, amount=1):
    # Get the values for the time step and put them into the measurement object
    # The measurement object is in the form of a json object
    # The json object is then passed to the network simulator, which will return it later
    # The measurement object will be used to update the grid model afterwards
    measurements = []
    for bus in net.res_bus.index:
        # Add measurement data for each bus, the values are taken from the power flow calculation
        # If amount is set below 1, random busses will not have any measurement data
        if np.random.rand() < amount:
            measurement = {
                "MeasurementData": {
                    "ActivePower": net.res_bus.loc[bus]["p_mw"],
                    "ReactivePower": net.res_bus.loc[bus]["q_mvar"],
                    "ApparentPower": "nA",
                    "PowerFactor": math.cos(net.res_bus.loc[bus]["va_degree"]),
                    "Voltage": net.res_bus.loc[bus]["vm_pu"],
                    "Current": "nA",
                    "Frequency": "nA",
                    "EnergyConsumption": "nA",
                    "MaximumDemand": "nA",
                    "MeterStatus": "nA",
                    "EventLogs": "nA"
                },
                "UserInformation": {
                    "ConsumerID": bus,
                    "ContractAccountNumber": f"CA{bus}",
                    "MeterPointAdministrationNumber": f"MPAN{bus}",
                    "AggregatorID": "Aggregator",
                    "SupplierID": "Supplier",
                    "DirectMarketerID": "DirectMarketer"
                }
            }
            measurements.append(measurement)
    return measurements


def send_to_network_sim(SMGW_data, timestep):
    # Creates a json file for each measurement object for the Network Simulator to access
    # The Network Simulator will then read the json file and return the measurement data
    for i, measurement in enumerate(SMGW_data):
        with open(f"./JSON/measurement_{timestep}_{i}.json", "w") as file:
            json.dump(measurement, file)
            file.close()

        # Send the measurement data files of the timestep to the network simulator
        # Send to port 8081, where the network simulator is listening
        Network.send_message("127.0.0.1", 8081 + i, json.dumps(measurement))


def receive_from_network_sim():
    # Listens to messages on port 8080
    # The messages are the measurement data from the Network Simulator
    # The messages are then parsed and returned
    # The messages are in the form of a file with json objects and multiple lines
    # The json object is then returned as a list of json objects

    message = Network.wait_for_message(8080)
    return message
    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #     s.bind((HOST, PORT))
    #     s.listen()
    #     conn, addr = s.accept()
    #     with conn:
    #         data = conn.recv(10240)
    #         print("Received Data from Comm-Sim")
    #         data = data.decode("utf-8").split("\n")
    #         complete_data = []
    #         for line in data:
    #             try:
    #                 j_line = json.loads(line)
    #                 complete_data.append(j_line)
    #             except json.JSONDecodeError:
    #                 print("Error decoding line, moving on")
    #                 continue
    #         return complete_data


def apply_absolute_values(net, absolute_values_dict, case_or_time_step):
    for elm_param in absolute_values_dict.keys():
        if absolute_values_dict[elm_param].shape[1]:
            elm = elm_param[0]
            param = elm_param[1]
            net[elm].loc[:, param] = absolute_values_dict[elm_param].loc[case_or_time_step]


def train_fdia():
    sb_code = "1-LV-semiurb4--0-sw"
    net = sb.get_simbench_net(sb_code)
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    # Create a counter that is incremented every time a bus is selected as the most effective bus for one of the four values
    # The counter is then used to select the most effective bus for the FDIA attack
    counter = pd.DataFrame(columns="mean max".split())
    # find the best attack busses for the Power FDIA
    best_attack_busses = [0, 1, 2, 8, 9, 40]
    # Create Dataframe for the training data
    ml_attack_vectors = [[0.5023904252186284, -75.48513248806981, 0.0027159030446879, 0.5023772373727942, -75.48512176931902, 0.0027159030446879, -0.023813605539262796, -75.48513248806981, 0.00034047372291199274, 0.3831896501861829, -75.48508233040766, 0.0027159030446879, -0.0238136055392628, -75.4850425168805, 0.0027159030446879, 0.5024075272068566, -75.48513213112842, 0.0025396571957809425], [-0.022092411468313415, -75.48513204626047, 0.0003635129625834609, -0.0238136055392628, -75.48466495153046, 0.0027159030446879, -0.0238136055392628, -75.48499703935767, -0.0015290993757105, -0.022169084517637227, -75.48513248806981, 0.0015899844162862282, 0.49858910204376533, -75.48513248806981, -0.0006405312204212496, 0.5024075272068566, -75.48473744367313, 0.0005219910940274107], [0.4911681901852966, -75.48513248806981, -0.0007691671944548414, -0.02312551667824045, -75.48487362144147, 0.001311672476157656, -0.02347388305802389, -75.48513248806981, 2.899620096920557e-05, -0.023267048930815028, -75.48454095996698, -0.00018635308402762936, 0.49386325236326106, -75.48513248806981, -0.0010520701298432488, 0.5016062803390072, -75.48506824737403, 0.0005895413919462616], [0.5024075272068566, -75.48513248806981, 0.0021501924046278703, -0.020219355714111358, -75.48497816837441, -0.000993331326311368, -0.0238136055392628, -75.48513000028996, -0.0009200083883400817, -0.02379866561372385, -75.48503067154061, -0.0005521332877924239, -0.023813605539262796, -75.4849188963283, -0.0013002909572126008, 0.5023703892101511, -75.48512997637678, 0.0012724364641738062], [0.5008788533001193, -75.48513248806981, -0.000834214338314886, -0.023691913247658347, -75.48511020385646, 0.0023695056059360306, -0.0238136055392628, -75.48513248806981, 0.0027159030446879, -0.0238136055392628, -75.48504660775876, 0.0006338899955534362, 0.49787974022952286, -75.48513248806981, 0.0008070243434939546, 0.502352509803484, -75.48487289328732, 0.0007917303091942911], [-0.023813605539262796, -75.48494727117193, -0.0015290993757105, -0.0238136055392628, -75.48513248806981, -0.000308769669209019, -0.0238136055392628, -75.48487428981646, 0.0013649297780810052, -0.023813605539262796, -75.48513248806981, -0.0015290993757105, -0.0238136055392628, -75.48504683982179, 0.0027159030446879, 0.501777214669688, -75.48511601943952, 0.0005628887817498501], [-0.0238136055392628, -75.48513248806981, 0.0007879627449199655, -0.0238136055392628, -75.4848362149774, 0.0012033062259602487, -0.0238136055392628, -75.48513248806981, 0.0027159030446879, -0.0238136055392628, -75.48513248806981, -0.0015290993757105, -0.020845757642909784, -75.48513248806981, -0.00022425152206482086, 0.5023030349873876, -75.48513248806981, -0.0010282123241437016], [0.5024075272068566, -75.48237807531672, 0.002592815633414392, 0.008121487867592828, -75.48506685532392, 0.0026609021711422962, -0.01836543060577468, -75.48510460406429, -0.0015290993757105, -0.023321387598808987, -75.48513248806981, 0.002017565618305868, -0.02031489134376958, -75.48405885890027, 0.0012348960519948456, 0.5024075272068566, -75.48458965537444, 0.0013953172454150612], [-0.022807526893800115, -75.48513248806981, -0.0015290993757105, -0.02381256950988799, -75.48513248806981, 0.00227068313452793, -0.0238136055392628, -75.48513248806981, -0.0009913360680236982, -0.0238136055392628, -75.48513248806981, -0.0010534245562118325, -0.0238136055392628, -75.48513248806981, 0.0016665581953298533, 0.5009100263419889, -75.48505706173013, 0.002572503537339245], [0.4851622944871191, -75.48421406598804, -0.00027353089160057063, -0.0238136055392628, -75.48513248806981, -0.0011748863589997225, -0.0238136055392628, -75.48513248806981, 0.0007854437678478773, -0.0238136055392628, -75.48513248806981, 0.0006197847514337259, -0.0238136055392628, -75.48513248806981, 0.0009503281480577391, 0.5024075272068566, -75.48513248806981, -0.0007772899905911719]]
    #for att_vec in ml_attack_vectors:
    for i in range(10):
        for j in range(96):
            start = time.time()
            # Take the load profile for a random timestep -> could be any other timestep
            apply_absolute_values(net, profiles, j)
            net.trafo.tap_pos = 1
            if j:
                pandapower.runpp(net, calculate_voltage_angles=True, init="results")
            else:
                pandapower.runpp(net, calculate_voltage_angles=True)
            # Jacobian Matrix is deleted after state estimation, so has to be taken here
            H = net._ppc["internal"]["J"].todense()
            SMGW_data = create_measurement(net)
            parse_measurement(SMGW_data, net)
            run_state_estimation(net)
            correct_data = net.res_bus_est
            # Let the FDIA attack the grid at selected busses
            # attack_data = fdia.deep_learning_fdia_inject(att_vec, best_attack_busses, SMGW_data)
            attack_data = fdia.random_generalized_fdia_liu(best_attack_busses, SMGW_data, net, H)
            parse_measurement(attack_data, net)
            run_state_estimation(net)
            differences = fdia.compute_differences(net.res_bus_est, correct_data)
            # Print the mean and max of the differences
            counter.loc[j] = [differences['d_p_mw'].mean(), differences['d_p_mw'].abs().max()]
        print(f"Iteration {i} Mean: ", counter.abs().mean().values, "\n", "Max: ", counter.abs().max().values)
    fdia.plot_attack(net, best_attack_busses)

def find_best_attack_busses():
    sb_code = "1-LV-semiurb4--0-sw"
    net = sb.get_simbench_net(sb_code)
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    # Create a counter that is incremented every time a bus is selected as the most effective bus for one of the four values
    # The counter is then used to select the most effective bus for the FDIA attack
    counter = pd.DataFrame(columns="vm_pu va_degree p_mw q_mvar".split())
    # find the best attack busses for the Power FDIA
    best_attack_busses = [0, 1, 2, 8, 9, 40]
    # Iterate over all time steps and show progress bar
    for j in alive_it(range(96)):
        start = time.time()
        # Take the load profile for a random timestep -> could be any other timestep
        apply_absolute_values(net, profiles, j)
        net.trafo.tap_pos = 1
        if j:
            pandapower.runpp(net, calculate_voltage_angles=True, init="results")
        else:
            pandapower.runpp(net, calculate_voltage_angles=True)
            bus_list = net.res_bus.index.to_list()
            bus_list.remove(129)
            for bus in best_attack_busses:
                bus_list.remove(bus)
            print(bus_list)
        SMGW_data = create_measurement(net)
        parse_measurement(SMGW_data, net)
        run_state_estimation(net)
        correct_data = net.res_bus_est
        effect = pd.DataFrame(columns="vm_pu va_degree p_mw q_mvar".split())
        # Let the FDIA attack the grid at every bus
        for bus in bus_list:
            fdia.random_fdia([bus], SMGW_data)
            parse_measurement(SMGW_data, net)
            run_state_estimation(net)
            differences = fdia.compute_differences(net.res_bus_est, correct_data).mean().transpose()
            # Add the mean of the differences to the effect dataframe
            effect = pd.concat([effect, differences], ignore_index=True, axis=1)
        # Plot the effect of the FDIA attack on the grid
        effect = effect.transpose().drop([0,1,2,3]).abs()
        effect.reset_index(drop=True, inplace=True)
        # Add the most effective bus to the counter
        print(j, time.time() - start)
        counter.loc[j] = effect.idxmax().values
    # Print the most effective bus for each value -> (Voltage: 42, Voltage Angle: 32, Power(both): 0)
    print(counter.mode())
    fdia.plot_attack(net, counter.mode().iloc[0])


if __name__ == "__main__":
    # main()
    # receive_from_network_sim()
    train_fdia()


