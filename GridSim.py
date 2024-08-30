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
    ml_attack_vectors = [[1.0499918485359072, 0.0001, -0.00084455132980042, 1.0499558175293406, 0.9999968726247114, 0.001, 1.05, 0.9999939348142032, 0.00043372624645374516, 1.05, 0.9999997527986004, -0.001, 1.0499999999999998, 1.0, -0.001, 1.0490521394246608, 1.0, 0.0008177260241593931], [0.95, 0.00014599473842194489, -0.0007693017879365356, 1.049980296098675, 0.9997234666636552, -0.0007085765741609701, 1.0499350695223644, 0.040511434336450956, 0.0007655737478168699, 1.05, 1.0, -0.0009875475001399142, 1.0498923821501391, 0.9999998876609555, -0.001, 1.0499208265730813, 0.9999637617013712, 0.001], [1.05, 0.0001, 0.0004881049807998631, 1.05, 0.9999967406237549, -0.00041376558089819074, 1.0499915403180966, 0.9999986299117376, -0.000974528308203119, 1.0499758402176038, 1.0, -0.001, 1.05, 1.0, -0.001, 1.0499694937705597, 0.9999991308831333, -0.000785060772984884], [1.0498052396828401, 0.00010011015516273611, 0.001, 1.05, 1.0, -0.000434665788667548, 1.0499836801690317, 0.9996835564916497, 0.001, 1.049980907782551, 0.9999985379169642, 0.0009930733036429171, 1.049921186534323, 1.0, -0.0009552781091016677, 1.05, 1.0, -0.001], [0.9500191588279145, 0.0001, -0.0009058690737835933, 1.05, 0.9998358017850938, 0.001, 1.0497953710594183, 0.5631515471307728, 0.001, 1.049979207486549, 1.0, -0.0003555267335980909, 1.049991346988847, 1.0, 0.000876074384785784, 1.049895072249448, 0.9999968090649372, 0.001], [1.0499357661472382, 0.0001, -0.0009034605584639367, 1.049957294834026, 1.0, 0.00042464594540332905, 1.0499026921844343, 0.9999774027004067, 0.0009127740365180655, 1.05, 1.0, -0.0008812942398350212, 1.05, 0.9999226685360623, -2.645645683108344e-05, 1.04996956858014, 1.0, -0.001], [1.0499865716921644, 0.00010000000000000002, -0.0007396550754814492, 1.05, 0.9999838440482194, 0.001, 1.0499873545453648, 1.0, 0.001, 1.05, 1.0, -0.0009773953164830345, 1.05, 1.0, -0.0009342322397160423, 1.049990563399782, 0.9999898353572818, 0.0009655003377482088], [1.049911762612934, 0.00011083147792361659, -0.0006990013384129555, 1.0499998806427697, 0.9999973493606512, -0.001, 1.049996921630266, 0.9999945199691959, -0.0006044659622651276, 1.05, 1.0, -0.001, 1.05, 1.0, -0.001, 1.05, 0.9999860902223319, -0.0007131428313477566], [1.0464312306260632, 0.00011183157669229795, -0.0008936645718903842, 1.05, 1.0, 0.0009686127076807877, 1.0499999999999998, 0.9999661654616117, -0.001, 1.0499995637411361, 0.9999972067517104, 0.000995519084978509, 1.0499951477397365, 0.9999990660814084, -0.0009268040867170073, 1.0499973410108292, 0.999999935802568, -0.00031108589240879953], [1.05, 0.0001, 0.0008949918175897046, 1.05, 0.9999318597971936, 0.001, 1.05, 0.9999712930369469, -0.0008409230158056539, 1.0499802375092036, 0.9999997313653898, -0.0009959625446394474, 1.05, 0.9999848420118687, 0.001, 1.049991832467094, 0.9999599290558077, 0.0009748365674369715]]
    i = 0
    # for i in range(10):
    for att_vec in ml_attack_vectors:
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
            attack_data = fdia.deep_learning_fdia_inject(att_vec, best_attack_busses, SMGW_data)
            # attack_data = fdia.targeted_generalized_fdia_liu(best_attack_busses, SMGW_data, net, H)
            parse_measurement(attack_data, net)
            run_state_estimation(net)
            differences = fdia.compute_differences(net.res_bus_est, correct_data)
            # Print the mean and max of the differences
            counter.loc[j] = [differences['d_p_mw'].mean(), differences['d_p_mw'].abs().max()]
        print(f"Iteration {i}: \nMean: ", counter.abs().mean().values, "\n", "Max: ", counter.abs().max().values)
        i += 1
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


