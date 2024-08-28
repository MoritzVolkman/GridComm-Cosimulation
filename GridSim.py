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
    ml_attack_vectors = [[-0.4387861194117782, 1265.6579417788366, 1.7095202931746514, 1.3253736606320166, 57.09706548202395, 0.015254962388122406, -3.525990042945379, 3.583428793695891, -1.0614315158975354, 3.557367214096606, 0.9284174794395893, -0.20246228510927816, 2.0933949640602068, -1.0993397581558755, 0.09649857916520244, 4.30192020068424, 6.728291270227092, 0.2935335365289294],[-0.09762665673864035, -0.1553075266923733, -3.679232121934628, 0.9482794169747338, -52.931188406676426, 13.17918077108424, -0.8752059763117089, 913.8077183335079, -1.8086225395401854, -46.73373012937805, 13.101947266688002, -1.408227191633174, -168.44861010796066, -4.4679332069654825, -2.006503882737631, 74.34682596296797, -2594.267718723237, -0.30035676226003627], [1.228909817440036, -217.86772014433006, -1.3941824482741314, 1.9183433963188625, -1.9181356672482794, 0.26108917739492776, -3.3289688869281937, 31.055410067670707, -2.4305798367492253, -0.6956899438437427, -2.165668984887502, 0.14354991419982377, -1.6873371612603127, -0.46541533132582313, 0.9937417794986341, 0.6784296352269369, -6.273085807941851, 0.8298789809403078], [1.0253650559254033, -10.61221135549944, -0.5816697650374542, -0.04545837628032309, 0.24182238504501125, -2.775219563432475, -0.41826101311939007, -0.7932403852716251, 0.5378464317978189, 3.08204428278822, -0.4378945539062312, 0.4099184615872077, -0.24910253832087587, 129.99194351242136, -1.802618785246599, 0.4929552424455872, -5.7400055560666035, -0.832989975222365], [-2.8358183977817255, 3.891943582213585, 1.1701588635290001, 2.304068770392082, 3.081429880089141, -0.1197414460211125, 1.7682014595467948, 0.46763904424715896, -3.9867482113042945, -22.769003623172367, 31059.49814483775, 2.1111863723719297, 1.8581521808443737, -21.029563688567837, -4.974923399024462, 0.6757427544027546, 0.48128599288443313, 0.9066089357312845], [-0.07500721103625568, 5.301077394732412, 2.572475299575159, 1.3125289934249404, 0.023630887738422257, 0.03794807821138213, 2.7268141212960915, 2.559402114055633, -0.3899178349746142, 5.925744996113906, 42737.30624500443, 2.132454823103411, 5.460568114980191, 1.1158761448627836, 1.9352627179744888, 7.417446785032827, 2.258579134874523, 2.4149543130684163], [0.5486743001420681, 0.7893965820267193, 1.418794522529899, 0.2989076267974704, -2.0906789581189806, 1.023664607867928, 1.3741754874973486, 0.7109750702476496, -0.9419175954687626, 0.9290426800777317, 1173.0516034347288, -3.9363669990974035, 4.468248544198573, 1.3062241645187884, 0.05816869664370461, 0.7621717820353142, -1.055288368953247, -0.5177778920066438], [0.3490568067536719, 1.6596308107445776, 1.3531477679510564, 2.377636022031874, 180.5715043562917, 0.7415313993677998, 0.5930552256569734, -0.6041722264789281, -1.5579408923277254, 7.658076426753499, 3.4844298092679855, 0.17150002277481058, 35.91410475817199, -0.11319011524516404, 1.3891160976167725, -1.53326623646677, 1.0654532891811939, 0.700818931204478], [1.8482036142065106, 0.9039463292819536, 2.4934319565144767, 0.034217401663091174, -4.565245401851052, -7.093413819812648, -6.236707704028456, 20.437715286566444, 1.934721615537562, -0.7070127429387686, 6.433923289126551, 1.961953510193894, 1.822925627398663, -4.757371616952808, 0.9421535943927168, -848.0544221712585, -11145.71849355082, 1.1240546995154026], [0.11667038059613753, 0.39940662418523787, 4.191633814272718, 7.13178385300657, 1.671508066494896, 7.195500830302686, -0.039284543701203524, 37.257331538233714, 1.440981803415208, 0.8134714373413244, 0.5024663983959256, -50.29518348316167, -0.9142742159980113, -0.7056453610030218, 0.9918771561466013, 9.709315919005242, 4.283453113536052, 107.50528440969393]]
    # Iterate over all time steps and show progress bar
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
            parse_measurement(attack_data, net)
            run_state_estimation(net)
            differences = fdia.compute_differences(net.res_bus_est, correct_data)
            # Print the mean and max of the differences
            counter.loc[j] = [differences['d_p_mw'].mean(), differences['d_p_mw'].abs().max()]
        print(f"Iteration {j} Mean: ", counter.abs().mean(), "\n", "Max: ", counter.abs().max())
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


