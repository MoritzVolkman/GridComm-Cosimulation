import json
import time
import numpy as np
import pandapower
import math
import pandas as pd
import simbench as sb
from pandapower import diagnostic
from pandapower import contingency

import FDIA as fdia
import matplotlib.pyplot as plt
import Network
import time


def main():
    # Load the Simbench data and configure the grid
    sb_code = "1-LV-semiurb4--0-sw"
    net = sb.get_simbench_net(sb_code)

    # let the netsim know about the amount of nodes
    Network.send_message("127.0.0.1", 10000, f"{len(net.bus)}")
    # Run the simulation
    # Takes net as input
    # Takes additional boolean value st_for_pf: True -> State Estimation for Power Flow Calculation, False -> Load Profile Data
    # Takes additional string value attack_mode: random, non-generalized, generalized, MLGA, None
    run_sim_with_stateest_for_powerflow(net)



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
    # Listens to messages on port 10000 for the measurement data from the network simulator
    message = Network.wait_for_message(10000)
    return message


def apply_absolute_values(net, absolute_values_dict, case_or_time_step):
    for elm_param in absolute_values_dict.keys():
        if absolute_values_dict[elm_param].shape[1]:
            elm = elm_param[0]
            param = elm_param[1]
            net[elm].loc[:, param] = absolute_values_dict[elm_param].loc[case_or_time_step]


def check_grid_health(net, verbose=False):
    diagnostic = pandapower.diagnostic(net, report_style="None")
    nminus1_cases = {"line": {"index": net.line.index.values}}
    contingencies = pandapower.contingency.run_contingency(net, nminus1_cases)
    if verbose:
        print(diagnostic)
        print(contingencies)
    else:
        if 'overload' in diagnostic:
            print(f" Overloads: {diagnostic['overload']}")
            print(contingencies)
        if 'isolated_sections' in diagnostic:
            print(f" Overloads: isolated sections: {diagnostic['isolated_sections']} \n ")
            print(contingencies)
            bus_counter = 0
            for isolated_section in diagnostic['isolated_sections']:
                for bus in isolated_section:
                    bus_counter += 1
            return bus_counter / len(net.bus)
        if "causes_overloading" in contingencies['trafo'] == [True]:
            print("Trafo is overloaded because of the following lines: \n")
            print(f"Index of cause {contingencies['trafo']['cause_index']} \n max loading percent {contingencies['trafo']['max_loading_percent']} \n")
            return 1.0
        else:
            print("Grid is healthy \n")
            return 0.0


def run_sim_with_stateest_for_powerflow(net, se_for_pf=False, attack_mode="random"):
    # Can be run with the state estimation for the power flow calculation or with load profile data
    # The attack mode can be set to None, random, non-generalized, generalized or MLGA
    # Load the Simbench data and configure the grid
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    liu_counter = 0
    grid_health = 0
    all_differences = []
    num_timesteps = 96
    if attack_mode == "MLGA":
        model, bounds = fdia.deep_learning_fdia_train_model()
        attack_vector = fdia.deep_learning_fdia_predict(model, bounds)
    for i in range(num_timesteps):
        net.trafo.tap_pos = 1
        if i:
            # apply the values from the state estimation to be used in the power flow calculation
            if se_for_pf:
                for bus in net.res_bus_est.index:
                    net["bus"].loc[bus, "p_mw"] = net.res_bus_est.loc[bus, "p_mw"]
                    net["bus"].loc[bus, "q_mvar"] = net.res_bus_est.loc[bus, "q_mvar"]
            else:
                apply_absolute_values(net, profiles, i)
            pandapower.runpp(net, calculate_voltage_angles=True)
        else:
            apply_absolute_values(net, profiles, i)
            pandapower.runpp(net, calculate_voltage_angles=True)
        SMGW_data = create_measurement(net)
        send_to_network_sim(SMGW_data, i)
        received_data = receive_from_network_sim()
        try:
            H = net._ppc["internal"]["J"].todense()
        except AttributeError:
            print("Jacobian Matrix not found")
            parse_measurement(json.loads(received_data))
            run_state_estimation(net)
            correct_data = net.res_bus_est
            grid_health += check_grid_health(net)
            attack_data = fdia.random_fdia([0, 1, 2, 8, 9, 40], SMGW_data)
            parse_measurement(attack_data, net)
        else:
            parse_measurement(json.loads(received_data))
            run_state_estimation(net)
            correct_data = net.res_bus_est
            grid_health += check_grid_health(net)
            match attack_mode:
                case "random":
                    attack_data = fdia.random_fdia([0, 1, 2, 8, 9, 40], SMGW_data)
                case "non-generalized":
                    attack_data = fdia.random_fdia_liu([0, 1, 2, 8, 9, 40], SMGW_data, net, H)
                case "generalized":
                    attack_data = fdia.random_generalized_fdia_liu([0, 1, 2, 8, 9, 40], SMGW_data, net, H)
                case "MLGA":
                    attack_data = fdia.deep_learning_fdia_inject(attack_vector, [0, 1, 2, 8, 9, 40], SMGW_data)
                case "None":
                    attack_data = SMGW_data
            parse_measurement(attack_data, net)
            liu_counter += 1
        run_state_estimation(net)
        differences = fdia.compute_differences(correct_data, net.res_bus_est)
        all_differences.append(differences)
    print(f"Percentage of Non-random attacks: {liu_counter/(i+1)}")
    print(f"Mean Effect: {fdia.plot_mean_and_std(all_differences)}")
    print(f"Grid Health: \n{grid_health/num_timesteps}")


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
    for j in range(96):
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
    # train_fdia()
    run_sim_with_stateest_for_powerflow()


