import json
import time
import numpy as np
import pandapower
import os
import simbench as sb

# Example measurement data as in the thesis
example_measurement = {
          "MeasurementData": {
            "ActivePower": 123.45,
            "ReactivePower": 67.89,
            "ApparentPower": 130.00,
            "PowerFactor": 0.95,
            "Voltage": 230,
            "Current": 5.4,
            "Frequency": 50,
            "EnergyConsumption": 1500.67,
            "MaximumDemand": 120,
            "MeterStatus": "ok",
            "EventLogs": [
              "Event1: Power Failure at 03:00",
              "Event2: Power Restoration at 03:10"
            ]
          },
          "UserInformation": {
            "ConsumerID": 0,
            "ContractAccountNumber": "CA7891011",
            "MeterPointAdministrationNumber": "MPAN987654",
            "AggregatorID": "AG87654321",
            "SupplierID": "SP12345678",
            "DirectMarketerID": "DM12345678"
          }
        }


def main():
    # Load the Simbench data and configure the grid
    sb_code = "1-LV-semiurb4--0-sw"
    net = sb.get_simbench_net(sb_code)
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    for i in range(3): # normally should be 96, 3 just for testing
        # Do a power flow calculation for each time step
        apply_absolute_values(net, profiles, i)
        net.trafo.tap_pos = 1
        pandapower.runpp(net)
        # print(net.res_bus.head())
        # create the measurement data for the time step
        SMGW_data = create_measurement(net)
        # print(SMGW_data)
        send_to_network_sim(SMGW_data, i)
        # send the measurement data to the network simulator
        # GO_data = send_to_network_sim(measurement)
        # parse the measurement data from the network simulator SMGW_data will be replaced by GO_data
        parse_measurement(SMGW_data, net) #replace SMGW_data with GO_data to incorporate network sim
        pandapower.estimation.remove_bad_data(net, init="slack")
        pandapower.estimation.estimate(net, init="slack", calculate_voltage_angles=True, zero_injection="aux_bus")
        # print(net.res_bus_est.head())


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
                    "PowerFactor": "nA",
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
        with open(f"measurement_{timestep}_{i}.json", "w") as file:
            json.dump(measurement, file)
            file.close()


def receive_from_network_sim(timeout=60):
    # Receives the measurement data from the Network Simulator
    # The Network Simulator will write the data into a json file
    # The function will read the json file and return the data
    # The function will wait for the timeout in seconds
    # If the timeout is reached, the function will return None
    # If the file is read, the function will return the data as a list of json objects
    # The function will return None if no file is found
    grid_data = []
    start = time.time()
    while time.time() - start < timeout:
        for file in os.listdir():
            if file.startswith("grid_") and file.endswith(".json"):
                with open(file, "r") as file:
                    measurement = json.load(file)
                    grid_data.append(measurement)
                    os.remove(file.name)
    if len(grid_data) > 0:
        return grid_data
    else:
        return None


def apply_absolute_values(net, absolute_values_dict, case_or_time_step):
    for elm_param in absolute_values_dict.keys():
        if absolute_values_dict[elm_param].shape[1]:
            elm = elm_param[0]
            param = elm_param[1]
            net[elm].loc[:, param] = absolute_values_dict[elm_param].loc[case_or_time_step]


if __name__ == "__main__":
    main()
    # receive_from_network_sim(timeout=60)