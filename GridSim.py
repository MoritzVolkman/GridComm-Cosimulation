import numpy as np
import pandapower
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandapower.topology as top
import pandapower.plotting as plot
import pandapower.control as control
import pandapower.networks as nw
import pandapower.timeseries as timeseries
from pandapower.timeseries.data_sources.frame_data import DFData
import simbench as sb

def main():
    # Load the Simbench data and configure the grid
    sb_code = "1-LV-semiurb4--0-sw"
    net = sb.get_simbench_net(sb_code)
    print(net)
    # df = pd.read_csv("timeseries.csv")
    # plot.simple_plot(net)
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
    """
    for i in range(0, len(df)):
        SMGW_data = create_measurement(net, i, df)
        GO_data = send_to_network_sim(measurement)
        parse_measurement(GO_data, net)
    """


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
    print(net.measurement)


def create_measurement(net, timestep, df):
    # Get the values for the time step and put them into the measurement object
    # The measurement object is in the form of a json object
    # The json object is then passed to the network simulator, which will return it later
    # The measurement object will be used to update the grid model afterwards
    data = df.iloc[timestep]
    measurements = []
    for bus in net["bus"].index:
        # Work in progress, has to be adapted once the CSV is available
        measurement = {
            "MeasurementData": {
                "ActivePower": data["ActivePower"],
                "ReactivePower": data["ReactivePower"],
                "ApparentPower": data["ApparentPower"],
                "PowerFactor": data["PowerFactor"],
                "Voltage": data["Voltage"],
                "Current": data["Current"],
                "Frequency": data["Frequency"],
                "EnergyConsumption": data["EnergyConsumption"],
                "MaximumDemand": data["MaximumDemand"],
                "MeterStatus": data["MeterStatus"],
                "EventLogs": data["EventLogs"]
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


def send_to_network_sim(SMGW_data):
    # Send the measurement data to the network simulator
    # The network simulator will return the data as received by the grid operator
    pass


if __name__ == "__main__":
    main()