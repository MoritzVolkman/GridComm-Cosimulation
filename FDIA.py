import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import simbench as sb
import pandapower

# Injects false data into the measurement data for the specified busses
# The mode specifies the type of false data that is injected
# Mode 0: Random values
# Mode 1: Uninformed obstruction of the system - trying to make the system unstable
# Mode 2: Informed obstruction of the system - trying to make the system unstable with calculations

def random_fdia(busses, measurements):
    # Select the JSON object from the list where the "ConsumerID" matches the bus
    # Add random values for ActivePower, ReactivePower and Voltage
    # Random Reactive Power should be in the range 10e-05 to -10e-05
    # Random Active Power should be in the range 10e-06 to 10e-03
    # Voltage should be close to 1.0
    # Voltage angle is not part of the state estimation, therefore it is not considered
    for bus in busses:
        for measurement in measurements:
            if measurement["UserInformation"]["ConsumerID"] == bus:
                measurement["MeasurementData"]["ActivePower"] = np.random.uniform(10e-06, 10e-02)
                measurement["MeasurementData"]["ReactivePower"] = np.random.uniform(-10e-05, 10e-05)
                measurement["MeasurementData"]["Voltage"] = np.random.uniform(0.95, 1.05)


def uninformed_fdia(busses, measurements):
    # Select the JSON object from the list where the "ConsumerID" matches the bus
    # Add values to the ActivePower, ReactivePower and Voltage that try to make the system unstable
    # ActivePower and ReactivePower should be high, Voltage should be low
    # The FDIA should try to bypass the bad data detection
    for bus in busses:
        for measurement in measurements:
            if measurement["UserInformation"]["ConsumerID"] == bus:
                measurement["MeasurementData"]["ActivePower"] = 0.3
                measurement["MeasurementData"]["ReactivePower"] = -0.005
                measurement["MeasurementData"]["Voltage"] = 0.999


def random_fdia_liu(busses, measurements,  net):
    # Random FDIA attack using the Liu method
    # Requires at least 6 attack busses to work
    H = net._ppc["internal"]["J"].todense()
    I_meter = net.bus.index.to_list()
    I_meter.remove(129) # Remove transformer MV bus
    for bus in busses:
        I_meter.remove(bus)
    print(I_meter)
    m, n = H.shape
    # Convert H matrix to a floating type for numerical stability
    H = H.astype(float)
    for j in I_meter:
        # Find a column where the j-th element is not zero
        swap_col = -1
        for i in range(n):
            if H[j, i] != 0:
                swap_col = i
                break

        if swap_col == -1:
            # If no such column is found, continue to the next j
            continue

        # Swap the found column with the first column
        H[:, [0, swap_col]] = H[:, [swap_col, 0]]

        # Reduce columns to zero out the j-th element
        for i in range(1, n):
            if H[j, i] != 0:
                factor = H[j, i] / H[j, 0]
                H[:, i] = H[:, i] - factor * H[:, 0]

        # After processing all bar_I_meter indices, find a suitable attack vector
        # An attack vector can be any non-zero column that has zero elements in indices of bar_I_meter.
    for i in range(n):
        column = H[:, i]
        if all(column[j] == 0 for j in I_meter):
            # Ensure it's a non-zero vector
            if np.any(column != 0):
                break
    for bus in busses:
        for measurement in measurements:
            if measurement["UserInformation"]["ConsumerID"] == bus:
                measurement["MeasurementData"]["ActivePower"] = column.item(bus, 0)
                measurement["MeasurementData"]["ReactivePower"] = column.item(bus, 0)
    return measurements



def plot_differences(correct_data, fdia_data):
    # Plot the differences between the correct and the FDIA data
    # The differences are calculated as the difference between the correct and the FDIA data
    # The differences are then plotted for each bus
    differences = compute_differences(correct_data, fdia_data)
    print("Average Differences in %: ")
    print(differences.mean())
    differences.iloc[0:42].plot(subplots=True,xlabel="Bus Number", ylabel="Difference in %",
                     title=["Voltage Difference", "Active Power Difference",
                            "Reactive Power Difference", "Voltage Angle Difference"])
    plt.show()
    return differences


def compute_differences(correct_data, fdia_data):
    # Computes the percentage differences between the correct and the FDIA data and puts them in the differences dataframe
    # The differences are calculated as (FDIA - Correct) / Correct * 100
    differences = pd.DataFrame()
    for column in correct_data.columns:
        differences[f"d_{column}"] = ((fdia_data[column] - correct_data[column]) / correct_data[column]) * 100
    return differences


def plot_attack(net, attack_buses):
    # Take the geodata of the attack_buses from net["bus_geodata"] and plot them as red dots
    # Take the geodata of the rest of the buses from net["bus_geodata"] and plot them as blue dots
    # Takes the start and end points of the lines "from_bus" and "to_bus" from net["line"] and plots them
    bus_geodata = net["bus_geodata"]
    attack_geodata = bus_geodata.loc[attack_buses]
    rest_geodata = bus_geodata.drop(attack_buses)
    line_geodata = net["line"]
    for line in line_geodata.iterrows():
        x0 = bus_geodata.loc[line[1]["from_bus"]]["x"]
        y0 = bus_geodata.loc[line[1]["from_bus"]]["y"]
        x1 = bus_geodata.loc[line[1]["to_bus"]]["x"]
        y1 = bus_geodata.loc[line[1]["to_bus"]]["y"]
        plt.plot([x0, x1], [y0, y1], color="black")
    plt.scatter(rest_geodata["x"], rest_geodata["y"], color="blue")
    plt.scatter(attack_geodata["x"], attack_geodata["y"], color="red")
    # Add the bus number next to each bus and move them a bit away from the bus
    for bus in bus_geodata.iterrows():
        plt.text(bus[1]["x"], bus[1]["y"], bus[0])
    plt.show()



