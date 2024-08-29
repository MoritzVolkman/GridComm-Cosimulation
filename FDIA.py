import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import simbench as sb
import pandapower
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from deap import base, creator, tools, algorithms

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
    return measurements


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


def random_fdia_liu(busses, measurements,  net, H):
    # Random FDIA attack using the Liu method
    # Requires at least 6 attack busses to work
    I_meter = net.bus.index.to_list()
    I_meter.remove(129) # Remove transformer MV bus
    for bus in busses:
        I_meter.remove(bus)
    # print(I_meter)
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
                measurement["MeasurementData"]["ActivePower"] += column.item(bus, 0)
                measurement["MeasurementData"]["ReactivePower"] += column.item(bus, 0)
    return measurements


def deep_learning_fdia_build_dataset(measurements, original_vals, net):
    # Get the voltage, ActivePower and ReactivePower for each bus from measurements
    # And put them in a format like this: [voltage0, active_power0, reactive_power0, voltage1,...]
    # Add the net.bus.state_est values to the same row as o_voltage0, o_active_power0, o_reactive_power0,...]
    # Use the deep learning model to predict the correct values for the FDIA data
    # Add the predicted values to the measurements
    row = []
    for bus in net.res_bus.index.to_list():
        if bus == 129:
            continue
        for measurement in measurements:
            if measurement["UserInformation"]["ConsumerID"] == bus:
                row.append(measurement["MeasurementData"]["Voltage"]-original_vals.loc[bus]["vm_pu"])
                row.append(measurement["MeasurementData"]["ActivePower"]-original_vals.loc[bus]["p_mw"])
                row.append(measurement["MeasurementData"]["ReactivePower"]-original_vals.loc[bus]["q_mvar"])
    for bus in net.res_bus.index.to_list():
        if bus == 129:
            continue
        row.append(net.res_bus_est.loc[bus]["vm_pu"])
        row.append(net.res_bus_est.loc[bus]["va_degree"])
        row.append(net.res_bus_est.loc[bus]["p_mw"])
        row.append(net.res_bus_est.loc[bus]["q_mvar"])
    return row


def deep_learning_fdia_train_model():
    # Load the dataset
    data = pd.read_csv('dataset.csv')

    # Find upper and lower limits for the input variables to pass bad data detection
    # Extract columns for V, P, and Q
    V_columns = [col for col in data.columns if col.startswith('V')]
    P_columns = [col for col in data.columns if col.startswith('P')]
    Q_columns = [col for col in data.columns if col.startswith('Q')]

    # Calculate the mean and max for each set of columns
    V_min = data[V_columns].min().min()
    V_max = data[V_columns].max().max()

    P_min = data[P_columns].min().min()
    P_max = data[P_columns].max().max()

    Q_min = data[Q_columns].min().min()
    Q_max = data[Q_columns].max().max()

    bounds = [
        (V_min, V_max), (P_min, P_max), (Q_min, Q_max),
        (V_min, V_max), (P_min, P_max), (Q_min, Q_max),
        (V_min, V_max), (P_min, P_max), (Q_min, Q_max),
        (V_min, V_max), (P_min, P_max), (Q_min, Q_max),
        (V_min, V_max), (P_min, P_max), (Q_min, Q_max),
        (V_min, V_max), (P_min, P_max), (Q_min, Q_max)
    ]

    # Select input features for buses 0, 1, 2, 8, 9, and 40
    input_columns = [
        'V0', 'P0', 'Q0',
        'V1', 'P1', 'Q1',
        'V2', 'P2', 'Q2',
        'V8', 'P8', 'Q8',
        'V9', 'P9', 'Q9',
        'V40', 'P40', 'Q40'
    ]

    # Select the entire output for all nodes if needed
    # Modify these according to your requirements, or only select the desired outputs
    output_columns = [
                         f'V_OUT{i}' for i in range(43)
                     ] + [
                         f'PHI_OUT{i}' for i in range(43)
                     ] + [
                         f'P_OUT{i}' for i in range(43)
                     ] + [
                         f'Q_OUT{i}' for i in range(43)
                     ]

    # Extract input and output data
    X_selected = data[input_columns].values
    y = data[output_columns].values

    # Preprocessing
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X_selected)
    y_scaled = scaler_y.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Build the neural network model
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(y_train.shape[1])  # output layers matches the number of output variables
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

    return model, bounds


def deep_learning_fdia_predict(model, bounds):
    # Use the pre-trained model to create the best possible input for the FDIA attack
    def evaluate(individual):
        input_array = np.array(individual).reshape(1, -1)
        prediction = model.predict(input_array)
        return np.max(prediction),

    def bounded_attr(min_val, max_val):
        return lambda: np.random.uniform(min_val, max_val)

    # Create classes for the optimization problem
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Adjust toolbox for individual creation within bounds
    toolbox = base.Toolbox()
    for i, (min_val, max_val) in enumerate(bounds):
        toolbox.register(f"attr_float_{i}", bounded_attr(min_val, max_val))

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        [toolbox.__getattribute__(f"attr_float_{i}") for i in range(len(bounds))],
        n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    def bounded_mutate(individual, indpb):
        for i, (min_val, max_val) in enumerate(bounds):
            if np.random.rand() < indpb:
                old_value = individual[i]
                new_value = np.random.normal(old_value, 0.1)
                new_value = np.clip(new_value, min_val, max_val)
                individual[i] = new_value
        return individual,

    toolbox.register("mutate", bounded_mutate, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Genetic Algorithm settings
    population = toolbox.population(n=300)
    ngen = 40
    cxpb = 0.5  # Crossover probability
    mutpb = 0.2  # Mutation probability

    # Run the genetic algorithm
    algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=True)

    # Extract the best individual
    best_individual = tools.selBest(population, k=1)[0]
    optimal_output = model.predict(np.array(best_individual).reshape(1, -1))

    print("Optimal Inputs (via GA):", list(best_individual))
    print("Optimal Output (via GA):", optimal_output)
    return list(best_individual)

def deep_learning_fdia_inject(att_vector, busses, measurements):
    av = att_vector.copy()
    for bus in busses:
        for measurement in measurements:
            if measurement["UserInformation"]["ConsumerID"] == bus:
                measurement["MeasurementData"]["ActivePower"] = av.pop()
                measurement["MeasurementData"]["ReactivePower"] = av.pop()
                measurement["MeasurementData"]["Voltage"] = av.pop()
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


if __name__ == "__main__":
    attack_vectors = []
    for i in range(9):
        model, bounds = deep_learning_fdia_train_model()
        att_vector = deep_learning_fdia_predict(model, bounds)
        attack_vectors.append(att_vector)
    print(attack_vectors)