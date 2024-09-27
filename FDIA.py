import time

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
    pass

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


def calculate_tau_a(net):
    # Calculate total active power demand in the network
    total_load = net.load['p_mw'].sum()

    # Set tau_a as a fraction of the total load, e.g., 5% of total load
    tau_a = total_load * 0.05
    # tau_a = total_load * 0.5
    return tau_a


def random_generalized_fdia_liu(busses, measurements, net, H):
    # Calculate tau_a from the network
    tau_a = calculate_tau_a(net)

    # Prepare the index of meters
    I_meter = net.bus.index.to_list()
    I_meter.remove(129)  # Remove transformer MV bus
    for bus in busses:
        I_meter.remove(bus)  # Remove attacked buses

    m, n = H.shape
    H = H.astype(float)  # Ensure numerical stability
    B_prime = np.copy(H)  # Start with H, which will be reduced to B'

    for j in I_meter:
        swap_col = -1
        for i in range(n):  # Find a column with a non-zero j-th element
            if B_prime[j, i] != 0:
                swap_col = i
                break

        if swap_col == -1:
            continue  # Skip if no column to swap

        # Swap the found column with the first column
        B_prime[:, [0, swap_col]] = B_prime[:, [swap_col, 0]]

        # Zero out the j-th element for each column
        for i in range(1, n):
            if B_prime[j, i] != 0:
                factor = B_prime[j, i] / B_prime[j, 0]
                B_prime[:, i] -= factor * B_prime[:, 0]

    # Create target vector t ensuring consistency
    t = np.zeros(m)
    k = len(I_meter)
    random_indices = np.random.choice(range(m), k, replace=False)
    t[random_indices] = np.random.rand(k) * tau_a

    # Solve for a'
    Bt = B_prime @ t
    B_prime_inv = np.linalg.pinv(B_prime)  # Pseudo-inverse
    d = np.random.rand(n)  # Random vector in case of null-space solution

    a_prime = B_prime_inv @ Bt + (np.eye(n) - B_prime_inv @ B_prime) @ d

    # Apply attack vector a_prime to measurements
    for bus in busses:
        for measurement in measurements:
            if measurement["UserInformation"]["ConsumerID"] == bus:
                index = net.bus.index.get_loc(bus)
                measurement["MeasurementData"]["ActivePower"] += a_prime[index]
                measurement["MeasurementData"]["ReactivePower"] += a_prime[index]

    return measurements


def targeted_generalized_fdia_liu(busses, measurements, net, H):
    # Example calculation of tau_a
    total_load = np.sum([measurement["MeasurementData"]["ActivePower"] for measurement in measurements])
    tau_a = total_load * 0.5

    I_meter = net.bus.index.to_list()
    if 129 in I_meter:
        I_meter.remove(129)  # Remove transformer MV bus
    for bus in busses:
        if bus in I_meter:
            I_meter.remove(bus)  # Remove attacked buses

    H = H.astype(float)
    m, n = H.shape

    # Basic transformation construction
    try:
        for idx in sorted(I_meter):
            column_found = False
            for col in range(n):
                if H[idx, col] != 0:
                    H[:, [0, col]] = H[:, [col, 0]]
                    column_found = True
                    break

            if not column_found:
                continue

            for col in range(1, n):  # Start from 1 since we just swapped column 0
                if H[idx, col] != 0:
                    factor = H[idx, col] / H[idx, 0]
                    H[:, col] -= factor * H[:, 0]

    except Exception as e:
        print(f"Error in transforming H: {e}")

    # Create attack vector b, and ensure norm conditions
    b = np.zeros(m)
    k = len(I_meter)
    random_indices = np.random.choice(range(m), k, replace=False)
    b[random_indices] = np.random.rand(k)

    norm_b = np.linalg.norm(b)
    if norm_b <= tau_a:
        t = np.zeros(m)
        t[random_indices] = b[random_indices]
    else:
        return measurements

    tw_plus_b = t + b

    B_s_tw_plus_b = H @ tw_plus_b

    try:
        a_prime = np.linalg.pinv(H) @ B_s_tw_plus_b
    except np.linalg.LinAlgError as e:
        print(f"Matrix inversion error: {e}")
        return measurements

    # Apply to measurement attack vectors
    for bus in busses:
        for measurement in measurements:
            if measurement["UserInformation"]["ConsumerID"] == bus:
                measurement["MeasurementData"]["ActivePower"] += a_prime[net.bus.index.get_loc(bus)]
                measurement["MeasurementData"]["ReactivePower"] += a_prime[net.bus.index.get_loc(bus)]

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

    # Set the min and max bounds for the input features
    V_min = 0.95
    V_max = 1.05

    P_min = 10e-06
    P_max = 10e-02

    Q_min = -10e-05
    Q_max = 10e-05

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

    def repair(individual):
        for i in range(len(individual)):
            min_val, max_val = bounds[i]
            if individual[i] < min_val:
                individual[i] = min_val
            elif individual[i] > max_val:
                individual[i] = max_val
        return individual

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
                individual[i] += np.random.normal(0, 0.1)  # Can adjust the scale
                # Ensure the mutated value is within bounds
                individual[i] = np.clip(individual[i], min_val, max_val)
        return individual,

    toolbox.register("mutate", bounded_mutate, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Genetic Algorithm settings
    population = toolbox.population(n=300)
    ngen = 40
    cxpb = 0.5  # Crossover probability
    mutpb = 0.2  # Mutation probability

    # Define function to repair and apply it after crossover and mutation
    def repair_population(pop):
        for ind in pop:
            repair(ind)

    # Run the genetic algorithm
    for gen in range(ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)

        repair_population(offspring)  # Ensure offspring respects bounds

        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, len(population))

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
    # Compute the differences
    differences = compute_differences(correct_data, fdia_data)

    # Compute the mean of each column
    mean_differences = differences.mean()
    print("Average Differences in %: ")
    print(mean_differences)
    # Plot the differences
    axes = differences.iloc[0:42].plot(
        subplots=True,
        xlabel="Bus Number",
        ylabel="Difference in %",
        figsize=(12.8, 7.2),
        title=["Voltage Difference", "Voltage Angle Difference", "Active Power Difference", "Reactive Power Difference"]
    )
    # Adjust each subplot to have y-axis ticks and labels on the right, and add the mean lines and annotations
    for i, ax in enumerate(axes):
        # Adjust y-axis
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("left")
        # Add a horizontal line for the mean
        ax.axhline(y=mean_differences.iloc[i], color='k', linestyle='--', linewidth=1.5)
        # Annotate the mean value
        ax.text(
            0.95, 0.95, f'Mean: {round(mean_differences.iloc[i], 3):.2f}%',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.5)
        )
    plt.tight_layout()
    plt.show()
    return differences


def plot_mean_and_std(differences_list):
    # Combine all differences into a single DataFrame
    all_differences = pd.concat(differences_list, keys=range(len(differences_list)), names=['Timestep'])
    list_of_colors = ["b", "tab:orange", "g", "r"]
    list_of_labels = ["Voltage Magnitude", "Voltage Angle", "Active Power", "Reactive Power"]

    # Compute mean and standard deviation per node and measurement type
    means = all_differences.groupby(level=1).mean()  # Mean over timesteps
    stds = all_differences.groupby(level=1).std()  # Standard deviation over timesteps

    # Overall mean for each measurement type for all nodes
    overall_means = means.mean()

    # Find the highest absolute value of the means and the corresponding standard deviation
    highest_abs_value_info = {}

    # Plot settings
    measurement_types = means.columns
    num_measurement_types = len(measurement_types)
    fig, axes = plt.subplots(nrows=num_measurement_types, figsize=(12.8, 7.2), sharex=True)

    if num_measurement_types == 1:
        axes = [axes]  # Ensure axes is always a list, even if there's only one subplot

    # Plot each measurement type
    for i, measurement in enumerate(measurement_types):
        ax = axes[i]
        mean_values = means.iloc[0:42][measurement]
        std_values = stds.iloc[0:42][measurement]

        # Calculate highest absolute value of mean differences
        highest_abs_value = mean_values.abs().max()
        bus_index = mean_values.abs().idxmax()
        standard_dev_bus = std_values.loc[bus_index]

        highest_abs_value_info[measurement] = (bus_index, highest_abs_value, standard_dev_bus)

        ax.plot(mean_values.index, mean_values, label=f'Mean', color=list_of_colors[i])
        ax.fill_between(mean_values.index, mean_values - std_values, mean_values + std_values,
                        alpha=0.3, color=list_of_colors[i], label=f'Standard Deviation')

        # Plot the overall mean as a dotted line
        overall_mean = overall_means[measurement]
        ax.axhline(overall_mean, color='black', linestyle=':', linewidth=1, label='Overall Mean')

        # Get rid of bug that shows overall means as large numbers although they are clearly zero
        if overall_mean > 1.0e10 or overall_mean < -1.0e10:
            overall_mean = 0
        # Annotate the overall mean value in the plot
        ax.text(0.95, 0.95, f'Overall mean: {overall_mean:.5f}%', transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

        # Customize plot
        ax.set_ylabel('% Difference')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("left")
        ax.set_title(list_of_labels[i])
        ax.legend(loc='lower left')  # Set legend to the bottom left corner

    ax.set_xlabel('Bus Number')
    plt.tight_layout()
    plt.show()

    # Print out the highest absolute value info for each measurement type
    for measurement, (bus_index, highest_abs_value, standard_dev_bus) in highest_abs_value_info.items():
        print(f"Highest absolute value of nodal mean deviations for {measurement} at bus {bus_index}, "
              f"deviation: {highest_abs_value:.5f} +- {standard_dev_bus:.5f}")

    return overall_means



def compute_differences(correct_data, fdia_data, epsilon=1e-6):
    # Computes the percentage differences between the correct and the FDIA data
    # The differences are calculated as (FDIA - Correct) / Correct * 100
    # Epsilon is a small value to prevent division by very small numbers
    differences = pd.DataFrame()

    for column in correct_data.columns:
        # Avoid division by zero or near-zero by using np.where
        denominator = correct_data[column].replace(0, epsilon)
        differences[f"d_{column}"] = ((fdia_data[column] - correct_data[column]) / denominator) * 100

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

