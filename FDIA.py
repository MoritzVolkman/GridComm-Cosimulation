import numpy as np

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


def informed_fdia(busses, measurements):
    pass
