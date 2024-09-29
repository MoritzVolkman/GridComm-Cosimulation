import unittest
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import pandapower.networks as nw
import sys
sys.path.append("..")
from GridSim import (create_measurement, parse_measurement, run_state_estimation, send_to_network_sim, apply_absolute_values)


class TestGridSimulation(unittest.TestCase):

    def setUp(self):
        # This setup runs before each test
        # Create a mock network to use for pandapower
        self.net = nw.example_multivoltage()
        self.net.trafo.tap_pos = 1

    def test_create_measurement_full(self):
        # Test create_measurement function with full measurements
        result = create_measurement(self.net, amount=1)

        # Assert that we have a measurement for each bus
        self.assertEqual(len(result), len(self.net.bus))
        for measurement in result:
            self.assertIn("MeasurementData", measurement)
            self.assertIn("UserInformation", measurement)
            self.assertIn("Voltage", measurement["MeasurementData"])

    def test_create_measurement_partial(self):
        # Test create_measurement with zero random threshold
        result = create_measurement(self.net, amount=0)

        # Assert that no measurements are created
        self.assertEqual(len(result), 0)

    def test_parse_measurement(self):
        # Create mock measurement data
        measurements = [{
            "MeasurementData": {"Voltage": 1.0, "ActivePower": 0.5, "ReactivePower": 0.3},
            "UserInformation": {"ConsumerID": 0}
        }]
        initial_measurement_count = len(self.net.measurement)

        # Run parse_measurement
        parse_measurement(measurements, self.net)

        # Check if new measurements are added
        self.assertEqual(len(self.net.measurement), initial_measurement_count + 3)

    @patch('pandapower.estimation.estimate')
    def test_run_state_estimation(self, mock_estimate):
        mock_estimate.return_value = None  # no return value, just ensure the function is called

        # Test with no bad data detected
        run_state_estimation(self.net)
        self.assertTrue(mock_estimate.called)

    @patch('your_module.Network.send_message')
    def test_send_to_network_sim(self, mock_send_message):
        # Mock data
        SMGW_data = [{
            "MeasurementData": {"Voltage": 1.0, "ActivePower": 0.5, "ReactivePower": 0.3},
            "UserInformation": {"ConsumerID": 0}
        }]
        timestep = 0

        send_to_network_sim(SMGW_data, timestep)

        # Check if the function was called with correct parameters
        self.assertTrue(mock_send_message.called)
        args, kwargs = mock_send_message.call_args
        self.assertEqual(args[0], '127.0.0.1')
        self.assertEqual(args[1], 8081)

    def test_apply_absolute_values(self):
        # Create a multi-index dataframe to act as profile data
        index = pd.MultiIndex.from_tuples([('load', 'q_mvar'), ('load', 'p_mw')], names=['element', 'parameter'])
        profiles_data = pd.DataFrame(data=[[0.1, 0.2], [0.1, 0.2]], columns=[0, 1], index=index)

        # Test application to a timestep
        apply_absolute_values(self.net, profiles_data, case_or_time_step=1)

        # Check if the network data has been updated correctly
        np.testing.assert_almost_equal(self.net['load']['q_mvar'].values, [0.2, 0.2])


if __name__ == '__main__':
    unittest.main()