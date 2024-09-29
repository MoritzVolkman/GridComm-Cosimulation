import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
sys.path.append("..")
from FDIA import (
    random_fdia,
    random_fdia_liu,
    calculate_tau_a,
    compute_differences,
    deep_learning_fdia_build_dataset,
    deep_learning_fdia_train_model
)


class TestFDIAFunctionality(unittest.TestCase):

    def setUp(self):
        # Example network setup that might be used
        self.net = MagicMock()
        self.net.bus.index = pd.Index([0, 1, 2, 8, 9, 40, 129])
        self.net.load['p_mw'] = pd.Series([1.0, 2.0, 3.0])
        self.net.res_bus = pd.DataFrame({
            'vm_pu': np.random.rand(50),
            'va_degree': np.random.rand(50),
            'p_mw': np.random.rand(50),
            'q_mvar': np.random.rand(50),
        })
        self.net.res_bus_est = self.net.res_bus

        self.measurements = [{
            "MeasurementData": {
                "Voltage": 1.0,
                "ActivePower": 0.5,
                "ReactivePower": 0.3
            },
            "UserInformation": {
                "ConsumerID": 0
            }
        } for _ in range(10)]

    def test_random_fdia(self):
        attacked_measurements = random_fdia([0], self.measurements)
        for measurement in attacked_measurements:
            if measurement["UserInformation"]["ConsumerID"] == 0:
                self.assertTrue(10e-06 <= measurement["MeasurementData"]["ActivePower"] <= 10e-02)
                self.assertTrue(-10e-05 <= measurement["MeasurementData"]["ReactivePower"] <= 10e-05)
                self.assertTrue(0.95 <= measurement["MeasurementData"]["Voltage"] <= 1.05)

    def test_calculate_tau_a(self):
        tau_a = calculate_tau_a(self.net)
        expected_tau_a = self.net.load['p_mw'].sum() * 0.05
        self.assertAlmostEqual(tau_a, expected_tau_a, places=3)

    def test_compute_differences(self):
        correct_data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        fdia_data = pd.DataFrame({'A': [2, 1], 'B': [6, 5]})
        differences = compute_differences(correct_data, fdia_data)
        expected_differences = pd.DataFrame({'d_A': [100.0, -50.0], 'd_B': [100.0, 25.0]})
        pd.testing.assert_frame_equal(differences, expected_differences, check_less_precise=2)

    @patch('your_module.pd.read_csv')
    @patch('your_module.StandardScaler.fit_transform')
    @patch('your_module.train_test_split')
    @patch('your_module.keras.Sequential')
    def test_deep_learning_fdia_train_model(self, mock_sequential, mock_train_test_split, mock_fit_transform,
                                            mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame(np.random.rand(100, 20), columns=[f'{i}' for i in range(20)])
        mock_train_test_split.return_value = (
        np.random.rand(80, 18), np.random.rand(20, 18), np.random.rand(80, 172), np.random.rand(20, 172))
        mock_fit_transform.side_effect = lambda x: x

        model, bounds = deep_learning_fdia_train_model()

        self.assertIsNotNone(model)
        self.assertEqual(len(bounds), 18)  # Example check, depends on actual implementation

    def test_deep_learning_fdia_build_dataset(self):
        original_vals = self.net.res_bus.copy()
        row = deep_learning_fdia_build_dataset(self.measurements, original_vals, self.net)

        expected_length = 2 * len(self.net.res_bus.index.drop(129)) * 3
        self.assertEqual(len(row), expected_length)


if __name__ == '__main__':
    unittest.main()