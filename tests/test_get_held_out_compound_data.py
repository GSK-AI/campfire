
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open
from data_collection.get_held_out_compound_data import get_probing_data, main

class TestGetHeldOutCompoundData(unittest.TestCase):

    def setUp(self):
        self.plates = pd.DataFrame({
            'ROW': [1, 2, 3],
            'COLUMN': [1, 2, 3],
            'split': ['test_out_compound_JCP1', 'test_out_all_JCP2', np.nan],
            'last_layer_1': [0.1, 0.2, 0.3],
            'last_layer_2': [0.4, 0.5, 0.6]
        })
        self.controls = pd.DataFrame({
            0: ['test_out_compound_JCP1', 'test_out_all_JCP2', np.nan],
            1: [np.nan, 'test_out_compound_JCP1', 'test_out_all_JCP2'],
            2: ['test_out_all_JCP2', np.nan, 'test_out_compound_JCP1']
        })
        self.num_samples = 1
        self.last_layer = 'layer'
        self.seed = 42

    def test_get_probing_data(self):
        result = get_probing_data(self.plates, self.controls, self.num_samples, self.last_layer, self.seed)
        self.assertFalse(result.empty)
        self.assertIn('compound', result.columns)
        self.assertIn('split', result.columns)

if __name__ == "__main__":
    unittest.main()