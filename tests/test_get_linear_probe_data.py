import unittest
import pandas as pd
from data_collection.get_linear_probe_data import get_probing_data

class TestGetLinearProbeData(unittest.TestCase):

    def setUp(self):
        # Setup mock data for testings
        self.plates = pd.DataFrame({
            'ROW': [1, 2, 3],
            'COLUMN': [1, 2, 3],
            'last_layer_1': [0.1, 0.2, 0.3],
            'last_layer_2': [0.4, 0.5, 0.6]
        })
        self.controls = pd.DataFrame({
            '1': ['compound_1_JCP2022_033924', 'compound_2_JCP2022_085227', 'compound_3_JCP2022_037716'],
            '2': ['compound_4_JCP2022_025848', 'compound_5_JCP2022_046054', 'compound_6_JCP2022_035095'],
            '3': ['compound_7_JCP2022_064022', 'compound_8_JCP2022_050797', 'compound_9_JCP2022_012818']
        })
        self.control_ids = ['2022_033924', '2022_046054', '2022_012818']
        self.num_samples = 1
        self.last_layer = 'last_layer'
        self.seed = 42

    def test_get_probing_data(self):
        result = get_probing_data(self.plates, self.controls, self.control_ids, self.num_samples, self.last_layer, self.seed)
        print(result,flush=True)
        self.assertEqual(len(result), 3)
        self.assertIn('TARGET', result.columns)
        self.assertIn('last_layer_1', result.columns)
        self.assertIn('last_layer_2', result.columns)
        # Check that only control ids appear in the result
        for control_id in result['compound']:
            self.assertIn(control_id, self.control_ids)

if __name__ == '__main__':
    unittest.main()