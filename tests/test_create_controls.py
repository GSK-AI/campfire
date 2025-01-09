import unittest
import pandas as pd
import numpy as np
from create_controls import set_split, get_held_out_compounds, split_target_plates, split_compound_plates

class TestCreateControls(unittest.TestCase):

    def test_set_split(self):
        row = {"Metadata_JCP2022": "compound_1"}
        held_out_compounds = ["compound_1", "compound_2"]
        self.assertEqual(set_split(row, held_out_compounds), 'test_out_all_')
        row = {"Metadata_JCP2022": "compound_3"}
        self.assertEqual(set_split(row, held_out_compounds), 'test_out_plate_')

    def test_get_held_out_compounds(self):
        compound_list = np.array(["compound_1", "compound_2", "compound_3", "compound_4"])
        control_list = ["compound_1"]
        N_held_out = 2
        held_out_compounds = get_held_out_compounds(compound_list, control_list, N_held_out)
        self.assertEqual(len(held_out_compounds), N_held_out)
        self.assertNotIn("compound_1", held_out_compounds)

    def test_split_target_plates(self):
        wells = pd.DataFrame({
            "Metadata_JCP2022": ["compound_1", "compound_2", "compound_3", "compound_4", "compound_5"] * 25,
            "Metadata_Plate": ["plate_" + str(i) for i in range(1, 26) for _ in range(5)]
        })
        control_list = ["compound_1"]
        N_held_out = 1
        hold_out_target_plates = ["plate_1", "plate_2", "plate_3", "plate_4", "plate_5"]
        seed = 42
        wells, held_out_compounds = split_target_plates(wells, control_list, N_held_out, hold_out_target_plates, seed, expected_num_compounds=5, Ntrain=5, Nval=2)
        self.assertIn("split", wells.columns)
        self.assertEqual(len(held_out_compounds), N_held_out)
        
        # Ensure that hold_out_target_plates are assigned correct splits
        for plate in hold_out_target_plates:
            plate_wells = wells[wells["Metadata_Plate"] == plate]
            self.assertTrue(all(plate_wells["split"].isin(['test_out_all_', 'test_out_plate_'])))

    def test_split_compound_plates(self):
        wells = pd.DataFrame({
            "Metadata_JCP2022": ["compound_1", "compound_2", "compound_3", "compound_4"]*10,
            "Metadata_Plate": ["plate_" + str(i) for i in range(1, 11) for _ in range(4)]
        })
        held_out_compounds = ["compound_1"]
        frac_compound_wells_in_train = 0.5
        wells = split_compound_plates(wells, held_out_compounds, frac_compound_wells_in_train,exp_num_compounds=4)
        self.assertIn("split", wells.columns)

if __name__ == '__main__':
    unittest.main()