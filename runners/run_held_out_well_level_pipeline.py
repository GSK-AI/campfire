import subprocess

scripts_in_order = [
    'data_collection/get_aggregate_data.py',
    'modelling/nearest_neighbour_classifier.py',
    'plotting/plot_held_out_well_level.py',
]

for script in scripts_in_order:
    subprocess.check_call(['python', script, '-c', 'configs/config_held_out_well_level.yaml'])