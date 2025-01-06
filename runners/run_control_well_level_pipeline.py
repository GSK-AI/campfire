import subprocess

scripts_in_order = [
    'data_collection/get_aggregate_control_data.py',
    'modelling/control_nearest_neighbour_classifier.py',
    'plotting/plot_control_well_level.py',
]

for script in scripts_in_order:
    subprocess.check_call(['python', script, '-c', 'configs/config_control_well_level.yaml'])