import subprocess

scripts_in_order = [
    'data_collection/get_linear_probe_data.py',
    'modelling/linear_probing.py',
]

for script in scripts_in_order:
    subprocess.check_call(['python', script, '-c', 'config.yaml'])