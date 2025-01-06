import subprocess

scripts_in_order = [
    'data_collection/get_linear_probe_data.py',
    'modelling/linear_probing.py',
]

for script in scripts_in_order:
    subprocess.check_call(['python', script, '-c', 'configs/unseen_channel/dino_vits8.yaml'])