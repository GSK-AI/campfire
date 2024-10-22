import subprocess

scripts_in_order = [
    'get_linear_probe_data.py',
    'linear_probing.py',
]

for script in scripts_in_order:
    subprocess.check_call(['python', script, '-c', 'config.yaml'])