import subprocess

scripts_in_order = [
    'get_held_out_compound_data.py',
    'held_out_linear_probing.py',
    'plot_held_out_comparison.py'
]

for script in scripts_in_order:
    subprocess.check_call(['python', script, '-c', 'config_held_out.yaml'])